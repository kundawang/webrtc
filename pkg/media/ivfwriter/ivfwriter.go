// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

// Package ivfwriter implements IVF media container writer
package ivfwriter

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
	"github.com/pion/rtp/codecs/av1/obu"
)

var (
	errFileNotOpened        = errors.New("file not opened")
	errInvalidNilPacket     = errors.New("invalid nil packet")
	errCodecUnset           = errors.New("codec is unset")
	errCodecAlreadySet      = errors.New("codec is already set")
	errNoSuchCodec          = errors.New("no codec for this MimeType")
	errInvalidMediaTimebase = errors.New("invalid media timebase")
	errEmptyFile            = errors.New("file is empty, not an IVF file")
	errInvalidIVFHeader     = errors.New("invalid IVF file header")
	errTruncatedFrame       = errors.New("IVF file ends with a truncated frame")
	errPTSRegression        = errors.New("frame PTS is not monotonically increasing")
	errEmptyFrame           = errors.New("frame payload must not be empty")
)

type (
	codec int

	// IVFWriter is used to take RTP packets and write them to an IVF on disk.
	IVFWriter struct {
		ioWriter     io.Writer
		count        uint64
		seenKeyFrame bool

		codec codec

		timebaseDenominator uint32
		timebaseNumerator   uint32
		firstFrameTimestamp uint32
		firstFrameSeen      bool
		clockRate           uint64
		videoWidth          uint16
		videoHeight         uint16

		directPTS bool

		// append state, zero valued for a freshly created file
		resuming       bool
		lastPTS        uint64
		ptsOffset      uint64
		ptsOffsetReady bool

		// VP8, VP9
		currentFrame []byte

		// AV1
		av1Depacketizer *codecs.AV1Depacketizer
	}
)

const (
	codecUnset codec = iota
	codecVP8
	codecVP9
	codecAV1

	mimeTypeVP8 = "video/VP8"
	mimeTypeVP9 = "video/VP9"
	mimeTypeAV1 = "video/AV1"

	ivfFileHeaderSize  = 32
	ivfFrameHeaderSize = 12
)

// New builds a new IVF writer.
func New(fileName string, opts ...Option) (*IVFWriter, error) {
	file, err := os.Create(fileName) //nolint:gosec
	if err != nil {
		return nil, err
	}

	return NewWith(file, opts...)
}

// NewWith initialize a new IVF writer with an io.Writer output.
func NewWith(out io.Writer, opts ...Option) (*IVFWriter, error) {
	if out == nil {
		return nil, errFileNotOpened
	}

	writer := &IVFWriter{
		ioWriter:            out,
		seenKeyFrame:        false,
		timebaseDenominator: 30,
		timebaseNumerator:   1,
		clockRate:           90000,
		videoWidth:          640,
		videoHeight:         480,
	}

	for _, o := range opts {
		if err := o(writer); err != nil {
			return nil, err
		}
	}

	if writer.codec == codecUnset {
		writer.codec = codecVP8
	}
	if err := writer.writeHeader(); err != nil {
		return nil, err
	}

	if writer.timebaseDenominator == 0 {
		return nil, errInvalidMediaTimebase
	}

	return writer, nil
}

// truncater is implemented by outputs that can discard the bytes after a
// given offset, e.g. *os.File.
type truncater interface {
	Truncate(size int64) error
}

// Append opens an existing IVF file for appending. Recording continues after
// the last complete frame already stored in the file and a new process can
// keep recording into the same file. On Close the frame count in the IVF
// header is rewritten with the total number of frames.
func Append(fileName string, opts ...Option) (*IVFWriter, error) {
	file, err := os.OpenFile(fileName, os.O_RDWR, 0o600) //nolint:gosec
	if err != nil {
		return nil, err
	}

	writer, err := AppendWith(file, opts...)
	if err != nil {
		_ = file.Close()

		return nil, err
	}

	return writer, nil
}

// AppendWith resumes an existing IVF file on a readable, writable and seekable
// output. The codec, dimensions, timebase, frame count and last frame PTS are
// recovered from the file before further RTP packets or frames are accepted.
//
// Existing frame bytes are never modified. Options that conflict with the file
// header result in an error and the writer is not created.
//
// If the file ends with a truncated frame (for example because the previous
// process crashed mid-write), the incomplete tail is removed when the output
// supports truncation and errTruncatedFrame is returned; calling AppendWith
// again then succeeds. The frame count stored in the header is treated as
// advisory: the frames actually present are counted and the header is
// corrected on Close.
func AppendWith(out io.ReadWriteSeeker, opts ...Option) (*IVFWriter, error) {
	if out == nil {
		return nil, errFileNotOpened
	}

	if _, err := out.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	header, err := readIVFHeader(out)
	if err != nil {
		return nil, err
	}

	configured, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	if err := validateConfig(configured, header); err != nil {
		return nil, err
	}

	writer := &IVFWriter{
		ioWriter:            out,
		codec:               header.codec,
		videoWidth:          header.width,
		videoHeight:         header.height,
		timebaseDenominator: header.timebaseDenominator,
		timebaseNumerator:   header.timebaseNumerator,
		clockRate:           90000,
		directPTS:           configured.directPTS,
	}
	if writer.timebaseDenominator == 0 {
		return nil, errInvalidMediaTimebase
	}

	frameCount, lastPTS, endOffset, scanErr := scanFrames(out)
	if scanErr != nil {
		handleTruncatedTail(out, scanErr, endOffset)

		return nil, scanErr
	}

	writer.resuming = true
	writer.count = frameCount
	writer.lastPTS = lastPTS
	writer.seenKeyFrame = frameCount > 0
	if _, err := out.Seek(endOffset, io.SeekStart); err != nil {
		return nil, err
	}

	return writer, nil
}

// applyOptions runs the options on a throwaway writer so the resulting
// configuration can be compared with the values stored in the file header.
func applyOptions(opts []Option) (*IVFWriter, error) {
	configured := &IVFWriter{}
	for _, option := range opts {
		if err := option(configured); err != nil {
			return nil, err
		}
	}

	return configured, nil
}

// validateConfig rejects options that disagree with the recovered header.
// Zero values mean "not configured" and therefore never conflict.
func validateConfig(configured *IVFWriter, header ivfHeaderInfo) error {
	for _, check := range []struct {
		name     string
		provided bool
		match    bool
	}{
		{name: "codec", provided: configured.codec != codecUnset, match: configured.codec == header.codec},
		{name: "width", provided: configured.videoWidth != 0, match: configured.videoWidth == header.width},
		{name: "height", provided: configured.videoHeight != 0, match: configured.videoHeight == header.height},
		{
			name:     "timebase denominator",
			provided: configured.timebaseDenominator != 0,
			match:    configured.timebaseDenominator == header.timebaseDenominator,
		},
		{
			name:     "timebase numerator",
			provided: configured.timebaseNumerator != 0,
			match:    configured.timebaseNumerator == header.timebaseNumerator,
		},
	} {
		if check.provided && !check.match {
			return fmt.Errorf("%w: %s", errInvalidIVFHeader, check.name)
		}
	}

	return nil
}

// handleTruncatedTail positions the output at the start of the incomplete
// tail and removes it when the output supports truncation.
func handleTruncatedTail(out io.ReadWriteSeeker, scanErr error, endOffset int64) {
	if !errors.Is(scanErr, errTruncatedFrame) {
		return
	}

	if t, ok := out.(truncater); ok {
		if err := t.Truncate(endOffset); err == nil {
			return
		}
	}

	// The output cannot be truncated: seeking to the last valid boundary at
	// least ensures a later write would overwrite the garbage, not extend it.
	_, _ = out.Seek(endOffset, io.SeekStart)
}

type ivfHeaderInfo struct {
	codec               codec
	width               uint16
	height              uint16
	timebaseDenominator uint32
	timebaseNumerator   uint32
}

func readIVFHeader(r io.Reader) (ivfHeaderInfo, error) {
	buffer := make([]byte, ivfFileHeaderSize)
	n, err := io.ReadFull(r, buffer)
	if err != nil {
		if errors.Is(err, io.EOF) {
			return ivfHeaderInfo{}, errEmptyFile
		}

		return ivfHeaderInfo{}, fmt.Errorf("%w: %w", errInvalidIVFHeader, err)
	}
	if n != ivfFileHeaderSize {
		return ivfHeaderInfo{}, errInvalidIVFHeader
	}

	header, err := parseIVFHeaderFields(buffer)
	if err != nil {
		return ivfHeaderInfo{}, err
	}

	return header, nil
}

func parseIVFHeaderFields(buffer []byte) (ivfHeaderInfo, error) {
	if string(buffer[0:4]) != "DKIF" {
		return ivfHeaderInfo{}, fmt.Errorf("%w: signature", errInvalidIVFHeader)
	}
	if binary.LittleEndian.Uint16(buffer[4:6]) != 0 {
		return ivfHeaderInfo{}, fmt.Errorf("%w: version", errInvalidIVFHeader)
	}
	if binary.LittleEndian.Uint16(buffer[6:8]) != ivfFileHeaderSize {
		return ivfHeaderInfo{}, fmt.Errorf("%w: header size", errInvalidIVFHeader)
	}

	info := ivfHeaderInfo{
		width:               binary.LittleEndian.Uint16(buffer[12:14]),
		height:              binary.LittleEndian.Uint16(buffer[14:16]),
		timebaseDenominator: binary.LittleEndian.Uint32(buffer[16:20]),
		timebaseNumerator:   binary.LittleEndian.Uint32(buffer[20:24]),
	}

	switch string(buffer[8:12]) {
	case "VP80":
		info.codec = codecVP8
	case "VP90":
		info.codec = codecVP9
	case "AV01":
		info.codec = codecAV1
	default:
		return ivfHeaderInfo{}, fmt.Errorf("%w: FourCC", errInvalidIVFHeader)
	}

	return info, nil
}

// scanFrames walks every complete frame starting at the current offset and
// returns the number of frames, the PTS of the last frame and the byte offset
// just past the last complete frame.
func scanFrames(r io.Reader) (count uint64, lastPTS uint64, endOffset int64, err error) {
	offset := int64(ivfFileHeaderSize)
	frameHeader := make([]byte, ivfFrameHeaderSize)

	for {
		_, readErr := io.ReadFull(r, frameHeader)
		if errors.Is(readErr, io.EOF) {
			return count, lastPTS, offset, nil
		} else if readErr != nil {
			return count, lastPTS, offset, fmt.Errorf("%w: %w", errTruncatedFrame, readErr)
		}

		frameSize := binary.LittleEndian.Uint32(frameHeader[0:4])
		pts := binary.LittleEndian.Uint64(frameHeader[4:12])
		if count > 0 && pts < lastPTS {
			return 0, 0, 0, errPTSRegression
		}

		//nolint:gosec // G115 frame sizes of real IVF files fit int64.
		frameEnd := offset + ivfFrameHeaderSize + int64(frameSize)
		if _, skipErr := io.CopyN(io.Discard, r, int64(frameSize)); skipErr != nil { //nolint:gosec
			return count, lastPTS, offset, fmt.Errorf("%w: %w", errTruncatedFrame, skipErr)
		}

		count++
		lastPTS = pts
		offset = frameEnd
	}
}

func (i *IVFWriter) writeHeader() error {
	header := make([]byte, ivfFileHeaderSize)
	copy(header[0:], "DKIF")                      // DKIF
	binary.LittleEndian.PutUint16(header[4:], 0)  // Version
	binary.LittleEndian.PutUint16(header[6:], 32) // Header size

	// FOURCC
	switch i.codec {
	case codecVP8:
		copy(header[8:], "VP80")
	case codecVP9:
		copy(header[8:], "VP90")
	case codecAV1:
		copy(header[8:], "AV01")
	default:
		return errCodecUnset
	}

	binary.LittleEndian.PutUint16(header[12:], i.videoWidth)          // Width in pixels
	binary.LittleEndian.PutUint16(header[14:], i.videoHeight)         // Height in pixels
	binary.LittleEndian.PutUint32(header[16:], i.timebaseDenominator) // Framerate denominator
	binary.LittleEndian.PutUint32(header[20:], i.timebaseNumerator)   // Framerate numerator
	binary.LittleEndian.PutUint32(header[24:], 900)                   // Frame count, will be updated on first Close() call
	binary.LittleEndian.PutUint32(header[28:], 0)                     // Unused

	_, err := i.ioWriter.Write(header)

	return err
}

func (i *IVFWriter) timestampToPts(timestamp uint64) uint64 {
	return timestamp * uint64(i.timebaseNumerator) / uint64(i.timebaseDenominator)
}

// WriteFrame appends a complete encoded frame with the given PTS in the IVF
// timebase units. It can be used independently of RTP depacketization and is
// the way to append frames while controlling the PTS directly. The PTS must
// not be smaller than the PTS of the previous frame.
func (i *IVFWriter) WriteFrame(frame []byte, pts uint64) error {
	if i.ioWriter == nil {
		return errFileNotOpened
	}
	if len(frame) == 0 {
		return errEmptyFrame
	}
	if i.count > 0 && pts < i.lastPTS {
		return errPTSRegression
	}

	if err := i.writeFrameAt(frame, pts); err != nil {
		return err
	}
	i.lastPTS = pts

	return nil
}

// writeFrameAt writes a frame header and payload. The caller is responsible for
// any PTS adjustment and validation.
func (i *IVFWriter) writeFrameAt(frame []byte, pts uint64) error {
	frameHeader := make([]byte, ivfFrameHeaderSize)
	//nolint:gosec // G115
	binary.LittleEndian.PutUint32(frameHeader[0:], uint32(len(frame))) // Frame length
	binary.LittleEndian.PutUint64(frameHeader[4:], pts)                // PTS

	i.count++

	if _, err := i.ioWriter.Write(frameHeader); err != nil {
		return err
	}
	_, err := i.ioWriter.Write(frame)

	return err
}

func (i *IVFWriter) writeFrame(frame []byte, timestamp uint64) error {
	var pts uint64
	if i.directPTS {
		// Direct PTS mode: use timestamp directly as PTS.
		pts = timestamp
	} else {
		// Existing behavior: convert using timebase.
		pts = i.timestampToPts(timestamp)
	}

	// A resumed recording receives a fresh RTP stream whose timestamps start
	// over. Shift the new PTS so it stays after the frames already on disk.
	if i.resuming && !i.ptsOffsetReady {
		if pts < i.lastPTS {
			i.ptsOffset = i.lastPTS - pts
		}
		i.ptsOffsetReady = true
	}
	pts += i.ptsOffset

	if err := i.writeFrameAt(frame, pts); err != nil {
		return err
	}
	i.lastPTS = pts

	return nil
}

// WriteRTP adds a new packet and writes the appropriate headers for it.
func (i *IVFWriter) WriteRTP(packet *rtp.Packet) error {
	if i.ioWriter == nil {
		return errFileNotOpened
	} else if len(packet.Payload) == 0 {
		return nil
	}

	if !i.firstFrameSeen {
		i.firstFrameTimestamp = packet.Timestamp
		i.firstFrameSeen = true
	}

	var timestamp uint64
	if i.directPTS {
		// Direct PTS mode: use RTP timestamp directly (no millisecond conversion).
		timestamp = uint64(packet.Timestamp - i.firstFrameTimestamp)
	} else {
		// Existing behavior: convert to milliseconds first.
		timestamp = 1000 * uint64(packet.Timestamp-i.firstFrameTimestamp) / i.clockRate
	}

	switch i.codec {
	case codecVP8:
		return i.writeVP8(packet, timestamp)
	case codecVP9:
		return i.writeVP9(packet, timestamp)
	case codecAV1:
		return i.writeAV1(packet, timestamp)
	default:
		return errCodecUnset
	}
}

func (i *IVFWriter) writeVP8(packet *rtp.Packet, timestamp uint64) error {
	vp8Packet := codecs.VP8Packet{}
	if _, err := vp8Packet.Unmarshal(packet.Payload); err != nil {
		return err
	}

	// a payload that carries only the VP8 descriptor has nothing to write.
	if len(vp8Packet.Payload) == 0 {
		return nil
	}

	isKeyFrame := (vp8Packet.Payload[0] & 0x01) == 0
	switch {
	case !i.seenKeyFrame && !isKeyFrame:
		return nil
	case i.currentFrame == nil && vp8Packet.S != 1:
		return nil
	}

	i.seenKeyFrame = true
	i.currentFrame = append(i.currentFrame, vp8Packet.Payload[0:]...)

	if !packet.Marker {
		return nil
	} else if len(i.currentFrame) == 0 {
		return nil
	}

	if err := i.writeFrame(i.currentFrame, timestamp); err != nil {
		return err
	}
	i.currentFrame = nil

	return nil
}

func (i *IVFWriter) writeVP9(packet *rtp.Packet, timestamp uint64) error {
	vp9Packet := codecs.VP9Packet{}
	if _, err := vp9Packet.Unmarshal(packet.Payload); err != nil {
		return err
	}

	switch {
	case !i.seenKeyFrame && vp9Packet.P:
		return nil
	case i.currentFrame == nil && !vp9Packet.B:
		return nil
	}

	i.seenKeyFrame = true
	i.currentFrame = append(i.currentFrame, vp9Packet.Payload[0:]...)

	if !packet.Marker {
		return nil
	} else if len(i.currentFrame) == 0 {
		return nil
	}

	// the timestamp must be sequential. webrtc mandates a clock rate of 90000
	// and we've assumed 30fps in the header.
	if err := i.writeFrame(i.currentFrame, timestamp); err != nil {
		return err
	}
	i.currentFrame = nil

	return nil
}

func (i *IVFWriter) writeAV1(packet *rtp.Packet, timestamp uint64) error {
	if i.av1Depacketizer == nil {
		i.av1Depacketizer = &codecs.AV1Depacketizer{}
	}

	payload, err := i.av1Depacketizer.Unmarshal(packet.Payload)
	if err != nil {
		return err
	}

	if !i.seenKeyFrame {
		isKeyFrame := i.av1Depacketizer.N || (len(payload) > 0 && obu.Type((payload[0]&0x78)>>3) == obu.OBUSequenceHeader)
		if !isKeyFrame {
			return nil
		}

		i.seenKeyFrame = true
	}

	i.currentFrame = append(i.currentFrame, payload...)
	if !packet.Marker {
		return nil
	}

	delimiter := obu.Header{
		Type:         obu.OBUTemporalDelimiter,
		HasSizeField: true,
	}
	frame := append(delimiter.Marshal(), 0)
	frame = append(frame, i.currentFrame...)

	if err := i.writeFrame(frame, timestamp); err != nil {
		return err
	}
	i.currentFrame = nil

	return nil
}

// Close stops the recording.
func (i *IVFWriter) Close() error {
	if i.ioWriter == nil {
		// Returns no error as it may be convenient to call
		// Close() multiple times
		return nil
	}

	defer func() {
		i.ioWriter = nil
	}()

	if ws, ok := i.ioWriter.(io.WriteSeeker); ok {
		// Update the framecount
		if _, err := ws.Seek(24, io.SeekStart); err != nil {
			return err
		}
		buff := make([]byte, 4)
		binary.LittleEndian.PutUint32(buff, uint32(i.count)) //nolint:gosec // G115
		if _, err := ws.Write(buff); err != nil {
			return err
		}
	}

	if closer, ok := i.ioWriter.(io.Closer); ok {
		return closer.Close()
	}

	return nil
}

// An Option configures a SampleBuilder.
type Option func(i *IVFWriter) error

// WithCodec configures if IVFWriter is writing AV1 or VP8 packets to disk.
func WithCodec(mimeType string) Option {
	return func(i *IVFWriter) error {
		if i.codec != codecUnset {
			return errCodecAlreadySet
		}

		switch mimeType {
		case mimeTypeVP8:
			i.codec = codecVP8
		case mimeTypeVP9:
			i.codec = codecVP9
		case mimeTypeAV1:
			i.codec = codecAV1
		default:
			return errNoSuchCodec
		}

		return nil
	}
}

func WithWidthAndHeight(width, height uint16) Option {
	return func(i *IVFWriter) error {
		i.videoWidth = width
		i.videoHeight = height

		return nil
	}
}

func WithFrameRate(numerator, denominator uint32) Option {
	return func(i *IVFWriter) error {
		i.timebaseNumerator = numerator
		i.timebaseDenominator = denominator

		return nil
	}
}

// WithDirectPTS enables direct use of RTP timestamps as PTS values
// without millisecond conversion.
//
// When this option is used, RTP timestamps are written directly as PTS values,
// preserving full timestamp precision. Use WithFrameRate to set the appropriate
// timebase (e.g., WithFrameRate(1, 90000) for standard 90kHz RTP clock).
//
// Example usage for standard RTP video (90kHz clock rate):
//
//	NewWith(file, WithFrameRate(1, 90000), WithDirectPTS())
func WithDirectPTS() Option {
	return func(i *IVFWriter) error {
		i.directPTS = true

		return nil
	}
}
