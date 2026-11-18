// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

// Package ivfreader implements IVF media container reader
package ivfreader

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	ivfFileHeaderSignature = "DKIF"
	ivfFileHeaderSize      = 32
	ivfFrameHeaderSize     = 12
)

var (
	errNilStream             = errors.New("stream is nil")
	errIncompleteFrameHeader = errors.New("incomplete frame header")
	errIncompleteFrameData   = errors.New("incomplete frame data")
	errIncompleteFileHeader  = errors.New("incomplete file header")
	errSignatureMismatch     = errors.New("IVF signature mismatch")
	errUnknownIVFVersion     = errors.New("IVF version unknown, parser may not parse correctly")
	errInvalidMediaTimebase  = errors.New("invalid media timebase")
	errSeekNotSupported      = errors.New("seek is not supported, stream does not implement io.Seeker")
	errSeekTargetOutOfRange  = errors.New("seek target is beyond the last frame")
)

// IVFFileHeader 32-byte header for IVF files
// https://wiki.multimedia.cx/index.php/IVF
type IVFFileHeader struct {
	signature           string // 0-3
	version             uint16 // 4-5
	headerSize          uint16 // 6-7
	FourCC              string // 8-11
	Width               uint16 // 12-13
	Height              uint16 // 14-15
	TimebaseDenominator uint32 // 16-19
	TimebaseNumerator   uint32 // 20-23
	NumFrames           uint32 // 24-27
	unused              uint32 // 28-31
}

// IVFFrameHeader 12-byte header for IVF frames
// https://wiki.multimedia.cx/index.php/IVF
type IVFFrameHeader struct {
	FrameSize uint32 // 0-3
	Timestamp uint64 // 4-11
}

// frameIndexEntry maps a frame timestamp to the absolute offset of its
// 12-byte frame header in the underlying stream. Frame payloads are never
// retained by the index.
type frameIndexEntry struct {
	timestamp uint64
	offset    int64
}

// IVFReader is used to read IVF files and return frame payloads.
type IVFReader struct {
	stream               io.Reader
	seeker               io.Seeker
	bytesReadSuccesfully int64
	timebaseDenominator  uint32
	timebaseNumerator    uint32
	frameIndex           []frameIndexEntry
	indexEndOffset       int64
	indexBuilt           bool
}

// NewWith returns a new IVF reader and IVF file header
// with an io.Reader input. Seek support is enabled when the stream also
// implements io.Seeker; otherwise ParseNextFrame keeps working while
// SeekToTimestamp reports errSeekNotSupported.
func NewWith(stream io.Reader) (*IVFReader, *IVFFileHeader, error) {
	if stream == nil {
		return nil, nil, errNilStream
	}

	reader := &IVFReader{
		stream: stream,
	}
	if seeker, ok := stream.(io.Seeker); ok {
		reader.seeker = seeker
	}

	header, err := reader.parseFileHeader()
	if err != nil {
		return nil, nil, err
	}
	if header.TimebaseDenominator == 0 || header.TimebaseNumerator == 0 {
		return nil, nil, errInvalidMediaTimebase
	}
	reader.timebaseDenominator = header.TimebaseDenominator
	reader.timebaseNumerator = header.TimebaseNumerator

	return reader, header, nil
}

// ResetReader resets the internal stream of IVFReader. This is useful
// for live streams, where the end of the file might be read without the
// data being finished.
func (i *IVFReader) ResetReader(reset func(bytesRead int64) io.Reader) {
	i.stream = reset(i.bytesReadSuccesfully)
	if seeker, ok := i.stream.(io.Seeker); ok {
		i.seeker = seeker
	} else {
		i.seeker = nil
	}
	// The replacement stream is a different source, so any cached index
	// must be discarded and rebuilt on the next SeekToTimestamp call.
	i.frameIndex = nil
	i.indexBuilt = false
	i.indexEndOffset = 0
}

// ptsToTimestamp converts a raw IVF presentation timestamp to the
// IVFFrameHeader.Timestamp time base using the file header time base.
// When the multiplication overflows uint64 the result saturates at
// math.MaxUint64 instead of wrapping: saturation keeps converted
// timestamps monotonic, which a wrapped value would not, so seek results
// stay well-defined even for extreme PTS values.
func (i *IVFReader) ptsToTimestamp(pts uint64) uint64 {
	denominator := uint64(i.timebaseDenominator)
	numerator := uint64(i.timebaseNumerator)
	if pts > math.MaxUint64/denominator {
		return math.MaxUint64
	}

	return pts * denominator / numerator
}

// ParseNextFrame reads from stream and returns IVF frame payload, header,
// and an error if there is incomplete frame data.
// Returns all nil values when no more frames are available.
func (i *IVFReader) ParseNextFrame() ([]byte, *IVFFrameHeader, error) {
	buffer := make([]byte, ivfFrameHeaderSize)
	var header *IVFFrameHeader

	bytesRead, err := io.ReadFull(i.stream, buffer)
	headerBytesRead := bytesRead
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, nil, errIncompleteFrameHeader
	} else if err != nil {
		return nil, nil, err
	}

	pts := binary.LittleEndian.Uint64(buffer[4:12])
	header = &IVFFrameHeader{
		FrameSize: binary.LittleEndian.Uint32(buffer[:4]),
		Timestamp: i.ptsToTimestamp(pts),
	}

	payload := make([]byte, header.FrameSize)
	bytesRead, err = io.ReadFull(i.stream, payload)
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, nil, errIncompleteFrameData
	} else if err != nil {
		return nil, nil, err
	}

	i.bytesReadSuccesfully += int64(headerBytesRead) + int64(bytesRead)

	return payload, header, err
}

// SeekToTimestamp positions the reader at the first frame whose timestamp is
// greater than or equal to target. The target uses the same converted time
// base as IVFFrameHeader.Timestamp, i.e. the raw IVF PTS scaled by
// TimebaseDenominator / TimebaseNumerator. The next ParseNextFrame call
// returns that frame.
//
// Seeking requires the underlying stream to implement io.Seeker. On a
// non-seekable stream the error is errSeekNotSupported and the reader is
// untouched. When no frame at or after target exists, including an empty
// file with zero frames, the error is errSeekTargetOutOfRange and the
// stream position is left unchanged, so a failed seek never disturbs an
// in-progress sequential parse. Repeated calls are independent: each one
// re-locates from the cached frame index without consuming frames.
func (i *IVFReader) SeekToTimestamp(target uint64) error {
	if i.seeker == nil {
		return errSeekNotSupported
	}

	if err := i.buildFrameIndex(); err != nil {
		return err
	}

	// Lower-bound search: the first entry with timestamp >= target. Equal
	// timestamps, including duplicate frame timestamps, resolve to the
	// earliest frame in file order.
	lo, hi := 0, len(i.frameIndex)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1) //nolint:gosec // G115, indices are small
		if i.frameIndex[mid].timestamp < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == len(i.frameIndex) {
		// buildFrameIndex restores the previous stream position, so a
		// failed seek leaves an in-progress sequential parse untouched.
		return errSeekTargetOutOfRange
	}

	offset := i.frameIndex[lo].offset
	if _, err := i.seeker.Seek(offset, io.SeekStart); err != nil {
		return err
	}
	i.bytesReadSuccesfully = offset

	return nil
}

// buildFrameIndex scans every frame once, on first use, and records the
// converted timestamp together with the frame header offset. Only the
// 12-byte frame headers are read; frame payloads are skipped with Seek, so
// no frame payload is buffered. The frame count declared by the file
// header is ignored and the frames actually present are indexed instead.
//
// The stream size is captured up front so a frame header whose declared
// payload does not fully fit is detected deterministically: that frame and
// everything after it is excluded from the index, while all complete
// frames before it stay indexed. A clean io.EOF after a complete frame ends
// the scan normally.
func (i *IVFReader) buildFrameIndex() error {
	if i.indexBuilt {
		return nil
	}

	previousPosition, err := i.seeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	streamSize, err := i.seeker.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	if _, err = i.seeker.Seek(ivfFileHeaderSize, io.SeekStart); err != nil {
		return err
	}

	i.indexEndOffset, err = i.scanFrameIndex(streamSize)
	if err != nil {
		return err
	}

	i.indexBuilt = true
	if _, err = i.seeker.Seek(previousPosition, io.SeekStart); err != nil {
		return err
	}

	return nil
}

// scanFrameIndex walks the frame sequence starting at the file header
// boundary and returns the offset immediately after the last complete
// frame. It reads only frame headers and skips payloads with Seek.
func (i *IVFReader) scanFrameIndex(streamSize int64) (int64, error) {
	buffer := make([]byte, ivfFrameHeaderSize)
	offset := int64(ivfFileHeaderSize)
	for offset < streamSize {
		nextOffset, complete, err := i.indexNextFrame(buffer, offset, streamSize)
		if err != nil {
			return 0, err
		}
		if !complete {
			// Truncated header, oversized payload or overflowing offset:
			// stop the scan and keep the frames already indexed.
			break
		}
		offset = nextOffset
	}

	return offset, nil
}

// indexNextFrame reads one frame header at offset and, when the declared
// payload fully fits inside streamSize, records the frame index entry and
// skips the payload. The returned complete flag is false when the frame
// cannot be fully read.
func (i *IVFReader) indexNextFrame(buffer []byte, offset, streamSize int64) (int64, bool, error) {
	if _, err := io.ReadFull(i.stream, buffer); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return offset, false, nil
		}

		return 0, false, err
	}

	frameSize := int64(binary.LittleEndian.Uint32(buffer[:4]))
	nextOffset := offset + ivfFrameHeaderSize + frameSize
	if nextOffset < offset || nextOffset > streamSize {
		return offset, false, nil
	}

	pts := binary.LittleEndian.Uint64(buffer[4:12])
	i.frameIndex = append(i.frameIndex, frameIndexEntry{
		timestamp: i.ptsToTimestamp(pts),
		offset:    offset,
	})

	if _, err := i.seeker.Seek(nextOffset, io.SeekStart); err != nil {
		return 0, false, err
	}

	return nextOffset, true, nil
}

// parseFileHeader reads 32 bytes from stream and returns
// IVF file header. This is always called before ParseNextFrame().
func (i *IVFReader) parseFileHeader() (*IVFFileHeader, error) {
	buffer := make([]byte, ivfFileHeaderSize)

	bytesRead, err := io.ReadFull(i.stream, buffer)
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, errIncompleteFileHeader
	} else if err != nil {
		return nil, err
	}

	header := &IVFFileHeader{
		signature:           string(buffer[:4]),
		version:             binary.LittleEndian.Uint16(buffer[4:6]),
		headerSize:          binary.LittleEndian.Uint16(buffer[6:8]),
		FourCC:              string(buffer[8:12]),
		Width:               binary.LittleEndian.Uint16(buffer[12:14]),
		Height:              binary.LittleEndian.Uint16(buffer[14:16]),
		TimebaseDenominator: binary.LittleEndian.Uint32(buffer[16:20]),
		TimebaseNumerator:   binary.LittleEndian.Uint32(buffer[20:24]),
		NumFrames:           binary.LittleEndian.Uint32(buffer[24:28]),
		unused:              binary.LittleEndian.Uint32(buffer[28:32]),
	}

	if header.signature != ivfFileHeaderSignature {
		return nil, errSignatureMismatch
	} else if header.version != uint16(0) {
		return nil, fmt.Errorf("%w: expected(0) got(%d)", errUnknownIVFVersion, header.version)
	}

	i.bytesReadSuccesfully += int64(bytesRead)

	return header, nil
}
