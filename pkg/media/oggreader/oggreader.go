// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

// Package oggreader implements the Ogg media container reader
package oggreader

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	pageHeaderTypeContinuation      = 0x01
	pageHeaderTypeBeginningOfStream = 0x02
	pageHeaderSignature             = "OggS"

	idPageBasePayloadLength = 19
	pageHeaderLen           = 27

	// defaultMaxPacketSize bounds the size of a single logical packet
	// assembled by ParseNextPacket. Packets growing beyond this limit are
	// discarded and reported with errPacketTooLarge.
	defaultMaxPacketSize = 1 << 20 // 1 MiB
)

var (
	errNilStream                       = errors.New("stream is nil")
	errBadIDPageSignature              = errors.New("bad header signature")
	errBadOpusTagsSignature            = errors.New("bad opus tags signature")
	errBadIDPageType                   = errors.New("wrong header, expected beginning of stream")
	errBadIDPageLength                 = errors.New("payload for id page must be 19 bytes")
	errBadIDPagePayloadSignature       = errors.New("bad payload signature")
	errShortPageHeader                 = errors.New("not enough data for payload header")
	errChecksumMismatch                = errors.New("expected and actual checksum do not match")
	errUnsupportedChannelMappingFamily = errors.New("unsupported channel mapping family")
	errPacketTooLarge                  = errors.New("packet exceeds maximum packet size")
	errPageSequenceGap                 = errors.New("page sequence gap detected, packets may have been lost")
	errDuplicatePage                   = errors.New("duplicate page detected, page ignored")
	errInvalidMaxPacketSize            = errors.New("max packet size must be greater than zero")
)

// OggReader is used to read Ogg files and return page payloads.
type OggReader struct {
	stream               io.Reader
	bytesReadSuccesfully int64
	checksumTable        *[256]uint32
	doChecksum           bool
	streams              map[uint32]*packetAssembler
	packetQueue          []oggPacket
	maxPacketSize        int
}

// OggHeader contains Opus codec metadata parsed from an Opus ID page.
// This header is extracted from an Ogg page payload that starts with the OpusHead
// signature (the first page of an Opus stream in an Ogg container).
//
// Use OggPageHeader.OpusPacketType() to classify a page payload as OpusHead,
// and OggPageHeader.ParseOpusHeader() to parse the OpusHead payload.
//
// https://tools.ietf.org/html/rfc7845.html#section-3
type OggHeader struct {
	ChannelMap   uint8
	Channels     uint8
	OutputGain   uint16
	PreSkip      uint16
	SampleRate   uint32
	Version      uint8
	StreamCount  uint8
	CoupledCount uint8
	// ChannelMapping we store it as a string to be comparable (maps/struct equality)
	// while still holding raw bytes.
	ChannelMapping string
}

// ParseOpusHead parses an Opus head from the page payload.
func ParseOpusHead(payload []byte) (*OggHeader, error) {
	if len(payload) < idPageBasePayloadLength {
		return nil, errBadIDPageLength
	}

	header := parseBasicHeaderFields(payload)
	if err := parseChannelMapping(header, payload); err != nil {
		return nil, err
	}

	return header, nil
}

// OggPageHeader is the metadata for a Page
// Pages are the fundamental unit of multiplexing in an Ogg stream
//
// https://tools.ietf.org/html/rfc7845.html#section-1
type OggPageHeader struct {
	GranulePosition uint64

	sig           [4]byte
	version       uint8
	headerType    uint8
	Serial        uint32
	index         uint32
	segmentsCount uint8
}

type HeaderType string

const (
	headerUnknown  HeaderType = ""
	HeaderOpusID   HeaderType = "OpusHead"
	HeaderOpusTags HeaderType = "OpusTags"
)

func opusPayloadSignature(payload []byte) (HeaderType, bool) {
	if len(payload) < 8 {
		return headerUnknown, false
	}

	sig := HeaderType(payload[:8])
	if sig == HeaderOpusID || sig == HeaderOpusTags {
		return sig, true
	}

	return headerUnknown, false
}

// HeaderType classifies the page.
func (p *OggPageHeader) HeaderType(payload []byte) (HeaderType, bool) {
	sig, ok := opusPayloadSignature(payload)

	if !ok || (sig == HeaderOpusID && p.headerType != pageHeaderTypeBeginningOfStream) {
		return headerUnknown, false
	}

	return sig, true
}

// OggPacketHeader is the metadata for a single logical packet returned by
// ParseNextPacket. A packet is the unit a codec consumes: one or more packets
// may share a page, and one packet may span several (continuation) pages.
type OggPacketHeader struct {
	// Serial identifies the logical bitstream the packet belongs to.
	Serial uint32
	// GranulePosition is the granule position of the page on which the
	// packet completed.
	GranulePosition uint64
	// HeaderType classifies the packet payload: HeaderOpusID for an OpusHead
	// packet, HeaderOpusTags for an OpusTags packet, empty otherwise.
	HeaderType HeaderType
}

// oggPacket is a fully assembled logical packet waiting to be consumed.
type oggPacket struct {
	header OggPacketHeader
	data   []byte
}

// oggPage is a raw page as read from the stream: parsed header, the lacing
// (segment) table and the page payload.
type oggPage struct {
	header   *OggPageHeader
	segments []byte
	payload  []byte
}

// packetAssembler holds the reassembly state of one logical bitstream.
type packetAssembler struct {
	partial   []byte
	lastIndex uint32
	started   bool
}

type Option func(*OggReader) error

// NewWith returns a new Ogg reader and Ogg header
// with an io.Reader input.
//
// Warning: NewWith only parses the first OpusHead (a single logical bitstream/track)
// and returns a single OggHeader. If you need to handle Ogg containers with multiple
// Opus headers/tracks, use NewWithOptions and scan pages (e.g. via ParseNextPage)
// to find and parse each OpusHead.
func NewWith(in io.Reader) (*OggReader, *OggHeader, error) {
	return newWith(in /* doChecksum */, true)
}

// NewWithOptions returns a new Ogg reader.
func NewWithOptions(in io.Reader, options ...Option) (*OggReader, error) {
	reader := &OggReader{
		stream:        in,
		checksumTable: generateChecksumTable(),
		doChecksum:    true,
		streams:       make(map[uint32]*packetAssembler),
		maxPacketSize: defaultMaxPacketSize,
	}

	for _, option := range options {
		if err := option(reader); err != nil {
			return nil, err
		}
	}

	return reader, nil
}

// WithDoChecksum is an option to set the doChecksum flag
// Default is true.
func WithDoChecksum(doChecksum bool) Option {
	return func(o *OggReader) error {
		o.doChecksum = doChecksum

		return nil
	}
}

// WithMaxPacketSize sets the maximum size in bytes of a single logical packet
// assembled by ParseNextPacket. Packets growing beyond this limit are
// discarded and reported with errPacketTooLarge. Default is 1 MiB.
func WithMaxPacketSize(maxPacketSize int) Option {
	return func(o *OggReader) error {
		if maxPacketSize <= 0 {
			return errInvalidMaxPacketSize
		}
		o.maxPacketSize = maxPacketSize

		return nil
	}
}

func newWith(in io.Reader, doChecksum bool) (*OggReader, *OggHeader, error) {
	if in == nil {
		return nil, nil, errNilStream
	}

	reader, err := NewWithOptions(in, WithDoChecksum(doChecksum))
	if err != nil {
		return nil, nil, err
	}

	header, err := reader.readOpusHeader()
	if err != nil {
		return nil, nil, err
	}

	return reader, header, nil
}

func (o *OggReader) readOpusHeader() (*OggHeader, error) {
	page, err := o.readPage()
	if err != nil {
		return nil, err
	}

	if err := validateOpusPageHeader(page.header, page.payload); err != nil {
		return nil, err
	}

	header := parseBasicHeaderFields(page.payload)
	if err := parseChannelMapping(header, page.payload); err != nil {
		return nil, err
	}

	// Queue the ID page as a packet so that ParseNextPacket observes the
	// complete logical stream, including the OpusHead packet.
	_ = o.processPage(page)

	return header, nil
}

func validateOpusPageHeader(pageHeader *OggPageHeader, payload []byte) error {
	if string(pageHeader.sig[:]) != pageHeaderSignature {
		return errBadIDPageSignature
	}

	if pageHeader.headerType != pageHeaderTypeBeginningOfStream {
		return errBadIDPageType
	}

	if len(payload) < idPageBasePayloadLength {
		return errBadIDPageLength
	}

	if sig, ok := opusPayloadSignature(payload); !ok || sig != HeaderOpusID {
		return fmt.Errorf("%w: expected OpusHead, got %s", errBadIDPagePayloadSignature, sig)
	}

	return nil
}

func parseBasicHeaderFields(payload []byte) *OggHeader {
	header := &OggHeader{}
	header.Version = payload[8]
	header.Channels = payload[9]
	header.PreSkip = binary.LittleEndian.Uint16(payload[10:12])
	header.SampleRate = binary.LittleEndian.Uint32(payload[12:16])
	header.OutputGain = binary.LittleEndian.Uint16(payload[16:18])
	header.ChannelMap = payload[18]

	return header
}

// parseChannelMapping parses channel mapping data based on the channel map family.
// https://datatracker.ietf.org/doc/html/rfc7845#section-5.1.1
// family mapping of 2 and 3 are defined in https://datatracker.ietf.org/doc/html/rfc8486
func parseChannelMapping(header *OggHeader, payload []byte) error {
	switch header.ChannelMap {
	case 0:
		return validatePayloadLength(payload, idPageBasePayloadLength)
	case 1, 2, 255:
		return parseExtendedChannelMapping(header, payload)
	case 3:
		return fmt.Errorf("%w: ambisonics family type 3 is not supported", errUnsupportedChannelMappingFamily)
	default:
		return errUnsupportedChannelMappingFamily
	}
}

func validatePayloadLength(payload []byte, expectedLen int) error {
	if len(payload) != expectedLen {
		return errBadIDPageLength
	}

	return nil
}

func parseExtendedChannelMapping(header *OggHeader, payload []byte) error {
	expectedPayloadLen := 21 + int(header.Channels)
	if err := validatePayloadLength(payload, expectedPayloadLen); err != nil {
		return err
	}

	header.StreamCount = payload[19]
	header.CoupledCount = payload[20]
	header.ChannelMapping = string(payload[21:expectedPayloadLen])

	return nil
}

// ParseNextPage reads from stream and returns Ogg page payload, header,
// and an error if there is incomplete page data.
func (o *OggReader) ParseNextPage() ([]byte, *OggPageHeader, error) {
	page, err := o.readPage()
	if err != nil {
		return nil, nil, err
	}

	return page.payload, page.header, nil
}

// readPage reads a single raw page (header, lacing table and payload) from
// the stream.
func (o *OggReader) readPage() (*oggPage, error) {
	header := make([]byte, pageHeaderLen)

	n, err := io.ReadFull(o.stream, header)
	if err != nil {
		return nil, err
	} else if n < len(header) {
		return nil, errShortPageHeader
	}

	pageHeader := &OggPageHeader{
		sig: [4]byte{header[0], header[1], header[2], header[3]},
	}

	pageHeader.version = header[4]
	pageHeader.headerType = header[5]
	pageHeader.GranulePosition = binary.LittleEndian.Uint64(header[6 : 6+8])
	pageHeader.Serial = binary.LittleEndian.Uint32(header[14 : 14+4])
	pageHeader.index = binary.LittleEndian.Uint32(header[18 : 18+4])
	pageHeader.segmentsCount = header[26]

	sizeBuffer := make([]byte, pageHeader.segmentsCount)
	if _, err = io.ReadFull(o.stream, sizeBuffer); err != nil {
		return nil, err
	}

	payloadSize := 0
	for _, s := range sizeBuffer {
		payloadSize += int(s)
	}

	payload := make([]byte, payloadSize)
	if _, err = io.ReadFull(o.stream, payload); err != nil {
		return nil, err
	}

	if o.doChecksum {
		var checksum uint32
		updateChecksum := func(v byte) {
			checksum = (checksum << 8) ^ o.checksumTable[byte(checksum>>24)^v]
		}

		for index := range header {
			// Don't include expected checksum in our generation
			if index > 21 && index < 26 {
				updateChecksum(0)

				continue
			}

			updateChecksum(header[index])
		}
		for _, s := range sizeBuffer {
			updateChecksum(s)
		}
		for index := range payload {
			updateChecksum(payload[index])
		}

		if binary.LittleEndian.Uint32(header[22:22+4]) != checksum {
			return nil, errChecksumMismatch
		}
	}

	o.bytesReadSuccesfully += int64(len(header) + len(sizeBuffer) + len(payload))

	return &oggPage{header: pageHeader, segments: sizeBuffer, payload: payload}, nil
}

// ParseNextPacket reads from the stream and returns the next complete logical
// packet. Packets are extracted from the lacing (segment) table of each page
// and reassembled across continuation pages; page headers are never included
// in the returned data. Pages of all logical bitstreams are demultiplexed:
// every returned packet is tagged with the serial of the stream it belongs to
// and the granule position of the page on which it completed.
//
// Do not interleave ParseNextPacket with ParseNextPage: pages consumed by one
// are invisible to the other.
//
// The returned error is io.EOF once the stream is fully consumed, or
// io.ErrUnexpectedEOF if the stream ends while a packet is still incomplete.
// A page that fails checksum validation yields errChecksumMismatch and any
// incomplete packets are dropped. Lost pages (errPageSequenceGap), duplicate
// pages (errDuplicatePage) and oversized packets (errPacketTooLarge, see
// WithMaxPacketSize) are non-fatal: packets that were completed on the same
// page remain queued and are returned by subsequent calls.
func (o *OggReader) ParseNextPacket() ([]byte, *OggPacketHeader, error) {
	for {
		if len(o.packetQueue) > 0 {
			pkt := o.packetQueue[0]
			o.packetQueue = o.packetQueue[1:]
			header := pkt.header

			return pkt.data, &header, nil
		}

		page, err := o.readPage()
		if err != nil {
			if errors.Is(err, io.EOF) {
				if o.hasPendingPacket() {
					// The stream ended before the pending packet completed.
					o.dropPendingPackets()

					return nil, nil, io.ErrUnexpectedEOF
				}

				return nil, nil, err
			}

			// The page could not be read or validated (e.g. checksum
			// mismatch); pending packets can no longer be completed.
			o.dropPendingPackets()

			return nil, nil, err
		}

		if err = o.processPage(page); err != nil {
			// Non-fatal: packets completed on this page stay queued and are
			// returned by subsequent calls.
			return nil, nil, err
		}
	}
}

// processPage folds one raw page into the reassembly state of its logical
// stream and queues every packet the page completes.
func (o *OggReader) processPage(page *oggPage) error {
	st := o.streams[page.header.Serial]
	if st == nil {
		st = &packetAssembler{}
		o.streams[page.header.Serial] = st
	}

	retErr := checkPageSequence(st, page.header)
	if errors.Is(retErr, errDuplicatePage) {
		return retErr
	}

	segOff := 0
	continued := page.header.headerType&pageHeaderTypeContinuation != 0
	switch {
	case continued && len(st.partial) == 0:
		// The start of the continued packet is missing (the stream was
		// joined mid-packet, or the preceding page was lost or oversized):
		// discard the orphaned leading fragment.
		segOff = skipLeadingFragment(page.segments)
	case !continued && len(st.partial) > 0:
		// The page that should have completed the pending packet never
		// arrived.
		st.partial = nil
		if retErr == nil {
			retErr = errPageSequenceGap
		}
	}

	if err := o.assembleSegments(st, page, segOff); err != nil && retErr == nil {
		retErr = err
	}

	return retErr
}

// checkPageSequence validates the page sequence number of one logical stream
// and drops state that can no longer produce a complete packet.
func checkPageSequence(st *packetAssembler, header *OggPageHeader) error {
	if st.started {
		switch {
		case header.index == st.lastIndex:
			return errDuplicatePage
		case header.index != st.lastIndex+1:
			// One or more pages were lost: a pending packet can no longer be
			// completed.
			st.partial = nil
			st.lastIndex = header.index

			return errPageSequenceGap
		}
	}

	st.started = true
	st.lastIndex = header.index

	return nil
}

// skipLeadingFragment returns the number of lacing values that make up the
// first (orphaned) packet fragment of a page.
func skipLeadingFragment(segments []byte) int {
	i := 0
	for i < len(segments) {
		v := segments[i]
		i++
		if v < 255 {
			break
		}
	}

	return i
}

// assembleSegments walks the lacing table of a page (skipping segOff orphaned
// leading segments), appends every fragment to the pending packet and queues
// each packet that terminates on this page.
func (o *OggReader) assembleSegments(st *packetAssembler, page *oggPage, segOff int) error {
	payload := page.payload
	for _, s := range page.segments[:segOff] {
		payload = payload[int(s):]
	}

	var retErr error
	segments := page.segments[segOff:]
	for i := 0; i < len(segments); i++ {
		v := int(segments[i])
		st.partial = append(st.partial, payload[:v]...)
		payload = payload[v:]

		if len(st.partial) > o.maxPacketSize {
			// Drop the oversized packet, including every following segment
			// up to and including the one that terminates it. If it does not
			// terminate on this page, the next page's continuation is
			// discarded as an orphaned fragment.
			st.partial = st.partial[:0]
			retErr = errPacketTooLarge
			for v == 255 && i+1 < len(segments) {
				i++
				v = int(segments[i])
				payload = payload[v:]
			}

			continue
		}

		if v < 255 {
			// A lacing value below 255 terminates the packet; a zero lacing
			// value yields a valid zero-length packet.
			o.queuePacket(page.header, st.partial)
			st.partial = st.partial[:0]
		}
	}

	return retErr
}

// queuePacket copies the assembled bytes and appends the packet to the queue.
func (o *OggReader) queuePacket(pageHeader *OggPageHeader, data []byte) {
	pkt := make([]byte, len(data))
	copy(pkt, data)

	sig, _ := opusPayloadSignature(pkt)

	o.packetQueue = append(o.packetQueue, oggPacket{
		data: pkt,
		header: OggPacketHeader{
			Serial:          pageHeader.Serial,
			GranulePosition: pageHeader.GranulePosition,
			HeaderType:      sig,
		},
	})
}

// hasPendingPacket reports whether any logical stream holds an incomplete
// packet.
func (o *OggReader) hasPendingPacket() bool {
	for _, st := range o.streams {
		if len(st.partial) > 0 {
			return true
		}
	}

	return false
}

// dropPendingPackets discards every incomplete packet.
func (o *OggReader) dropPendingPackets() {
	for _, st := range o.streams {
		st.partial = nil
	}
}

// ResetReader resets the internal stream of OggReader. This is useful
// for live streams, where the end of the file might be read without the
// data being finished.
func (o *OggReader) ResetReader(reset func(bytesRead int64) io.Reader) {
	o.stream = reset(o.bytesReadSuccesfully)
}

func generateChecksumTable() *[256]uint32 {
	var table [256]uint32
	const poly = 0x04c11db7

	for i := range table {
		r := uint32(i) << 24 //nolint:gosec // G115
		for range 8 {
			if (r & 0x80000000) != 0 {
				r = (r << 1) ^ poly
			} else {
				r <<= 1
			}
		}
		table[i] = (r & 0xffffffff) //nolint:gosec // no out of bounds access here.
	}

	return &table
}

// OpusTags is the metadata for an OpusTags page.
// https://www.xiph.org/vorbis/doc/v-comment.html
type OpusTags struct {
	Vendor       string
	UserComments []UserComment
}

// UserComment is a key-value pair of a vorbis comment.
type UserComment struct {
	Comment string
	Value   string
}

// ParseOpusTags parses an OpusTags from the page payload.
// https://datatracker.ietf.org/doc/html/rfc7845#section-5.2
func ParseOpusTags(payload []byte) (*OpusTags, error) {
	const (
		headerMagicLen = 8
		u32Size        = 4
		minHeaderLen   = headerMagicLen + u32Size + u32Size
	)

	if err := validateOpusTagsHeader(payload, minHeaderLen); err != nil {
		return nil, err
	}

	vendor, vendorEnd, err := parseVendorString(payload, headerMagicLen, u32Size, minHeaderLen)
	if err != nil {
		return nil, err
	}

	userComments, err := parseUserComments(payload, vendorEnd, u32Size)
	if err != nil {
		return nil, err
	}

	return &OpusTags{
		Vendor:       vendor,
		UserComments: userComments,
	}, nil
}

func validateOpusTagsHeader(payload []byte, minHeaderLen int) error {
	if len(payload) < minHeaderLen {
		return fmt.Errorf("%w: payload too short", errBadOpusTagsSignature)
	}

	got := HeaderType(payload[:8])
	if got != HeaderOpusTags {
		return fmt.Errorf("%w: expected %q, got %q", errBadOpusTagsSignature, HeaderOpusTags, got)
	}

	return nil
}

func parseVendorString(payload []byte, headerMagicLen, u32Size, minHeaderLen int) (string, int, error) {
	vendorLen32 := binary.LittleEndian.Uint32(payload[headerMagicLen : headerMagicLen+u32Size])
	if int(vendorLen32) > len(payload)-minHeaderLen {
		return "", 0, fmt.Errorf("%w: payload too short for vendor string", errBadOpusTagsSignature)
	}
	vendorLen := int(vendorLen32)

	vendorStart := headerMagicLen + u32Size
	vendorEnd := vendorStart + vendorLen
	if vendorEnd+u32Size > len(payload) {
		return "", 0, fmt.Errorf("%w: payload too short for vendor+comment count", errBadOpusTagsSignature)
	}

	vendor := string(payload[vendorStart:vendorEnd])

	return vendor, vendorEnd, nil
}

func parseUserComments(payload []byte, vendorEnd, u32Size int) ([]UserComment, error) {
	userCommentCount32 := binary.LittleEndian.Uint32(payload[vendorEnd : vendorEnd+u32Size])
	if int(userCommentCount32) > (len(payload)-vendorEnd)/u32Size {
		return nil, fmt.Errorf("%w: unreasonable comment count", errBadOpusTagsSignature)
	}
	userCommentCount := int(userCommentCount32)

	pos := vendorEnd + u32Size
	userComments := make([]UserComment, userCommentCount)

	for i := range userComments {
		comment, nextPos, err := parseSingleUserComment(payload, pos, u32Size, i)
		if err != nil {
			return nil, err
		}
		userComments[i] = comment
		pos = nextPos
	}

	return userComments, nil
}

func parseSingleUserComment(payload []byte, pos, u32Size, index int) (UserComment, int, error) {
	if pos+u32Size > len(payload) {
		return UserComment{}, 0, fmt.Errorf("%w: payload too short for comment len %d", errBadOpusTagsSignature, index)
	}

	commentLen32 := binary.LittleEndian.Uint32(payload[pos : pos+u32Size])
	pos += u32Size

	commentLen := int(commentLen32)
	if commentLen < 0 || pos+commentLen > len(payload) {
		return UserComment{}, 0, fmt.Errorf("%w: payload too short for comment %d", errBadOpusTagsSignature, index)
	}

	comment := string(payload[pos : pos+commentLen])
	pos += commentLen

	parts := strings.SplitN(comment, "=", 2)
	if len(parts) != 2 {
		return UserComment{}, 0, fmt.Errorf("%w: invalid comment %d", errBadOpusTagsSignature, index)
	}

	return UserComment{
		Comment: parts[0],
		Value:   parts[1],
	}, pos, nil
}
