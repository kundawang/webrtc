// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package h264reader

import (
	"bytes"
	"errors"
	"io"
	"strconv"
)

// Sentinel errors reported by NextAccessUnit.
var (
	// ErrTruncatedNAL is returned when a NAL unit ends before the mandatory
	// syntax elements required to group access units can be parsed, for
	// example a slice NAL unit with no decodable first_mb_in_slice.
	ErrTruncatedNAL = errors.New("truncated NAL unit")
	// ErrFrameTooLarge is returned when an access unit grows beyond the limit
	// configured with WithMaxAccessUnitSize.
	ErrFrameTooLarge = errors.New("h264 access unit exceeds maximum size")
)

var (
	errInvalidMaxAccessUnitSize = errors.New("max access unit size must be zero or positive")
	errInvalidExpGolombCode     = errors.New("invalid unsigned Exp-Golomb code")
)

type truncatedNALEError struct {
	unitType NalUnitType
	size     int
}

func (e truncatedNALEError) Error() string {
	return "nal unit type " + e.unitType.String() + " is truncated (" +
		strconv.Itoa(e.size) + " bytes)"
}

func errTruncatedNAL(unitType NalUnitType, size int) error {
	return truncatedNALEError{unitType: unitType, size: size}
}

type frameTooLargeError struct {
	currentSize int
	nalSize     int
	limit       int
}

func (e frameTooLargeError) Error() string {
	return "access unit size " + strconv.Itoa(e.currentSize) +
		" plus " + strconv.Itoa(e.nalSize) + " byte nal unit exceeds limit of " +
		strconv.Itoa(e.limit)
}

// Minimum payload sizes (NAL header byte excluded) used to recognize a
// truncated NAL unit before parsing its syntax elements.
const (
	minSlicePayloadSize = 1 // first_mb_in_slice must be decodable
	minAUDPayloadSize   = 1 // primary_pic_type
	minSPSPayloadSize   = 3 // profile_idc, constraint flags, level_idc
	minPPSPayloadSize   = 1
	minSEIPayloadSize   = 1 // payloadType + payloadSize at least
)

// NAL header occupies the first byte of every NAL unit's Data.
const nalHeaderSize = 1

// AccessUnit is one H.264 access unit: all NAL units of a single
// primary coded picture in their original stream order and bytes.
//
// Parameter set and SEI NAL units that precede a picture are attached to the
// access unit of the next displayable picture, as required by most container
// formats. SPS and PPS additionally snapshot the parameter sets that are
// active while the picture is decoded.
type AccessUnit struct {
	// NALUs holds every NAL unit belonging to this access unit, including an
	// optional leading AUD, parameter sets and SEI, in original order. The
	// byte slices alias the data produced by NextNAL and are not modified.
	NALUs []*NAL

	// KeyFrame reports whether the access unit contains an IDR slice
	// (NAL unit type 5).
	KeyFrame bool

	// SPS is the latest sequence parameter set seen up to and including this
	// access unit, or nil if none has been seen yet. In Annex-B streams an
	// SPS is carried inline immediately before the picture it applies to, so
	// a parameter set carried inside this unit is reported here as well as in
	// NALUs.
	SPS *NAL

	// PPS is the latest picture parameter set seen up to and including this
	// access unit, or nil if none has been seen yet.
	PPS *NAL

	// ParameterSetChanged reports that an SPS or PPS carried by this access
	// unit differs from the one active before it, i.e. a parameter set switch
	// takes effect at this picture.
	ParameterSetChanged bool

	// HasAUD reports whether the access unit starts with an access unit
	// delimiter NAL unit.
	HasAUD bool

	size int
}

// Size returns the total size in bytes of the NAL units carried by this
// access unit, excluding Annex-B start codes.
func (au *AccessUnit) Size() int {
	return au.size
}

// WithMaxAccessUnitSize limits the total NAL unit size of an access unit.
// When appending a NAL unit would exceed the limit, NextAccessUnit returns
// ErrFrameTooLarge. A zero or negative value disables the limit, which is the
// default.
func WithMaxAccessUnitSize(maxSize int) Option {
	return func(r *H264Reader) error {
		if maxSize < 0 {
			return errInvalidMaxAccessUnitSize
		}

		r.maxAccessUnitSize = maxSize

		return nil
	}
}

// NextAccessUnit returns the next access unit grouped from the underlying
// Annex-B stream.
//
// Access unit boundaries are detected as follows:
//   - An AUD NAL unit always starts a new access unit.
//   - SPS, PPS and SEI NAL units close the picture in progress and are
//     attached to the next displayable picture.
//   - A primary VCL (slice) NAL unit starts a new access unit when the pending
//     access unit already contains a primary VCL NAL unit and either the new
//     slice is an IDR slice, or its first_mb_in_slice syntax element is zero.
//     The first slice of every H.264 picture has first_mb_in_slice equal to
//     zero while continuation slices of a multi-slice picture carry strictly
//     increasing values, so this separates pictures without an AUD.
//   - Data partition NAL units (types 2-4) and auxiliary coded slice NAL units
//     (type 19) belong to the picture of the primary slice they accompany and
//     never start a new access unit on their own.
//
// The last NAL unit does not need a trailing start code: the final access unit
// is flushed at io.EOF. Truncated NAL units produce ErrTruncatedNAL and
// oversized access units produce ErrFrameTooLarge. io.EOF is returned once all
// complete access units have been delivered.
//
// NextAccessUnit shares the underlying stream with NextNAL; the two iteration
// styles must not be interleaved on the same reader.
func (reader *H264Reader) NextAccessUnit() (*AccessUnit, error) {
	for reader.readyAU == nil {
		nal, err := reader.NextNAL()
		if err != nil {
			if errors.Is(err, io.EOF) {
				if reader.pendingAU != nil {
					reader.readyAU = reader.takePending()

					break
				}

				return nil, io.EOF
			}

			return nil, err
		}

		if err := reader.groupNAL(nal); err != nil {
			return nil, err
		}
	}

	au := reader.readyAU
	reader.readyAU = nil

	return au, nil
}

func (reader *H264Reader) groupNAL(nal *NAL) error {
	if err := reader.checkMinSize(nal); err != nil {
		return err
	}

	switch nal.UnitType {
	case NalUnitTypeAUD:
		reader.emitPendingIfVCL()
		reader.ensurePending().HasAUD = true

		return reader.appendNAL(nal)
	case NalUnitTypeSPS:
		return reader.noteParameterSet(nal, true)
	case NalUnitTypePPS:
		return reader.noteParameterSet(nal, false)
	case NalUnitTypeSEI:
		// SEI belongs to the next displayable picture, so a picture already
		// in progress is closed first.
		reader.emitPendingIfVCL()
		reader.ensurePending()

		return reader.appendNAL(nal)
	case NalUnitTypeCodedSliceNonIdr, NalUnitTypeCodedSliceIdr:
		return reader.groupPrimaryVCL(nal)
	case NalUnitTypeCodedSliceDataPartitionA,
		NalUnitTypeCodedSliceDataPartitionB,
		NalUnitTypeCodedSliceDataPartitionC,
		NalUnitTypeCodedSliceAux:
		// Data partitions and auxiliary slices accompany the pending primary
		// picture rather than opening a new access unit.
		reader.ensurePending()

		return reader.appendNAL(nal)
	default:
		// End of sequence/stream, filler and any other non-VCL NAL unit stay
		// with the pending access unit.
		reader.ensurePending()

		return reader.appendNAL(nal)
	}
}

func (reader *H264Reader) noteParameterSet(nal *NAL, isSPS bool) error {
	var active *NAL
	if isSPS {
		active = reader.activeSPS
	} else {
		active = reader.activePPS
	}

	if active == nil || !bytes.Equal(active.Data, nal.Data) {
		reader.parameterSetChanged = true
		if isSPS {
			reader.activeSPS = nal
		} else {
			reader.activePPS = nal
		}
	}

	// Parameter sets belong to the next displayable picture, so a picture
	// already in progress is closed first.
	reader.emitPendingIfVCL()
	reader.ensurePending()

	return reader.appendNAL(nal)
}

func (reader *H264Reader) groupPrimaryVCL(nal *NAL) error {
	firstMB, err := parseFirstMBInSlice(nal)
	if err != nil {
		return err
	}

	pending := reader.pendingAU
	if pending != nil && pending.hasPrimaryVCL() && reader.startsNewPicture(nal, firstMB) {
		reader.emitPendingIfVCL()
	}

	reader.ensurePending()
	if nal.UnitType == NalUnitTypeCodedSliceIdr {
		reader.pendingAU.KeyFrame = true
	}

	return reader.appendNAL(nal)
}

// startsNewPicture reports whether a primary slice NAL unit begins a new
// picture relative to the slices already grouped in the pending access unit.
//
// In a H.264 coded picture the first slice has first_mb_in_slice equal to zero
// and every following slice carries a strictly increasing, non-zero value.
// A switch between IDR and non-IDR slice NAL unit types is also a picture
// boundary. A non-zero first_mb_in_slice of the same slice type is therefore a
// continuation slice of the picture in progress.
func (reader *H264Reader) startsNewPicture(nal *NAL, firstMB uint32) bool {
	if firstMB == 0 {
		return true
	}

	// A non-zero first_mb is a continuation slice; an IDR slice can only
	// continue an IDR picture, so a type switch still closes the picture.
	return reader.pendingAU.isIDR() != (nal.UnitType == NalUnitTypeCodedSliceIdr)
}

func (au *AccessUnit) isIDR() bool {
	for _, nal := range au.NALUs {
		if nal.UnitType == NalUnitTypeCodedSliceIdr {
			return true
		}
	}

	return false
}

func (reader *H264Reader) checkMinSize(nal *NAL) error {
	payloadSize := len(nal.Data) - nalHeaderSize

	var minSize int
	switch nal.UnitType { //nolint:exhaustive // only truncated-prone types have minimums
	case NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceDataPartitionA,
		NalUnitTypeCodedSliceDataPartitionB,
		NalUnitTypeCodedSliceDataPartitionC,
		NalUnitTypeCodedSliceIdr,
		NalUnitTypeCodedSliceAux:
		minSize = minSlicePayloadSize
	case NalUnitTypeAUD:
		minSize = minAUDPayloadSize
	case NalUnitTypeSPS, NalUnitTypeSpsExt:
		minSize = minSPSPayloadSize
	case NalUnitTypePPS:
		minSize = minPPSPayloadSize
	case NalUnitTypeSEI:
		minSize = minSEIPayloadSize
	}

	if payloadSize < minSize {
		return errors.Join(ErrTruncatedNAL, errTruncatedNAL(nal.UnitType, len(nal.Data)))
	}

	return nil
}

func (reader *H264Reader) ensurePending() *AccessUnit {
	if reader.pendingAU == nil {
		reader.pendingAU = &AccessUnit{}
	}

	// Latch any parameter set switch observed since the pending access unit
	// was opened; repeated identical parameter sets clear the flag elsewhere.
	reader.pendingAU.ParameterSetChanged = reader.pendingAU.ParameterSetChanged ||
		reader.parameterSetChanged
	reader.parameterSetChanged = false

	return reader.pendingAU
}

// takePending finalizes the pending access unit, snapshotting the parameter
// sets latest seen up to this point. Parameter sets carried inside the unit
// (they physically belong to the following picture) are included so that the
// consumer gets the configuration needed to decode the picture alongside it.
func (reader *H264Reader) takePending() *AccessUnit {
	au := reader.pendingAU
	reader.pendingAU = nil
	au.SPS = reader.activeSPS
	au.PPS = reader.activePPS

	return au
}

func (reader *H264Reader) emitPendingIfVCL() {
	if reader.pendingAU != nil && reader.pendingAU.hasVCL() {
		reader.readyAU = reader.takePending()
	}
}

func (reader *H264Reader) appendNAL(nal *NAL) error {
	if reader.maxAccessUnitSize > 0 && reader.pendingAU.size+len(nal.Data) > reader.maxAccessUnitSize {
		return errors.Join(ErrFrameTooLarge, frameTooLargeError{
			currentSize: reader.pendingAU.size,
			nalSize:     len(nal.Data),
			limit:       reader.maxAccessUnitSize,
		})
	}

	reader.pendingAU.size += len(nal.Data)
	reader.pendingAU.NALUs = append(reader.pendingAU.NALUs, nal)

	return nil
}

func (au *AccessUnit) hasVCL() bool {
	for _, nal := range au.NALUs {
		if isVCL(nal.UnitType) {
			return true
		}
	}

	return false
}

func (au *AccessUnit) hasPrimaryVCL() bool {
	for _, nal := range au.NALUs {
		if nal.UnitType == NalUnitTypeCodedSliceNonIdr ||
			nal.UnitType == NalUnitTypeCodedSliceIdr {
			return true
		}
	}

	return false
}

func isVCL(unitType NalUnitType) bool {
	switch unitType {
	case NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceDataPartitionA,
		NalUnitTypeCodedSliceDataPartitionB,
		NalUnitTypeCodedSliceDataPartitionC,
		NalUnitTypeCodedSliceIdr,
		NalUnitTypeCodedSliceAux:
		return true
	default:
		return false
	}
}

// parseFirstMBInSlice extracts the first_mb_in_slice unsigned Exp-Golomb
// syntax element, the first element of a H.264 slice header. Emulation
// prevention bytes (0x03 after 0x00 0x00) are skipped while reading.
func parseFirstMBInSlice(nal *NAL) (uint32, error) {
	rbsp := ebspToRBSP(nal.Data[nalHeaderSize:])
	reader := bitReader{data: rbsp}

	firstMB, err := reader.readUE()
	if err != nil {
		return 0, errors.Join(ErrTruncatedNAL, errTruncatedNAL(nal.UnitType, len(nal.Data)))
	}

	return firstMB, nil
}

// ebspToRBSP removes H.264 emulation prevention bytes: every 0x03 preceded by
// two zero bytes is not part of the RBSP.
func ebspToRBSP(ebsp []byte) []byte {
	rbsp := make([]byte, 0, len(ebsp))

	zeroCount := 0
	for _, b := range ebsp {
		if zeroCount >= 2 && b == 0x03 {
			zeroCount = 0

			continue
		}

		rbsp = append(rbsp, b)
		if b == 0 {
			zeroCount++
		} else {
			zeroCount = 0
		}
	}

	return rbsp
}

// Maximum number of leading zero bits accepted while decoding an unsigned
// Exp-Golomb code into a uint32. A code with 32 leading zero bits cannot be
// represented in uint32 and is invalid here.
const expGolombMaxLeadingZeros = 32

type bitReader struct {
	data   []byte
	offset uint
}

func (r *bitReader) readBit() (byte, error) {
	byteIndex := r.offset / 8
	if byteIndex >= uint(len(r.data)) {
		return 0, io.ErrUnexpectedEOF
	}

	bitIndex := r.offset % 8
	r.offset++

	return (r.data[byteIndex] >> (7 - bitIndex)) & 0x01, nil
}

// readUE decodes an unsigned Exp-Golomb coded syntax element (ue(v)).
func (r *bitReader) readUE() (uint32, error) {
	leadingZeros := 0
	for {
		bit, err := r.readBit()
		if err != nil {
			return 0, err
		}

		if bit != 0 {
			break
		}

		leadingZeros++
		if leadingZeros >= expGolombMaxLeadingZeros {
			return 0, errInvalidExpGolombCode
		}
	}

	var suffix uint32
	for i := 0; i < leadingZeros; i++ {
		bit, err := r.readBit()
		if err != nil {
			return 0, err
		}

		suffix = (suffix << 1) | uint32(bit)
	}

	return (1 << leadingZeros) - 1 + suffix, nil
}
