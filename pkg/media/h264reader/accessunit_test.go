// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package h264reader

import (
	"bytes"
	"errors"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

// naluSpec describes one NAL unit of a fixed Annex-B test stream.
type naluSpec struct {
	header  byte   // NAL header byte, determines RefIdc and UnitType
	payload []byte // RBSP bytes following the NAL header
	prefix  int    // start code length: 3 or 4 bytes
}

// annexBStream joins NAL unit specifications with Annex-B start codes. Every
// stream deliberately ends on the last NAL unit without a trailing start code.
func annexBStream(specs ...naluSpec) []byte {
	stream := make([]byte, 0)

	for _, spec := range specs {
		switch spec.prefix {
		case 3:
			stream = append(stream, 0x00, 0x00, 0x01)
		default:
			stream = append(stream, 0x00, 0x00, 0x00, 0x01)
		}

		stream = append(stream, spec.header)
		stream = append(stream, spec.payload...)
	}

	return stream
}

func naluBytes(spec naluSpec) []byte {
	out := make([]byte, 0, len(spec.payload)+1)
	out = append(out, spec.header)
	out = append(out, spec.payload...)

	return out
}

// readAllAccessUnits drains the reader and returns every access unit.
func readAllAccessUnits(t *testing.T, reader *H264Reader) []*AccessUnit {
	t.Helper()

	accessUnits := make([]*AccessUnit, 0)

	for {
		accessUnit, err := reader.NextAccessUnit()
		if errors.Is(err, io.EOF) {
			break
		}

		require.NoError(t, err)
		require.NotNil(t, accessUnit)

		accessUnits = append(accessUnits, accessUnit)
	}

	return accessUnits
}

func requireNALHeaders(t *testing.T, accessUnit *AccessUnit, expected ...byte) {
	t.Helper()
	require.Len(t, accessUnit.NALUs, len(expected))

	for i, header := range expected {
		require.NotEmpty(t, accessUnit.NALUs[i].Data)
		require.Equal(t, header, accessUnit.NALUs[i].Data[0])
	}
}

// Stream A: AUD-delimited, contains SEI, an IDR keyframe split into two
// slices, an inter-frame also split into two slices, and a final IDR.
func streamA() []naluSpec {
	return []naluSpec{
		{header: 0x09, payload: []byte{0xF0}, prefix: 4},                   // AUD
		{header: 0x67, payload: []byte{0x42, 0x00, 0x1E, 0xAB}, prefix: 4}, // SPS
		{header: 0x68, payload: []byte{0xCE, 0x38, 0x80}, prefix: 4},       // PPS
		{header: 0x06, payload: []byte{0x05, 0x10}, prefix: 4},             // SEI
		{header: 0x65, payload: []byte{0x80, 0x11, 0x22}, prefix: 4},       // IDR slice, first_mb = 0
		{header: 0x65, payload: []byte{0x30, 0x33}, prefix: 4},             // IDR slice, first_mb = 5 (continuation)
		{header: 0x09, payload: []byte{0xF0}, prefix: 4},                   // AUD
		{header: 0x41, payload: []byte{0x80, 0x44}, prefix: 4},             // non-IDR slice, first_mb = 0
		{header: 0x41, payload: []byte{0x60, 0x55}, prefix: 4},             // non-IDR slice, first_mb = 2 (continuation)
		{header: 0x09, payload: []byte{0xF0}, prefix: 4},                   // AUD
		{header: 0x65, payload: []byte{0x80, 0x66}, prefix: 4},             // IDR slice, first_mb = 0
	}
}

// Stream B: no AUD, 3-byte start codes, boundaries inferred purely from slice
// first_mb_in_slice, including a multi-slice picture and several single-slice
// pictures separated without any delimiter.
func streamB() []naluSpec {
	return []naluSpec{
		{header: 0x67, payload: []byte{0x42, 0x00, 0x1E, 0xAB}, prefix: 3}, // SPS
		{header: 0x68, payload: []byte{0xCE, 0x38, 0x80}, prefix: 3},       // PPS
		{header: 0x41, payload: []byte{0x80, 0xAA}, prefix: 3},             // non-IDR slice, first_mb = 0
		{header: 0x41, payload: []byte{0x60, 0xBB}, prefix: 3},             // non-IDR slice, first_mb = 2 (continuation)
		{header: 0x41, payload: []byte{0x80, 0xCC}, prefix: 3},             // non-IDR slice, first_mb = 0
		{header: 0x41, payload: []byte{0x80, 0xEE}, prefix: 3},             // non-IDR slice, first_mb = 0
		{header: 0x65, payload: []byte{0x80, 0xDD}, prefix: 3},             // IDR slice, first_mb = 0
	}
}

func TestNextAccessUnitStreamAWithAUDAndSEI(t *testing.T) {
	require := require.New(t)

	specs := streamA()
	reader, err := NewReaderWithOptions(
		bytes.NewReader(annexBStream(specs...)),
		WithIncludeSEI(true),
	)
	require.NoError(err)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 3)

	// Frame 1: AUD + SPS + PPS + SEI + two IDR slices, keyframe.
	frame1 := accessUnits[0]
	require.True(frame1.KeyFrame)
	require.True(frame1.HasAUD)
	require.True(frame1.ParameterSetChanged)
	requireNALHeaders(t, frame1, 0x09, 0x67, 0x68, 0x06, 0x65, 0x65)
	require.Equal(NalUnitTypeSPS, frame1.SPS.UnitType)
	require.Equal(NalUnitTypePPS, frame1.PPS.UnitType)
	require.Same(frame1.SPS, frame1.NALUs[1])
	require.Same(frame1.PPS, frame1.NALUs[2])

	// Frame 2: AUD + two non-IDR slices sharing one picture.
	frame2 := accessUnits[1]
	require.False(frame2.KeyFrame)
	require.True(frame2.HasAUD)
	require.False(frame2.ParameterSetChanged)
	requireNALHeaders(t, frame2, 0x09, 0x41, 0x41)
	require.Same(frame1.SPS, frame2.SPS)
	require.Same(frame1.PPS, frame2.PPS)

	// Frame 3: AUD + single IDR slice, keyframe.
	frame3 := accessUnits[2]
	require.True(frame3.KeyFrame)
	require.True(frame3.HasAUD)
	require.False(frame3.ParameterSetChanged)
	requireNALHeaders(t, frame3, 0x09, 0x65)

	// NALU original order and bytes are preserved.
	flattened := make([]byte, 0)
	for _, accessUnit := range accessUnits {
		for _, nal := range accessUnit.NALUs {
			flattened = append(flattened, nal.Data...)
		}
	}

	expected := make([]byte, 0)
	for _, spec := range specs {
		expected = append(expected, naluBytes(spec)...)
	}

	require.Equal(expected, flattened)
}

func TestNextAccessUnitStreamAWithoutSEIOption(t *testing.T) {
	require := require.New(t)

	reader, err := NewReader(bytes.NewReader(annexBStream(streamA()...)))
	require.NoError(err)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 3)

	// SEI is skipped by the default reader; everything else stays in order.
	requireNALHeaders(t, accessUnits[0], 0x09, 0x67, 0x68, 0x65, 0x65)
	requireNALHeaders(t, accessUnits[1], 0x09, 0x41, 0x41)
	requireNALHeaders(t, accessUnits[2], 0x09, 0x65)
}

func TestNextAccessUnitStreamBNoAUDSliceBoundaries(t *testing.T) {
	require := require.New(t)

	specs := streamB()
	reader, err := NewReader(bytes.NewReader(annexBStream(specs...)))
	require.NoError(err)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 4)

	// Frame 1: SPS + PPS + two non-IDR slices grouped by first_mb.
	frame1 := accessUnits[0]
	require.False(frame1.KeyFrame)
	require.False(frame1.HasAUD)
	require.True(frame1.ParameterSetChanged)
	requireNALHeaders(t, frame1, 0x67, 0x68, 0x41, 0x41)
	require.Equal(specs[0].payload, frame1.SPS.Data[1:])
	require.Equal(specs[1].payload, frame1.PPS.Data[1:])

	// Frame 2 and 3: single non-IDR pictures opened by first_mb == 0, with no
	// AUD present. They inherit the active parameter sets unchanged.
	frame2 := accessUnits[1]
	require.False(frame2.KeyFrame)
	require.False(frame2.HasAUD)
	require.False(frame2.ParameterSetChanged)
	requireNALHeaders(t, frame2, 0x41)
	require.Same(frame1.SPS, frame2.SPS)
	require.Same(frame1.PPS, frame2.PPS)

	frame3 := accessUnits[2]
	require.False(frame3.KeyFrame)
	require.False(frame3.HasAUD)
	require.False(frame3.ParameterSetChanged)
	requireNALHeaders(t, frame3, 0x41)
	require.Same(frame1.SPS, frame3.SPS)
	require.Same(frame1.PPS, frame3.PPS)

	// Frame 4: an IDR slice opens a new keyframe picture even without an AUD.
	frame4 := accessUnits[3]
	require.True(frame4.KeyFrame)
	require.False(frame4.HasAUD)
	require.False(frame4.ParameterSetChanged)
	requireNALHeaders(t, frame4, 0x65)
	require.Same(frame1.SPS, frame4.SPS)
	require.Same(frame1.PPS, frame4.PPS)

	// Byte-for-byte preservation across every access unit.
	flattened := make([]byte, 0)
	for _, accessUnit := range accessUnits {
		for _, nal := range accessUnit.NALUs {
			flattened = append(flattened, nal.Data...)
		}
	}

	expected := make([]byte, 0)
	for _, spec := range specs {
		expected = append(expected, naluBytes(spec)...)
	}

	require.Equal(expected, flattened)
}

func TestNextAccessUnitParameterSetSwitchWithAUD(t *testing.T) {
	require := require.New(t)

	// Two AUD-delimited pictures; the second keyframe carries a changed SPS/PPS.
	specs := []naluSpec{
		{header: 0x09, payload: []byte{0xF0}, prefix: 4},                   // AUD
		{header: 0x67, payload: []byte{0x42, 0x00, 0x1E, 0xAB}, prefix: 4}, // SPS v1
		{header: 0x68, payload: []byte{0xCE, 0x38, 0x80}, prefix: 4},       // PPS v1
		{header: 0x41, payload: []byte{0x80, 0xAA}, prefix: 4},             // non-IDR
		{header: 0x09, payload: []byte{0xF0}, prefix: 4},                   // AUD
		{header: 0x67, payload: []byte{0x4D, 0x00, 0x28, 0xCD}, prefix: 4}, // SPS v2
		{header: 0x68, payload: []byte{0x1B, 0x70, 0x90}, prefix: 4},       // PPS v2
		{header: 0x65, payload: []byte{0x80, 0xDD}, prefix: 4},             // IDR
	}
	reader := CreateReader(annexBStream(specs...), require)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 2)

	frame1 := accessUnits[0]
	require.False(frame1.KeyFrame)
	require.True(frame1.ParameterSetChanged)
	require.Equal(specs[1].payload, frame1.SPS.Data[1:])
	require.Equal(specs[2].payload, frame1.PPS.Data[1:])

	frame2 := accessUnits[1]
	require.True(frame2.KeyFrame)
	require.True(frame2.ParameterSetChanged)
	requireNALHeaders(t, frame2, 0x09, 0x67, 0x68, 0x65)
	require.Equal(specs[5].payload, frame2.SPS.Data[1:])
	require.Equal(specs[6].payload, frame2.PPS.Data[1:])
}

func TestNextAccessUnitEOFAndNoTrailingStartCode(t *testing.T) {
	require := require.New(t)

	// Single NAL unit, 3-byte prefix, nothing follows it.
	stream := []byte{0x00, 0x00, 0x01, 0x65, 0x80}
	reader := CreateReader(stream, require)

	accessUnit, err := reader.NextAccessUnit()
	require.NoError(err)
	require.NotNil(accessUnit)
	require.True(accessUnit.KeyFrame)
	requireNALHeaders(t, accessUnit, 0x65)
	require.Equal([]byte{0x65, 0x80}, accessUnit.NALUs[0].Data)

	next, err := reader.NextAccessUnit()
	require.ErrorIs(err, io.EOF)
	require.Nil(next)
}

func TestNextAccessUnitEmptyStream(t *testing.T) {
	require := require.New(t)

	reader := CreateReader([]byte{}, require)

	accessUnit, err := reader.NextAccessUnit()
	require.ErrorIs(err, io.EOF)
	require.Nil(accessUnit)
}

func TestNextAccessUnitTruncatedNAL(t *testing.T) {
	require := require.New(t)

	// Slice NAL unit with only the header byte: first_mb_in_slice cannot be
	// decoded.
	stream := annexBStream(
		naluSpec{header: 0x67, payload: []byte{0x42, 0x00, 0x1E}, prefix: 4},
		naluSpec{header: 0x65, payload: []byte{}, prefix: 4},
	)
	reader := CreateReader(stream, require)

	accessUnit, err := reader.NextAccessUnit()
	require.ErrorIs(err, ErrTruncatedNAL)
	require.Nil(accessUnit)

	var truncated truncatedNALEError
	require.True(errors.As(err, &truncated))
	require.Equal(NalUnitTypeCodedSliceIdr, truncated.unitType)
	require.Contains(truncated.Error(), "CodedSliceIdr(5)")

	// SPS below its minimum payload size is also truncated.
	stream = annexBStream(
		naluSpec{header: 0x67, payload: []byte{0x42, 0x00}, prefix: 4},
	)
	reader = CreateReader(stream, require)

	_, err = reader.NextAccessUnit()
	require.ErrorIs(err, ErrTruncatedNAL)
}

func TestNextAccessUnitTruncatedExpGolombWithEmulationPrevention(t *testing.T) {
	require := require.New(t)

	// A leading 0x00 0x00 0x03 sequence is an emulation prevention escape;
	// after removal no Exp-Golomb code remains.
	stream := annexBStream(
		naluSpec{header: 0x41, payload: []byte{0x00, 0x00, 0x03, 0x00}, prefix: 4},
	)
	reader := CreateReader(stream, require)

	_, err := reader.NextAccessUnit()
	require.ErrorIs(err, ErrTruncatedNAL)

	// With a decodable first_mb_in_slice the same escape sequence is harmless.
	stream = annexBStream(
		naluSpec{header: 0x41, payload: []byte{0x80, 0x00, 0x00, 0x03, 0x00}, prefix: 4},
	)
	reader = CreateReader(stream, require)

	accessUnit, err := reader.NextAccessUnit()
	require.NoError(err)
	requireNALHeaders(t, accessUnit, 0x41)
	require.Equal([]byte{0x41, 0x80, 0x00, 0x00, 0x03, 0x00}, accessUnit.NALUs[0].Data)
}

func TestNextAccessUnitFrameTooLarge(t *testing.T) {
	require := require.New(t)

	stream := annexBStream(
		naluSpec{header: 0x67, payload: []byte{0x42, 0x00, 0x1E, 0xAB}, prefix: 4},
		naluSpec{header: 0x65, payload: []byte{0x80, 0x11, 0x22, 0x33}, prefix: 4},
	)
	reader, err := NewReaderWithOptions(
		bytes.NewReader(stream),
		WithMaxAccessUnitSize(6),
	)
	require.NoError(err)

	_, err = reader.NextAccessUnit()
	require.ErrorIs(err, ErrFrameTooLarge)

	var tooLarge frameTooLargeError
	require.True(errors.As(err, &tooLarge))
	require.Equal(6, tooLarge.limit)
	require.Contains(tooLarge.Error(), "exceeds limit of 6")
}

func TestBitReaderExpGolomb(t *testing.T) {
	require := require.New(t)

	tt := []struct {
		name   string
		data   []byte
		bits   uint
		expect uint32
	}{
		{"codeNum zero (1)", []byte{0x80}, 1, 0},
		{"codeNum one (010)", []byte{0x40}, 3, 1},
		{"codeNum two (011)", []byte{0x60}, 3, 2},
		{"codeNum five (00110)", []byte{0x30}, 5, 5},
	}

	for _, tc := range tt {
		reader := bitReader{data: tc.data}
		value, err := reader.readUE()
		require.NoError(err, tc.name)
		require.Equal(tc.expect, value, tc.name)
		require.Equal(tc.bits, reader.offset, tc.name)
	}

	// Reading past the end reports an incomplete stream rather than panicking.
	emptyReader := bitReader{data: []byte{}}
	_, err := emptyReader.readUE()
	require.ErrorIs(err, io.ErrUnexpectedEOF)

	// A run of more than 32 zero bits cannot encode a uint32 codeNum.
	zeroReader := bitReader{data: make([]byte, 5)}
	_, err = zeroReader.readUE()
	require.ErrorIs(err, errInvalidExpGolombCode)
}

func TestWithMaxAccessUnitSizeInvalid(t *testing.T) {
	require := require.New(t)

	reader, err := NewReaderWithOptions(
		bytes.NewReader(annexBStream(naluSpec{header: 0x09, payload: []byte{0xF0}, prefix: 4})),
		WithMaxAccessUnitSize(-1),
	)
	require.Error(err)
	require.Nil(reader)
}

func TestNextAccessUnitDataPartitionsAndAuxiliarySlice(t *testing.T) {
	require := require.New(t)

	stream := annexBStream(
		naluSpec{header: 0x09, payload: []byte{0xF0}, prefix: 4},       // AUD
		naluSpec{header: 0x41, payload: []byte{0x80, 0xAA}, prefix: 4}, // primary non-IDR slice
		naluSpec{header: 0x02, payload: []byte{0x11}, prefix: 4},       // data partition A
		naluSpec{header: 0x03, payload: []byte{0x22}, prefix: 4},       // data partition B
		naluSpec{header: 0x04, payload: []byte{0x33}, prefix: 4},       // data partition C
		naluSpec{header: 0x13, payload: []byte{0x44}, prefix: 4},       // auxiliary coded slice
		naluSpec{header: 0x09, payload: []byte{0xF0}, prefix: 4},       // AUD
		naluSpec{header: 0x65, payload: []byte{0x80, 0xBB}, prefix: 4}, // IDR
	)
	reader := CreateReader(stream, require)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 2)

	require.False(accessUnits[0].KeyFrame)
	requireNALHeaders(t, accessUnits[0], 0x09, 0x41, 0x02, 0x03, 0x04, 0x13)

	require.True(accessUnits[1].KeyFrame)
	requireNALHeaders(t, accessUnits[1], 0x09, 0x65)
}

func TestNextAccessUnitKeepsFillerAndEndOfStream(t *testing.T) {
	require := require.New(t)

	stream := annexBStream(
		naluSpec{header: 0x09, payload: []byte{0xF0}, prefix: 4},       // AUD
		naluSpec{header: 0x41, payload: []byte{0x80, 0xAA}, prefix: 4}, // non-IDR slice
		naluSpec{header: 0x0C, payload: []byte{0xFF, 0xFF}, prefix: 4}, // filler data
		naluSpec{header: 0x0B, payload: []byte{0x00}, prefix: 4},       // end of stream
	)
	reader := CreateReader(stream, require)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 1)
	requireNALHeaders(t, accessUnits[0], 0x09, 0x41, 0x0C, 0x0B)
}

func TestNextAccessUnitRepeatedIdenticalParameterSet(t *testing.T) {
	require := require.New(t)

	sps := naluSpec{header: 0x67, payload: []byte{0x42, 0x00, 0x1E, 0xAB}, prefix: 4}
	stream := annexBStream(
		sps,
		naluSpec{header: 0x68, payload: []byte{0xCE, 0x38, 0x80}, prefix: 4},
		naluSpec{header: 0x41, payload: []byte{0x80, 0xAA}, prefix: 4},
		sps, // identical SPS repeated
		naluSpec{header: 0x41, payload: []byte{0x80, 0xBB}, prefix: 4},
	)
	reader := CreateReader(stream, require)

	accessUnits := readAllAccessUnits(t, reader)
	require.Len(accessUnits, 2)
	require.True(accessUnits[0].ParameterSetChanged)
	require.False(accessUnits[1].ParameterSetChanged)
	require.Same(accessUnits[0].SPS, accessUnits[1].SPS)
}

func TestNextAccessUnitSizeExcludesStartCodes(t *testing.T) {
	require := require.New(t)

	spec := naluSpec{header: 0x65, payload: []byte{0x80, 0x11, 0x22}, prefix: 4}
	reader := CreateReader(annexBStream(spec), require)

	accessUnit, err := reader.NextAccessUnit()
	require.NoError(err)
	require.Equal(len(naluBytes(spec)), accessUnit.Size())
}

func TestNextNALStillWorksAfterAccessUnitFeatures(t *testing.T) {
	require := require.New(t)

	stream := annexBStream(streamB()...)
	reader := CreateReader(stream, require)

	naluTypes := make([]NalUnitType, 0)

	for {
		nal, err := reader.NextNAL()
		if errors.Is(err, io.EOF) {
			break
		}

		require.NoError(err)
		naluTypes = append(naluTypes, nal.UnitType)
	}

	require.Equal([]NalUnitType{
		NalUnitTypeSPS,
		NalUnitTypePPS,
		NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceNonIdr,
		NalUnitTypeCodedSliceIdr,
	}, naluTypes)
}
