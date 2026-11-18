// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package ivfreader

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// buildIVFHeader builds a 32-byte IVF file header with the given time base
// and declared frame count. The frame count is deliberately configurable
// because the seek index must trust the frames on disk rather than it.
func buildIVFHeader(denominator, numerator, declaredFrames uint32) []byte {
	header := make([]byte, ivfFileHeaderSize)
	copy(header[0:4], ivfFileHeaderSignature)
	binary.LittleEndian.PutUint16(header[4:6], 0) // version
	binary.LittleEndian.PutUint16(header[6:8], ivfFileHeaderSize)
	copy(header[8:12], "VP80")
	binary.LittleEndian.PutUint16(header[12:14], 176) // width
	binary.LittleEndian.PutUint16(header[14:16], 144) // height
	binary.LittleEndian.PutUint32(header[16:20], denominator)
	binary.LittleEndian.PutUint32(header[20:24], numerator)
	binary.LittleEndian.PutUint32(header[24:28], declaredFrames)

	return header
}

// buildIVFFrame builds one 12-byte frame header followed by payload.
func buildIVFFrame(pts uint64, payload []byte) []byte {
	frame := make([]byte, ivfFrameHeaderSize+len(payload))
	binary.LittleEndian.PutUint32(frame[0:4], uint32(len(payload))) //nolint:gosec // G115
	binary.LittleEndian.PutUint64(frame[4:12], pts)
	copy(frame[ivfFrameHeaderSize:], payload)

	return frame
}

// seekTestContainer describes one frame in the seek test container.
type seekTestFrame struct {
	pts     uint64
	payload []byte
}

func buildSeekContainer(t *testing.T, frames []seekTestFrame, declaredFrames uint32) ([]byte, *bytes.Reader) {
	t.Helper()

	// Time base 30000/1000 (30fps): converted timestamp is pts*30,
	// matching the header used by buildIVFContainer.
	ivf := buildIVFHeader(30000, 1000, declaredFrames)
	for _, frame := range frames {
		ivf = append(ivf, buildIVFFrame(frame.pts, frame.payload)...)
	}

	return ivf, bytes.NewReader(append([]byte(nil), ivf...))
}

func TestIVFReader_SeekToTimestamp(t *testing.T) {
	assert := assert.New(t)

	// Fixed frames with two duplicate raw PTS values (frames 2 and 3 both
	// carry PTS 1). Converted timestamps at 30000/1000 are pts*30:
	// 0, 30, 30, 60, 90.
	frames := []seekTestFrame{
		{pts: 0, payload: []byte{0x01, 0x02}},
		{pts: 1, payload: []byte{0x03, 0x04, 0x05}},
		{pts: 1, payload: []byte{0x06, 0x07}},
		{pts: 2, payload: []byte{0x08}},
		{pts: 3, payload: []byte{0x09, 0x0A, 0x0B, 0x0C}},
	}
	// Header claims a wrong frame count on purpose.
	rawBytes, ivf := buildSeekContainer(t, frames, 999)

	reader, _, err := NewWith(ivf)
	assert.Nil(err, "IVFReader should be created")

	// Every complete frame must be indexed regardless of the wrong frame
	// count declared in the header.
	assert.NoError(reader.SeekToTimestamp(0))
	assert.Len(reader.frameIndex, len(frames), "index must use actual frames, not header NumFrames")

	// First frame: target before the earliest timestamp.
	assert.NoError(reader.SeekToTimestamp(0))
	assert.Equal(reader.frameIndex[0].offset, reader.bytesReadSuccesfully, "seek must park at the indexed frame header offset")
	payload, header, err := reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[0].payload, payload, "frame bytes after seek must match the indexed frame")
	assert.Equal(uint64(0), header.Timestamp)

	// A target strictly before the first frame's timestamp lands there.
	assert.NoError(reader.SeekToTimestamp(0))
	payload, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[0].payload, payload)

	// Exact match of the duplicated timestamp must resolve to the first
	// frame carrying it, in file order.
	assert.NoError(reader.SeekToTimestamp(30))
	assert.Equal(int64(ivfFileHeaderSize)+ivfFrameHeaderSize+2, reader.frameIndex[1].offset)
	assert.Equal(reader.frameIndex[1].offset, reader.bytesReadSuccesfully)
	payload, header, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[1].payload, payload, "duplicate timestamp must resolve to the earliest frame")
	assert.Equal(uint64(30), header.Timestamp)

	// Repeated seek to the same timestamp gives the same frame again,
	// proving seeks do not consume frames.
	assert.NoError(reader.SeekToTimestamp(30))
	payload, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[1].payload, payload)

	// The second frame with the duplicate timestamp is reachable by a
	// sequential ParseNextFrame after the seek.
	payload, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[2].payload, payload)

	// Target strictly between two timestamps picks the later frame.
	assert.NoError(reader.SeekToTimestamp(31))
	payload, header, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[3].payload, payload)
	assert.Equal(uint64(60), header.Timestamp)

	// Seek to the last frame and verify offset and bytes.
	assert.NoError(reader.SeekToTimestamp(90))
	assert.Equal(reader.frameIndex[4].offset, reader.bytesReadSuccesfully)
	payload, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(frames[4].payload, payload)

	// After seeking, the remaining frames read sequentially produce the
	// same bytes as parsing the fixed container from the beginning.
	assert.NoError(reader.SeekToTimestamp(0))
	reference, _, err := NewWith(bytes.NewReader(rawBytes))
	assert.NoError(err)
	for {
		gotPayload, gotHeader, gotErr := reader.ParseNextFrame()
		wantPayload, wantHeader, wantErr := reference.ParseNextFrame()
		assert.Equal(wantErr, gotErr)
		assert.Equal(wantHeader, gotHeader)
		assert.Equal(wantPayload, gotPayload)
		if errors.Is(gotErr, io.EOF) {
			break
		}
		assert.NoError(gotErr)
	}
}

func TestIVFReader_SeekTargetOutOfRange(t *testing.T) {
	assert := assert.New(t)

	frames := []seekTestFrame{
		{pts: 0, payload: []byte{0x01}},
		{pts: 1, payload: []byte{0x02}},
	}
	_, ivf := buildSeekContainer(t, frames, 2)
	reader, _, err := NewWith(ivf)
	assert.Nil(err)

	// Target beyond the last frame: deterministic error and the stream is
	// left untouched, so a sequential parse afterwards starts at frame 1.
	err = reader.SeekToTimestamp(91)
	assert.ErrorIs(err, errSeekTargetOutOfRange)
	payload, _, err := reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal([]byte{0x01}, payload)

	// A later, in-range seek still works after the failed one.
	assert.NoError(reader.SeekToTimestamp(30))
	payload, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal([]byte{0x02}, payload)

	// Target one unit past the last timestamp is out of range as well,
	// and the failed seek does not move the stream away from the end.
	err = reader.SeekToTimestamp(31)
	assert.ErrorIs(err, errSeekTargetOutOfRange)
	_, _, err = reader.ParseNextFrame()
	assert.ErrorIs(err, io.EOF)
}

func TestIVFReader_SeekEmptyFile(t *testing.T) {
	assert := assert.New(t)

	// Valid file header, zero frames.
	ivf := bytes.NewReader(buildIVFHeader(30000, 1000, 0))
	reader, _, err := NewWith(ivf)
	assert.Nil(err)

	err = reader.SeekToTimestamp(0)
	assert.ErrorIs(err, errSeekTargetOutOfRange)
	assert.Empty(reader.frameIndex)
	assert.Equal(int64(ivfFileHeaderSize), reader.indexEndOffset)

	// The failed seek leaves the stream right after the file header, so a
	// sequential parse still reports a clean EOF.
	_, _, err = reader.ParseNextFrame()
	assert.ErrorIs(err, io.EOF)
}

func TestIVFReader_SeekNotSupported(t *testing.T) {
	assert := assert.New(t)

	// bytes.Buffer implements io.Reader but not io.Seeker.
	ivf := buildIVFContainer(&[]byte{})
	reader, _, err := NewWith(ivf)
	assert.Nil(err)

	err = reader.SeekToTimestamp(0)
	assert.ErrorIs(err, errSeekNotSupported)

	// The non-seekable reader still serves the original frame API.
	_, _, err = reader.ParseNextFrame()
	assert.ErrorIs(err, io.EOF)
}

func TestIVFReader_SeekWithTruncatedFrame(t *testing.T) {
	assert := assert.New(t)

	completeFrames := []seekTestFrame{
		{pts: 0, payload: []byte{0x01, 0x02}},
		{pts: 1, payload: []byte{0x03, 0x04}},
	}
	rawIVF, _ := buildSeekContainer(t, completeFrames, 2)
	completeSize := int64(len(rawIVF))

	// Case 1: a frame header that declares 4 payload bytes but provides
	// only 2 (truncated payload).
	truncatedPayload := make([]byte, ivfFrameHeaderSize+2)
	binary.LittleEndian.PutUint32(truncatedPayload[0:4], 4)
	binary.LittleEndian.PutUint64(truncatedPayload[4:12], 2)
	corrupted := bytes.NewReader(append(append([]byte(nil), rawIVF...), truncatedPayload...))
	reader, _, err := NewWith(corrupted)
	assert.Nil(err)

	assert.NoError(reader.SeekToTimestamp(0))
	assert.Len(reader.frameIndex, len(completeFrames), "truncated frame must not be indexed")
	assert.Equal(completeSize, reader.indexEndOffset)

	// Frames before the truncation remain seekable and byte-identical.
	assert.NoError(reader.SeekToTimestamp(30))
	payload, _, err := reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal(completeFrames[1].payload, payload)

	// No frame exists at or after the truncated frame's timestamp.
	assert.ErrorIs(reader.SeekToTimestamp(60), errSeekTargetOutOfRange)

	// Case 2: a frame header cut off mid-header.
	halfHeader := []byte{0x01, 0x00, 0x00, 0x00, 0x00}
	corrupted2 := bytes.NewReader(append(append([]byte(nil), rawIVF...), halfHeader...))
	reader2, _, err := NewWith(corrupted2)
	assert.Nil(err)
	assert.NoError(reader2.SeekToTimestamp(0))
	assert.Len(reader2.frameIndex, len(completeFrames))
	assert.Equal(completeSize, reader2.indexEndOffset)
	assert.ErrorIs(reader2.SeekToTimestamp(60), errSeekTargetOutOfRange)
}

func TestIVFReader_SeekIndexBuildDoesNotConsumeStream(t *testing.T) {
	assert := assert.New(t)

	frames := []seekTestFrame{
		{pts: 0, payload: []byte{0x01}},
		{pts: 1, payload: []byte{0x02}},
	}
	_, ivf := buildSeekContainer(t, frames, 2)
	reader, _, err := NewWith(ivf)
	assert.Nil(err)

	// A sequential reader on an identical container.
	rawIVF, _ := buildSeekContainer(t, frames, 2)
	reference, _, err := NewWith(bytes.NewReader(rawIVF))
	assert.NoError(err)

	// Park the reader at the second frame before any seek, then force an
	// index build with an out-of-range target. The build must restore the
	// previous stream position even though the seek itself fails, so the
	// following parse still starts at the second frame.
	_, _, err = reader.ParseNextFrame()
	assert.NoError(err)
	_, _, refErr := reference.ParseNextFrame()
	assert.NoError(refErr)

	assert.ErrorIs(reader.SeekToTimestamp(math.MaxUint64), errSeekTargetOutOfRange)
	payload, header, parseErr := reader.ParseNextFrame()
	wantPayload, wantHeader, wantErr := reference.ParseNextFrame()
	assert.Equal(wantErr, parseErr)
	assert.Equal(wantHeader, header)
	assert.Equal(wantPayload, payload)
}

func TestIVFReader_SeekTimestampOverflowSaturates(t *testing.T) {
	assert := assert.New(t)

	// Time base 30000/1 keeps the overflow condition reachable with a PTS
	// close to the uint64 boundary.
	header := buildIVFHeader(30000, 1, 2)
	first := buildIVFFrame(0, []byte{0x01})
	hugePTS := uint64(math.MaxUint64/30000) + 1
	second := buildIVFFrame(hugePTS, []byte{0x02})
	ivf := bytes.NewReader(append(append(header, first...), second...))

	reader, _, err := NewWith(ivf)
	assert.Nil(err)
	assert.NoError(reader.SeekToTimestamp(0))
	assert.Len(reader.frameIndex, 2)

	// The overflowing conversion saturates instead of wrapping, so the
	// huge-PTS frame sorts after the zero-PTS frame and is reachable.
	assert.Equal(uint64(0), reader.frameIndex[0].timestamp)
	assert.Equal(uint64(math.MaxUint64), reader.frameIndex[1].timestamp)

	// Saturation stays consistent with ParseNextFrame.
	payload, parseHeader, parseErr := reader.ParseNextFrame()
	assert.NoError(parseErr)
	assert.Equal([]byte{0x01}, payload)
	assert.Equal(uint64(0), parseHeader.Timestamp)
	payload, parseHeader, parseErr = reader.ParseNextFrame()
	assert.NoError(parseErr)
	assert.Equal([]byte{0x02}, payload)
	assert.Equal(uint64(math.MaxUint64), parseHeader.Timestamp)

	// A saturated frame is reachable by seeking to the saturated value.
	assert.NoError(reader.SeekToTimestamp(math.MaxUint64))
	payload, _, parseErr = reader.ParseNextFrame()
	assert.NoError(parseErr)
	assert.Equal([]byte{0x02}, payload)
}

func TestIVFReader_SeekAfterResetReader(t *testing.T) {
	assert := assert.New(t)

	frames := []seekTestFrame{
		{pts: 0, payload: []byte{0x01}},
		{pts: 1, payload: []byte{0x02}},
	}
	_, ivf := buildSeekContainer(t, frames, 2)
	reader, _, err := NewWith(ivf)
	assert.Nil(err)
	assert.NoError(reader.SeekToTimestamp(30))
	assert.Len(reader.frameIndex, 2)

	// ResetReader with another seekable stream must invalidate the old
	// index; the next seek rebuilds it from the replacement stream.
	_, replacement := buildSeekContainer(t, frames[:1], 1)
	reader.ResetReader(func(_ int64) io.Reader {
		return replacement
	})
	assert.False(reader.indexBuilt)
	assert.NoError(reader.SeekToTimestamp(0))
	assert.Len(reader.frameIndex, 1)
	payload, _, err := reader.ParseNextFrame()
	assert.NoError(err)
	assert.Equal([]byte{0x01}, payload)
}
