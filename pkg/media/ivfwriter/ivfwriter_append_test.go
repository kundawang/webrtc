// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package ivfwriter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4/pkg/media/ivfreader"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// memFile is a readable, writable and seekable in-memory file that also
// supports Truncate, mirroring the subset of *os.File used by Append.
type memFile struct {
	data   []byte
	offset int64
	fail   bool
}

var (
	errSimulatedWrite   = errors.New("simulated write failure")
	errNegativePosition = errors.New("negative position")
	errInvalidSize      = errors.New("invalid truncate size")
)

func (m *memFile) Read(buf []byte) (int, error) {
	if m.offset >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(buf, m.data[m.offset:])
	m.offset += int64(n)

	return n, nil
}

func (m *memFile) Write(buf []byte) (int, error) {
	if m.fail {
		return 0, errSimulatedWrite
	}
	end := m.offset + int64(len(buf))
	if end > int64(cap(m.data)) {
		grown := make([]byte, end, 2*end+1)
		copy(grown, m.data)
		m.data = grown
	}
	if end > int64(len(m.data)) {
		m.data = m.data[:end]
	}
	copy(m.data[m.offset:end], buf)
	m.offset = end

	return len(buf), nil
}

func (m *memFile) Seek(offset int64, whence int) (int64, error) {
	var target int64
	switch whence {
	case io.SeekStart:
		target = offset
	case io.SeekCurrent:
		target = m.offset + offset
	case io.SeekEnd:
		target = int64(len(m.data)) + offset
	}
	if target < 0 {
		return 0, errNegativePosition
	}
	m.offset = target

	return target, nil
}

func (m *memFile) Truncate(size int64) error {
	if size < 0 || size > int64(len(m.data)) {
		return errInvalidSize
	}
	m.data = m.data[:size]
	// *os.File.Truncate does not move the file offset, and the append flow
	// always re-seeks before writing, so keep the offset untouched.
	return nil
}

func (m *memFile) Close() error { return nil }

func (m *memFile) Len() int { return len(m.data) }

func (m *memFile) Bytes() []byte { return m.data }

type parsedFrame struct {
	payload []byte
	pts     uint64
}

// parseIVF decodes an IVF container without depending on the writer: it
// validates the header and returns every frame with its raw PTS.
func parseIVF(t *testing.T, data []byte) (ivfreader.IVFFileHeader, []parsedFrame) {
	t.Helper()
	require.GreaterOrEqual(t, len(data), ivfFileHeaderSize)

	header := ivfreader.IVFFileHeader{
		FourCC:              string(data[8:12]),
		Width:               binary.LittleEndian.Uint16(data[12:14]),
		Height:              binary.LittleEndian.Uint16(data[14:16]),
		TimebaseDenominator: binary.LittleEndian.Uint32(data[16:20]),
		TimebaseNumerator:   binary.LittleEndian.Uint32(data[20:24]),
		NumFrames:           binary.LittleEndian.Uint32(data[24:28]),
	}
	assert.Equal(t, "DKIF", string(data[0:4]))
	assert.Equal(t, uint16(0), binary.LittleEndian.Uint16(data[4:6]))
	assert.Equal(t, uint16(32), binary.LittleEndian.Uint16(data[6:8]))

	frames := []parsedFrame{}
	offset := ivfFileHeaderSize
	for offset < len(data) {
		require.GreaterOrEqual(t, len(data)-offset, ivfFrameHeaderSize)
		size := binary.LittleEndian.Uint32(data[offset : offset+4])
		pts := binary.LittleEndian.Uint64(data[offset+4 : offset+12])
		offset += ivfFrameHeaderSize
		require.GreaterOrEqual(t, len(data)-offset, int(size))
		frames = append(frames, parsedFrame{payload: append([]byte(nil), data[offset:offset+int(size)]...), pts: pts})
		offset += int(size)
	}
	assert.Equal(t, len(data), offset)

	return header, frames
}

func vp8Packet(timestamp uint32, keyframe bool, payload byte) *rtp.Packet {
	// VP8 payload descriptor byte 0x10 (S=1) followed by a VP8 payload whose
	// first bit distinguishes keyframe (0) from interframe (1).
	frame := []byte{0x10, 0x00, 0x00, 0x9d, 0x01, payload}
	if !keyframe {
		frame[1] = 0x01
	}

	return &rtp.Packet{
		Header:  rtp.Header{Marker: true, Timestamp: timestamp},
		Payload: frame,
	}
}

func vp9Packet(timestamp uint32, keyframe bool, payload byte) *rtp.Packet {
	// VP9 descriptor with I=1 (picture ID present) and B=1 (frame start),
	// matching the layout used by TestIVFWriter_VP9.
	descriptor := []byte{0x88, 0x80, 0x00}
	if !keyframe {
		descriptor[0] |= 0x40 // P=1
	}

	return &rtp.Packet{
		Header:  rtp.Header{Marker: true, Timestamp: timestamp},
		Payload: append(descriptor, payload),
	}
}

func TestIVFWriter_Append_VP8(t *testing.T) {
	testAppendRTP(t, mimeTypeVP8, codecVP8, vp8Packet, 1)
}

func TestIVFWriter_Append_VP9(t *testing.T) {
	testAppendRTP(t, mimeTypeVP9, codecVP9, vp9Packet, 3)
}

// testAppendRTP builds a small file over three writer sessions (two resumes)
// and checks that no existing byte changes, every frame is preserved and the
// RTP timestamps of each new session are shifted after the on-disk PTS.
func testAppendRTP(
	t *testing.T,
	mimeType string,
	wantCodec codec,
	makePacket func(timestamp uint32, keyframe bool, payload byte) *rtp.Packet,
	descriptorSize int,
) {
	t.Helper()
	file := &memFile{}

	// All sessions use direct PTS so RTP timestamps map 1:1 to IVF PTS.
	opts := func() []Option {
		return []Option{WithCodec(mimeType), WithFrameRate(1, 90000), WithDirectPTS()}
	}

	type sessionFrame struct {
		timestamp uint32
		keyframe  bool
		payload   byte
	}
	sessions := [][]sessionFrame{
		{{0, true, 0xA1}, {3000, false, 0xA2}},
		{{0, false, 0xB1}, {3000, false, 0xB2}},
		{{0, false, 0xC1}, {3000, false, 0xC2}},
	}

	wantFrames := []parsedFrame{}
	wantPTS := []uint64{0, 3000, 3000, 6000, 6000, 9000}

	for sessionIdx, frames := range sessions {
		var writer *IVFWriter
		var err error
		if sessionIdx == 0 {
			writer, err = NewWith(file, opts()...)
		} else {
			writer, err = AppendWith(file, opts()...)
		}
		require.NoError(t, err)
		require.Equal(t, wantCodec, writer.codec)
		require.Equal(t, uint64(len(wantFrames)), writer.count)
		if sessionIdx > 0 {
			require.True(t, writer.resuming)
			require.Equal(t, wantPTS[len(wantFrames)-1], writer.lastPTS)
			require.True(t, writer.seenKeyFrame)
		}

		prefix := append([]byte(nil), file.Bytes()...)
		for idx, frame := range frames {
			require.NoError(t, writer.WriteRTP(makePacket(frame.timestamp, frame.keyframe, frame.payload)))
			// Existing bytes must never be modified.
			assert.True(t, bytes.HasPrefix(file.Bytes(), prefix))
			prefix = append([]byte(nil), file.Bytes()...)

			pkt := makePacket(frame.timestamp, frame.keyframe, frame.payload)
			wantFrames = append(wantFrames, parsedFrame{
				payload: append([]byte(nil), pkt.Payload[descriptorSize:]...),
				pts:     wantPTS[sessionIdx*2+idx],
			})
		}
		require.NoError(t, writer.Close())
		header, gotFrames := parseIVF(t, file.Bytes())
		assert.Equal(t, len(wantFrames), int(header.NumFrames))
		assert.Equal(t, len(wantFrames), len(gotFrames))
		assert.Equal(t, wantFrames, gotFrames)
	}

	// The finished file must be parseable by the public ivfreader package.
	reader, readerHeader, err := ivfreader.NewWith(bytes.NewReader(file.Bytes()))
	require.NoError(t, err)
	assert.Equal(t, len(wantFrames), int(readerHeader.NumFrames))
	for idx := range wantFrames {
		payload, frameHeader, frameErr := reader.ParseNextFrame()
		require.NoError(t, frameErr)
		assert.Equal(t, wantFrames[idx].payload, payload)
		// ivfreader converts PTS back to timebase units:
		// timestamp = PTS * denominator / numerator.
		assert.Equal(t, wantFrames[idx].pts*90000, frameHeader.Timestamp)
	}
	_, _, err = reader.ParseNextFrame()
	assert.ErrorIs(t, err, io.EOF)

	// Reopen once more without writing: Close must keep the file untouched.
	writer, err := AppendWith(file, opts()...)
	require.NoError(t, err)
	assert.Equal(t, uint64(len(wantFrames)), writer.count)
	before := append([]byte(nil), file.Bytes()...)
	require.NoError(t, writer.Close())
	assert.Equal(t, before, file.Bytes())
}

func TestIVFWriter_Append_WriteFrame(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{0xAA, 0xBB}, 100))
	require.NoError(t, writer.Close())

	// Direct frame append with explicit PTS: no RTP offset is applied.
	writer, err = AppendWith(file, WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	assert.Equal(t, uint64(1), writer.count)
	assert.Equal(t, uint64(100), writer.lastPTS)
	require.NoError(t, writer.WriteFrame([]byte{0xCC}, 250))
	require.NoError(t, writer.Close())

	header, frames := parseIVF(t, file.Bytes())
	assert.Equal(t, 2, int(header.NumFrames))
	require.Equal(t, []parsedFrame{
		{payload: []byte{0xAA, 0xBB}, pts: 100},
		{payload: []byte{0xCC}, pts: 250},
	}, frames)
}

func TestIVFWriter_Append_LegacyTimebase(t *testing.T) {
	// Default 1/30 timebase with millisecond timestamps: resumed RTP frames
	// must still land after the frames already on disk.
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8))
	require.NoError(t, err)
	require.NoError(t, writer.WriteRTP(vp8Packet(0, true, 0x01)))
	// One second of RTP clock later: 1000 ms -> PTS 33.
	require.NoError(t, writer.WriteRTP(vp8Packet(90000, false, 0x02)))
	require.NoError(t, writer.Close())

	_, frames := parseIVF(t, file.Bytes())
	firstSessionLastPTS := frames[len(frames)-1].pts
	require.Equal(t, uint64(33), firstSessionLastPTS)

	writer, err = AppendWith(file)
	require.NoError(t, err)
	// A brand new RTP stream restarts at timestamp 0. Without the resume
	// offset its PTS would be 0 and go backwards.
	require.NoError(t, writer.WriteRTP(vp8Packet(0, false, 0x03)))
	assert.Equal(t, firstSessionLastPTS, writer.ptsOffset)
	require.NoError(t, writer.Close())

	_, frames = parseIVF(t, file.Bytes())
	require.Len(t, frames, 3)
	assert.Equal(t, firstSessionLastPTS, frames[2].pts, "resumed PTS must not go backwards")
	assert.GreaterOrEqual(t, frames[2].pts, frames[1].pts)
}

func TestIVFWriter_Append_HeaderOnlyFile(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP9), WithWidthAndHeight(320, 240))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	assert.Equal(t, ivfFileHeaderSize, file.Len())

	writer, err = AppendWith(file)
	require.NoError(t, err)
	assert.Equal(t, uint64(0), writer.count)
	assert.False(t, writer.seenKeyFrame)

	// With no frame on disk yet the gating behaves like a fresh writer:
	// non-keyframes are dropped until the first keyframe arrives.
	require.NoError(t, writer.WriteRTP(vp9Packet(0, false, 0x10)))
	assert.Equal(t, uint64(0), writer.count)
	require.NoError(t, writer.WriteRTP(vp9Packet(3000, true, 0x11)))
	require.NoError(t, writer.Close())

	header, frames := parseIVF(t, file.Bytes())
	assert.Equal(t, "VP90", header.FourCC)
	assert.Equal(t, uint16(320), header.Width)
	assert.Equal(t, uint16(240), header.Height)
	require.Len(t, frames, 1)
}

func TestIVFWriter_Append_ConfigConflict(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8), WithWidthAndHeight(640, 480), WithFrameRate(1, 30))
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{1}, 0))
	require.NoError(t, writer.Close())

	_, err = AppendWith(file, WithCodec(mimeTypeVP9))
	assert.ErrorIs(t, err, errInvalidIVFHeader)

	_, err = AppendWith(file, WithWidthAndHeight(320, 480))
	assert.ErrorIs(t, err, errInvalidIVFHeader)

	_, err = AppendWith(file, WithWidthAndHeight(640, 240))
	assert.ErrorIs(t, err, errInvalidIVFHeader)

	_, err = AppendWith(file, WithFrameRate(1, 60))
	assert.ErrorIs(t, err, errInvalidIVFHeader)

	_, err = AppendWith(file, WithFrameRate(2, 30))
	assert.ErrorIs(t, err, errInvalidIVFHeader)

	// Options matching the header are accepted and the file is unchanged.
	before := append([]byte(nil), file.Bytes()...)
	writer, err = AppendWith(file, WithCodec(mimeTypeVP8), WithWidthAndHeight(640, 480), WithFrameRate(1, 30))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	assert.Equal(t, before, file.Bytes())
}

func TestIVFWriter_Append_TruncatedTail(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{0xAA, 0xBB}, 10))
	require.NoError(t, writer.WriteFrame([]byte{0xCC}, 20))
	require.NoError(t, writer.Close())

	completeLen := file.Len()

	// Simulate a crash in the middle of the second frame: keep the first
	// frame intact and append a partial frame header and payload.
	firstFrameEnd := ivfFileHeaderSize + ivfFrameHeaderSize + 2
	_, err = file.Seek(int64(firstFrameEnd), io.SeekStart)
	require.NoError(t, err)
	require.NoError(t, file.Truncate(int64(firstFrameEnd)))
	corrupt := append(
		[]byte{0x05, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		0xDD, 0xEE,
	)
	_, err = file.Write(corrupt)
	require.NoError(t, err)

	// First resume reports the truncated tail and removes it.
	writer, err = AppendWith(file)
	assert.Nil(t, writer)
	assert.ErrorIs(t, err, errTruncatedFrame)
	assert.Equal(t, firstFrameEnd, file.Len())

	// Second resume succeeds and only the first complete frame remains.
	writer, err = AppendWith(file)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), writer.count)
	assert.Equal(t, uint64(10), writer.lastPTS)
	require.NoError(t, writer.WriteFrame([]byte{0xFF}, 30))
	require.NoError(t, writer.Close())

	header, frames := parseIVF(t, file.Bytes())
	assert.Equal(t, 2, int(header.NumFrames))
	require.Equal(t, []parsedFrame{
		{payload: []byte{0xAA, 0xBB}, pts: 10},
		{payload: []byte{0xFF}, pts: 30},
	}, frames)
	assert.LessOrEqual(t, file.Len(), completeLen)
}

// seekOnlyMemFile wraps memFile but hides Truncate so AppendWith falls back
// to seeking over the truncated tail.
type seekOnlyMemFile struct {
	inner *memFile
}

func (s *seekOnlyMemFile) Read(buf []byte) (int, error)  { return s.inner.Read(buf) }
func (s *seekOnlyMemFile) Write(buf []byte) (int, error) { return s.inner.Write(buf) }
func (s *seekOnlyMemFile) Seek(offset int64, whence int) (int64, error) {
	return s.inner.Seek(offset, whence)
}

func TestIVFWriter_Append_TruncatedTailWithoutTruncate(t *testing.T) {
	inner := &memFile{}
	file := &seekOnlyMemFile{inner: inner}

	writer, err := NewWith(inner, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{0xAA, 0xBB}, 10))
	require.NoError(t, writer.Close())

	firstFrameEnd := ivfFileHeaderSize + ivfFrameHeaderSize + 2
	_, err = inner.Seek(int64(firstFrameEnd), io.SeekStart)
	require.NoError(t, err)
	require.NoError(t, inner.Truncate(int64(firstFrameEnd)))
	_, err = file.Write([]byte{0x05, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xDD})
	require.NoError(t, err)

	// The tail cannot be removed: the error is still returned and the stream
	// is positioned at the last valid boundary.
	writer, err = AppendWith(file)
	assert.Nil(t, writer)
	assert.ErrorIs(t, err, errTruncatedFrame)
	pos, err := file.Seek(0, io.SeekCurrent)
	require.NoError(t, err)
	assert.Equal(t, int64(firstFrameEnd), pos)

	// Truncate manually as a caller would and resume successfully.
	require.NoError(t, inner.Truncate(int64(firstFrameEnd)))
	writer, err = AppendWith(file)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), writer.count)
	require.NoError(t, writer.WriteFrame([]byte{0xFF}, 20))
	require.NoError(t, writer.Close())

	_, frames := parseIVF(t, inner.Bytes())
	require.Equal(t, []parsedFrame{
		{payload: []byte{0xAA, 0xBB}, pts: 10},
		{payload: []byte{0xFF}, pts: 20},
	}, frames)
}

func TestIVFWriter_Append_FrameCountMismatch(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{1, 2}, 0))
	require.NoError(t, writer.WriteFrame([]byte{3, 4}, 100))
	require.NoError(t, writer.Close())

	// Corrupt the header frame count: resume must trust the frames actually
	// present and Close must rewrite the correct total count.
	_, err = file.Seek(24, io.SeekStart)
	require.NoError(t, err)
	_, err = file.Write([]byte{0x09, 0x00, 0x00, 0x00})
	require.NoError(t, err)

	writer, err = AppendWith(file)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), writer.count)
	require.NoError(t, writer.WriteFrame([]byte{5}, 200))
	require.NoError(t, writer.Close())

	header, frames := parseIVF(t, file.Bytes())
	assert.Equal(t, uint32(3), header.NumFrames)
	assert.Len(t, frames, 3)
}

func TestIVFWriter_Append_PTSRegression(t *testing.T) {
	// Regression inside the existing file: resume is refused and the file is
	// not modified. The file is assembled by hand because WriteFrame itself
	// rejects non-monotonic PTS.
	file := &memFile{}
	headerWriter, err := NewWith(file, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, headerWriter.Close())

	frame := func(payload byte, pts uint64) []byte {
		buf := make([]byte, ivfFrameHeaderSize+1)
		binary.LittleEndian.PutUint32(buf[0:4], 1)
		binary.LittleEndian.PutUint64(buf[4:12], pts)
		buf[ivfFrameHeaderSize] = payload

		return buf
	}
	_, err = file.Seek(0, io.SeekEnd)
	require.NoError(t, err)
	_, err = file.Write(frame(1, 100))
	require.NoError(t, err)
	_, err = file.Write(frame(2, 50))
	require.NoError(t, err)

	before := append([]byte(nil), file.Bytes()...)
	_, err = AppendWith(file)
	assert.ErrorIs(t, err, errPTSRegression)
	assert.Equal(t, before, file.Bytes())

	// Regression on a newly appended frame via WriteFrame.
	file2 := &memFile{}
	writer, err := NewWith(file2, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{1}, 100))
	require.NoError(t, writer.Close())

	writer, err = AppendWith(file2)
	require.NoError(t, err)
	assert.ErrorIs(t, writer.WriteFrame([]byte{2}, 99), errPTSRegression)
	// The rejected frame is not counted.
	assert.Equal(t, uint64(1), writer.count)
	// Equal PTS is allowed; nothing was written by the rejected call.
	require.NoError(t, writer.WriteFrame([]byte{2}, 100))
	require.NoError(t, writer.Close())

	_, frames := parseIVF(t, file2.Bytes())
	require.Len(t, frames, 2)
	assert.Equal(t, []uint64{100, 100}, []uint64{frames[0].pts, frames[1].pts})
}

func TestIVFWriter_Append_InvalidFiles(t *testing.T) {
	t.Run("nil stream", func(t *testing.T) {
		writer, err := AppendWith(nil)
		assert.Nil(t, writer)
		assert.ErrorIs(t, err, errFileNotOpened)
	})

	t.Run("empty file", func(t *testing.T) {
		writer, err := AppendWith(&memFile{})
		assert.Nil(t, writer)
		assert.ErrorIs(t, err, errEmptyFile)
	})

	t.Run("truncated header", func(t *testing.T) {
		file := &memFile{}
		_, err := file.Write(bytes.Repeat([]byte{0x44}, 10))
		require.NoError(t, err)
		writer, err := AppendWith(file)
		assert.Nil(t, writer)
		assert.ErrorIs(t, err, errInvalidIVFHeader)
	})

	t.Run("bad signature", func(t *testing.T) {
		file := &memFile{}
		writer, err := NewWith(file)
		require.NoError(t, err)
		require.NoError(t, writer.Close())
		file.data[0] = 'X'
		_, err = AppendWith(file)
		assert.ErrorIs(t, err, errInvalidIVFHeader)
	})

	t.Run("unknown FourCC", func(t *testing.T) {
		file := &memFile{}
		writer, err := NewWith(file)
		require.NoError(t, err)
		require.NoError(t, writer.Close())
		copy(file.data[8:12], "XXXX")
		_, err = AppendWith(file)
		assert.ErrorIs(t, err, errInvalidIVFHeader)
	})
}

func TestIVFWriter_Append_WriteFailure(t *testing.T) {
	file := &memFile{}

	writer, err := NewWith(file, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{1}, 0))
	require.NoError(t, writer.Close())

	writer, err = AppendWith(file)
	require.NoError(t, err)
	file.fail = true
	assert.Error(t, writer.WriteFrame([]byte{2}, 10))
	assert.Error(t, writer.WriteRTP(vp8Packet(3000, false, 0x02)))
}

func TestIVFWriter_Append_OnDisk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "recording.ivf")

	writer, err := New(path, WithCodec(mimeTypeVP8), WithFrameRate(1, 90000), WithDirectPTS())
	require.NoError(t, err)
	require.NoError(t, writer.WriteFrame([]byte{0x01, 0x02}, 0))
	require.NoError(t, writer.Close())

	// Simulate a process restart: reopen the same file and keep recording.
	writer, err = Append(path)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), writer.count)
	require.NoError(t, writer.WriteFrame([]byte{0x03}, 9000))
	require.NoError(t, writer.Close())

	written, err := os.ReadFile(filepath.Clean(path)) //nolint:gosec // test-owned temp file
	require.NoError(t, err)
	reader, header, err := ivfreader.NewWith(bytes.NewReader(written))
	require.NoError(t, err)
	assert.Equal(t, uint32(2), header.NumFrames)

	payload, frameHeader, err := reader.ParseNextFrame()
	require.NoError(t, err)
	assert.Equal(t, []byte{0x01, 0x02}, payload)
	assert.Equal(t, uint64(0), frameHeader.Timestamp)

	payload, frameHeader, err = reader.ParseNextFrame()
	require.NoError(t, err)
	assert.Equal(t, []byte{0x03}, payload)
	// ivfreader reports timebase units (PTS * denominator / numerator).
	assert.Equal(t, uint64(9000)*90000, frameHeader.Timestamp)

	// A third session proves the file can be reopened repeatedly.
	writer, err = Append(path)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), writer.count)
	require.NoError(t, writer.Close())
}
