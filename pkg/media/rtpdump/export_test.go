// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package rtpdump

import (
	"bytes"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSSRC1 uint32 = 0x11111111
	testSSRC2 uint32 = 0x22222222
)

// rtpPacketForTest builds a minimal RTP version 2 packet with a fixed header
// carrying ssrc followed by payload.
func rtpPacketForTest(t *testing.T, ssrc uint32, seq uint16, payload byte) []byte {
	t.Helper()

	pkt := make([]byte, rtpMinHeaderLen+1)
	pkt[0] = 0x80 // version 2
	pkt[1] = 96
	pkt[2] = byte(seq >> 8)   //nolint:gosec // G115
	pkt[3] = byte(seq)        //nolint:gosec // G115
	pkt[8] = byte(ssrc >> 24) //nolint:gosec // G115
	pkt[9] = byte(ssrc >> 16) //nolint:gosec // G115
	pkt[10] = byte(ssrc >> 8) //nolint:gosec // G115
	pkt[11] = byte(ssrc)      //nolint:gosec // G115
	pkt[12] = payload

	return pkt
}

// buildTestDump writes a dump containing the given packets and returns its
// serialized bytes.
func buildTestDump(t *testing.T, hdr Header, packets []Packet) []byte {
	t.Helper()

	buf := bytes.NewBuffer(nil)

	writer, err := NewWriter(buf, hdr)
	require.NoError(t, err)

	for _, pkt := range packets {
		require.NoError(t, writer.WritePacket(pkt))
	}

	return buf.Bytes()
}

// readAllPackets decodes every packet from a serialized dump.
func readAllPackets(t *testing.T, data []byte) (Header, []Packet) {
	t.Helper()

	reader, hdr, err := NewReader(bytes.NewReader(data))
	require.NoError(t, err)

	var packets []Packet

	for {
		pkt, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		require.NoError(t, nextErr)

		packets = append(packets, pkt)
	}

	return hdr, packets
}

// failingWriter fails every write after the given number of successful writes
// with errWriteInterrupt.
type failingWriter struct {
	remaining int
}

var errWriteInterrupt = errors.New("simulated write interruption")

func (f *failingWriter) Write(p []byte) (int, error) {
	if f.remaining == 0 {
		return 0, errWriteInterrupt
	}
	f.remaining--

	return len(p), nil
}

func TestExport(t *testing.T) {
	hdr := Header{
		Start:  time.Unix(100, 0).UTC(),
		Source: net.IPv4(10, 0, 0, 1),
		Port:   5004,
	}

	rtp1a := rtpPacketForTest(t, testSSRC1, 1, 0xA1)
	rtp2a := rtpPacketForTest(t, testSSRC2, 2, 0xB1)
	rtp1b := rtpPacketForTest(t, testSSRC1, 3, 0xA2)
	rtp2b := rtpPacketForTest(t, testSSRC2, 4, 0xB2)
	rtcp := []byte{0x81, 0xCB, 0x00, 0x0C, 0x90, 0x2F, 0x9E, 0x2E, 0x03, 0x46, 0x4F, 0x4F}

	// The offsets are deliberately non-monotonic and include equal offsets to
	// show that ordering comes from the stream, not the timestamps.
	inputPackets := []Packet{
		{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtp1a},
		{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtp2a},
		{Offset: 30 * time.Millisecond, IsRTCP: true, Payload: rtcp},
		{Offset: 30 * time.Millisecond, IsRTCP: false, Payload: rtp1b},
		{Offset: 15 * time.Millisecond, IsRTCP: false, Payload: rtp2b},
	}
	input := buildTestDump(t, hdr, inputPackets)

	for _, test := range []struct {
		name        string
		opts        ExportOptions
		wantResult  ExportResult
		wantPackets []Packet
	}{
		{
			name: "two SSRCs, half-open window keeps the start boundary",
			opts: ExportOptions{Start: 10 * time.Millisecond, End: 40 * time.Millisecond},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 5,
				OutsideWindow:   0,
			},
			// input order, offsets and payload bytes are all preserved.
			wantPackets: inputPackets,
		},
		{
			name: "end boundary is exclusive",
			opts: ExportOptions{Start: 10 * time.Millisecond, End: 30 * time.Millisecond},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 3,
				OutsideWindow:   2,
			},
			wantPackets: []Packet{
				{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtp1a},
				{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtp2a},
				// the 15ms packet arrives last in the stream but is still
				// inside the window; input order is preserved.
				{Offset: 15 * time.Millisecond, IsRTCP: false, Payload: rtp2b},
			},
		},
		{
			name: "empty window produces a valid empty dump",
			opts: ExportOptions{Start: 100 * time.Millisecond, End: 200 * time.Millisecond},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 0,
				OutsideWindow:   5,
			},
		},
		{
			name: "zero-length window matches nothing",
			opts: ExportOptions{Start: 30 * time.Millisecond, End: 30 * time.Millisecond},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 0,
				OutsideWindow:   5,
			},
		},
		{
			name: "SSRC filter keeps one stream, RTCP always passes through",
			opts: ExportOptions{Start: 0, End: 1 * time.Second, SSRC: testSSRC1},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 3,
				SSRCFiltered:    2,
			},
			wantPackets: []Packet{
				{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtp1a},
				{Offset: 30 * time.Millisecond, IsRTCP: true, Payload: rtcp},
				{Offset: 30 * time.Millisecond, IsRTCP: false, Payload: rtp1b},
			},
		},
		{
			name: "SSRC filter combined with the window",
			opts: ExportOptions{
				Start: 20 * time.Millisecond,
				End:   30 * time.Millisecond,
				SSRC:  testSSRC2,
			},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 1,
				OutsideWindow:   4,
			},
			wantPackets: []Packet{
				{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtp2a},
			},
		},
		{
			name: "rebase rewrites offsets relative to the window start",
			opts: ExportOptions{
				Start:      10 * time.Millisecond,
				End:        40 * time.Millisecond,
				RebaseTime: true,
			},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 5,
			},
			wantPackets: []Packet{
				{Offset: 0, IsRTCP: false, Payload: rtp1a},
				{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtp2a},
				{Offset: 20 * time.Millisecond, IsRTCP: true, Payload: rtcp},
				{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtp1b},
				{Offset: 5 * time.Millisecond, IsRTCP: false, Payload: rtp2b},
			},
		},
		{
			name: "rebase with SSRC filter",
			opts: ExportOptions{
				Start:      15 * time.Millisecond,
				End:        40 * time.Millisecond,
				SSRC:       testSSRC2,
				RebaseTime: true,
			},
			wantResult: ExportResult{
				RecordsRead:     5,
				RecordsExported: 3,
				OutsideWindow:   1,
				SSRCFiltered:    1,
			},
			wantPackets: []Packet{
				{Offset: 5 * time.Millisecond, IsRTCP: false, Payload: rtp2a},
				{Offset: 15 * time.Millisecond, IsRTCP: true, Payload: rtcp},
				{Offset: 0, IsRTCP: false, Payload: rtp2b},
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			out := bytes.NewBuffer(nil)

			result, err := Export(bytes.NewReader(input), out, test.opts)
			require.NoError(t, err)
			assert.Equal(t, test.wantResult, result)

			// The output is always a valid dump whose header is the input
			// header.
			outHdr, outPackets := readAllPackets(t, out.Bytes())
			assert.Equal(t, hdr, outHdr)
			assert.Equal(t, test.wantPackets, outPackets)
		})
	}
}

func TestExportPreservesOriginalBytes(t *testing.T) {
	hdr := Header{Start: time.Unix(7, 0).UTC(), Source: net.IPv4(1, 2, 3, 4), Port: 4000}
	packets := []Packet{
		{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC1, 100, 0xCC)},
		{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC2, 101, 0xDD)},
	}
	input := buildTestDump(t, hdr, packets)

	out := bytes.NewBuffer(nil)

	result, err := Export(bytes.NewReader(input), out, ExportOptions{
		Start:      10 * time.Millisecond,
		End:        30 * time.Millisecond,
		RebaseTime: true,
	})
	require.NoError(t, err)
	assert.Equal(t, 2, result.RecordsExported)

	// Rebase only touches the offset field of the record header: the RTP
	// payload bytes must appear verbatim in the output records.
	_, outPackets := readAllPackets(t, out.Bytes())
	require.Len(t, outPackets, len(packets))
	for i, pkt := range packets {
		assert.Equal(t, pkt.Payload, outPackets[i].Payload)
	}
}

func TestExportErrors(t *testing.T) {
	hdr := Header{Start: time.Unix(1, 0).UTC(), Source: net.IPv4(0, 0, 0, 0)}
	packets := []Packet{
		{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC1, 1, 0x01)},
		{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC1, 2, 0x02)},
		{Offset: 30 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC1, 3, 0x03)},
	}
	input := buildTestDump(t, hdr, packets)

	t.Run("end before start", func(t *testing.T) {
		_, err := Export(bytes.NewReader(input), bytes.NewBuffer(nil), ExportOptions{
			Start: 30 * time.Millisecond,
			End:   10 * time.Millisecond,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.True(t, exportErr.HeaderRecord)
		assert.ErrorIs(t, err, ErrInvalidWindow)
	})

	t.Run("truncated record is located by index", func(t *testing.T) {
		// Drop the last payload byte so the final record's length field points
		// past the end of the stream.
		truncated := make([]byte, len(input)-1)
		copy(truncated, input)

		_, err := Export(bytes.NewReader(truncated), bytes.NewBuffer(nil), ExportOptions{
			Start: 0,
			End:   1 * time.Second,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.Equal(t, 2, exportErr.RecordIndex)
		assert.False(t, exportErr.HeaderRecord)
		assert.ErrorIs(t, err, ErrMalformedRecord)
		assert.ErrorIs(t, err, errMalformed)
	})

	t.Run("bad record length is located by index", func(t *testing.T) {
		// Replace the first packet record's length field with zero, which the
		// Reader rejects as malformed.
		corrupt := append([]byte(nil), input...)
		firstRecordOffset := len("#!rtpplay1.0 0.0.0.0/0\n") + headerLen
		corrupt[firstRecordOffset] = 0x00
		corrupt[firstRecordOffset+1] = 0x00

		_, err := Export(bytes.NewReader(corrupt), bytes.NewBuffer(nil), ExportOptions{
			Start: 0,
			End:   1 * time.Second,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.Equal(t, 0, exportErr.RecordIndex)
		assert.ErrorIs(t, err, ErrMalformedRecord)
	})

	t.Run("malformed input header", func(t *testing.T) {
		_, err := Export(bytes.NewReader([]byte("not a dump")), bytes.NewBuffer(nil), ExportOptions{
			Start: 0,
			End:   1 * time.Second,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.True(t, exportErr.HeaderRecord)
		assert.ErrorIs(t, err, errMalformed)
	})

	t.Run("write interruption while writing the header", func(t *testing.T) {
		_, err := Export(bytes.NewReader(input), &failingWriter{}, ExportOptions{
			Start: 0,
			End:   1 * time.Second,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.True(t, exportErr.HeaderRecord)
		assert.ErrorIs(t, err, ErrWriteFailed)
		assert.ErrorIs(t, err, errWriteInterrupt)
	})

	t.Run("write interruption on the second record is located", func(t *testing.T) {
		// The preamble and the 16-byte header are the first two writes.
		_, err := Export(bytes.NewReader(input), &failingWriter{remaining: 3}, ExportOptions{
			Start: 0,
			End:   1 * time.Second,
		})
		var exportErr *ExportError
		require.ErrorAs(t, err, &exportErr)
		assert.Equal(t, 1, exportErr.RecordIndex)
		assert.False(t, exportErr.HeaderRecord)
		assert.ErrorIs(t, err, ErrWriteFailed)
		assert.ErrorIs(t, err, errWriteInterrupt)
	})

	t.Run("truncated RTP record filtered by SSRC is counted, not an error", func(t *testing.T) {
		withTruncated := []Packet{
			{Offset: 10 * time.Millisecond, IsRTCP: false, Payload: rtpPacketForTest(t, testSSRC1, 1, 0x01)},
			// Shorter than the fixed RTP header: SSRC cannot be read.
			{Offset: 20 * time.Millisecond, IsRTCP: false, Payload: []byte{0x80, 0x60}},
		}
		dump := buildTestDump(t, hdr, withTruncated)

		result, err := Export(bytes.NewReader(dump), bytes.NewBuffer(nil), ExportOptions{
			Start: 0,
			End:   1 * time.Second,
			SSRC:  testSSRC1,
		})
		require.NoError(t, err)
		assert.Equal(t, ExportResult{
			RecordsRead:     2,
			RecordsExported: 1,
			SSRCFiltered:    1,
		}, result)
	})
}

func TestRTPSSRC(t *testing.T) {
	for _, test := range []struct {
		name    string
		payload []byte
		want    uint32
		ok      bool
	}{
		{
			name:    "valid rtp packet",
			payload: rtpPacketForTest(t, 0xDEADBEEF, 1, 0),
			want:    0xDEADBEEF,
			ok:      true,
		},
		{
			name:    "too short for a fixed header",
			payload: []byte{0x80, 0x60, 0x00, 0x01},
			ok:      false,
		},
		{
			name:    "wrong version",
			payload: []byte{0x40, 0x60, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44},
			ok:      false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ssrc, ok := rtpSSRC(test.payload)
			assert.Equal(t, test.ok, ok)
			assert.Equal(t, test.want, ssrc)
		})
	}
}
