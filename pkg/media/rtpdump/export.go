// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package rtpdump

import (
	"errors"
	"io"
	"strconv"
	"time"
)

// rtpMinHeaderLen is the length of the fixed RTP header (RFC 3550), which is
// the smallest packet that can carry an SSRC.
const rtpMinHeaderLen = 12

// rtpSSRCByteOffset is the offset of the 32-bit synchronization source
// identifier inside the fixed RTP header.
const rtpSSRCByteOffset = 8

// rtpVersion is the value of the two-bit version field for RTP version 2.
const rtpVersion = 2

var (
	// ErrInvalidWindow is returned when the export window does not satisfy
	// Start <= End.
	ErrInvalidWindow = errors.New("rtpdump: export window end must not be before start")

	// ErrMalformedRecord is returned when the input stream contains a record
	// that cannot be decoded. It is wrapped in an ExportError carrying the
	// index of the offending record.
	ErrMalformedRecord = errors.New("rtpdump: malformed record")

	// ErrWriteFailed is returned when writing an exported record or the output
	// header fails. It is wrapped in an ExportError carrying the index of the
	// record that could not be written (ExportError.HeaderRecord is true when
	// the failure happened while writing the file header, before any record).
	ErrWriteFailed = errors.New("rtpdump: write failed")
)

// ExportOptions configures a call to Export.
type ExportOptions struct {
	// Start is the inclusive lower bound of the half-open [Start, End)
	// window, measured as an offset relative to the start of recording, the
	// same unit as Packet.Offset.
	Start time.Duration
	// End is the exclusive upper bound of the half-open [Start, End) window.
	// A record whose Offset equals End is not exported.
	End time.Duration
	// SSRC, when non-zero, limits the export to RTP packets carrying this
	// synchronization source identifier. RTCP packets are not filtered and
	// always pass through. A zero value disables SSRC filtering.
	SSRC uint32
	// RebaseTime, when true, rewrites the Offset of every exported packet so
	// that it is measured relative to Start (the first exported packet is
	// written with Offset zero). When false, the original offsets recorded in
	// the input are preserved.
	RebaseTime bool
}

// ExportResult summarizes an Export call. All fields except RecordsExported
// report how many input records were not written and why, so an unexpected
// result can be located without replaying the stream.
type ExportResult struct {
	// RecordsExported is the number of records written to the output.
	RecordsExported int
	// OutsideWindow is the number of records whose Offset fell outside the
	// half-open [Start, End) window.
	OutsideWindow int
	// SSRCFiltered is the number of RTP records that were inside the window but
	// dropped by the SSRC filter, including truncated RTP records whose SSRC
	// could not be read.
	SSRCFiltered int
	// RecordsRead is the total number of records decoded from the input,
	// including records that were filtered out.
	RecordsRead int
}

// ExportError wraps a failure reported by Export together with the position of
// the record involved, so callers can locate the offending record in the
// input stream.
type ExportError struct {
	// RecordIndex is the zero-based index of the input record involved, among
	// the records decoded in order. It is zero when HeaderRecord is true.
	RecordIndex int
	// HeaderRecord is true when the failure happened while writing the output
	// file header, before any record was processed.
	HeaderRecord bool
	// Err is the underlying error. Failures while processing records wrap
	// ErrMalformedRecord or ErrWriteFailed; failures while opening the input
	// or writing the output header carry the underlying error directly.
	Err error
}

// Error implements the error interface.
func (e *ExportError) Error() string {
	if e.HeaderRecord {
		return "rtpdump: export failed while writing header: " + e.Err.Error()
	}

	return "rtpdump: export failed at record " + strconv.Itoa(e.RecordIndex) + ": " + e.Err.Error()
}

// Unwrap exposes the wrapped error for errors.Is and errors.As.
func (e *ExportError) Unwrap() error {
	return e.Err
}

// Export streams the records of an RTPDump read from r into a new, valid
// RTPDump written to w, keeping only the records selected by opts.
//
// Records are processed strictly in input order and each record is held in
// memory only for the duration of its own handling, so the full recording is
// never loaded. The payload bytes of exported packets are copied unchanged:
// the RTP/RTCP bytes on disk are byte-for-byte those of the input.
//
// The window is half-open: a record is inside the window when
// Start <= Offset < End. Records with equal offsets, and records whose
// offsets do not increase monotonically, are valid and need no special
// handling; ordering is taken from the stream, not the offsets. When no
// record matches the window the output is still a valid, empty RTPDump and
// no error is returned.
//
// A malformed or truncated input record, or a write failure, terminates the
// export and is returned as an ExportError whose RecordIndex identifies the
// record being processed when the failure occurred.
func Export(r io.Reader, w io.Writer, opts ExportOptions) (ExportResult, error) {
	if opts.End < opts.Start {
		return ExportResult{}, &ExportError{HeaderRecord: true, Err: ErrInvalidWindow}
	}

	reader, hdr, err := NewReader(r)
	if err != nil {
		return ExportResult{}, &ExportError{HeaderRecord: true, Err: err}
	}

	writer, err := NewWriter(w, hdr)
	if err != nil {
		return ExportResult{}, &ExportError{HeaderRecord: true, Err: errors.Join(ErrWriteFailed, err)}
	}

	var result ExportResult

	for {
		pkt, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			return result, nil
		}
		if nextErr != nil {
			return result, &ExportError{
				RecordIndex: result.RecordsRead,
				Err:         errors.Join(ErrMalformedRecord, nextErr),
			}
		}

		// RecordsRead counts records successfully decoded; the index of the
		// next record is therefore captured before any early continue.
		result.RecordsRead++

		if pkt.Offset < opts.Start || pkt.Offset >= opts.End {
			result.OutsideWindow++

			continue
		}

		if opts.SSRC != 0 && !pkt.IsRTCP {
			ssrc, ok := rtpSSRC(pkt.Payload)
			if !ok || ssrc != opts.SSRC {
				result.SSRCFiltered++

				continue
			}
		}

		if opts.RebaseTime {
			pkt.Offset -= opts.Start
		}

		err = writer.WritePacket(pkt)
		if err != nil {
			return result, &ExportError{
				RecordIndex: result.RecordsRead - 1,
				Err:         errors.Join(ErrWriteFailed, err),
			}
		}

		result.RecordsExported++
	}
}

// rtpSSRC reads the synchronization source identifier from the fixed header of
// an RTP packet. It returns ok false when the payload is too short to hold a
// fixed header or does not carry RTP version 2.
func rtpSSRC(payload []byte) (uint32, bool) {
	if len(payload) < rtpMinHeaderLen || payload[0]>>6 != rtpVersion {
		return 0, false
	}

	return uint32(payload[rtpSSRCByteOffset])<<24 | //nolint:gosec // G115
		uint32(payload[rtpSSRCByteOffset+1])<<16 | //nolint:gosec // G115
		uint32(payload[rtpSSRCByteOffset+2])<<8 | //nolint:gosec // G115
		uint32(payload[rtpSSRCByteOffset+3]), true //nolint:gosec // G115
}
