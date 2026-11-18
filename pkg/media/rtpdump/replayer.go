// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package rtpdump

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"
)

var (
	// ErrClosed is returned by Play when the Replayer has been closed.
	ErrClosed = errors.New("rtpdump: replayer is closed")
	// ErrAlreadyPlaying is returned by Play when another Play call is still in progress.
	ErrAlreadyPlaying = errors.New("rtpdump: Play is already in progress")
	// ErrOffsetRegression is returned by Play when a packet's offset is smaller
	// than the offset of a packet earlier in the file.
	ErrOffsetRegression = errors.New("rtpdump: packet offset regressed")
	// ErrInvalidRate is returned when a playback rate is zero, negative, NaN or infinite.
	ErrInvalidRate = errors.New("rtpdump: playback rate must be a positive finite number")
	// ErrInvalidClock is returned when a nil Clock is configured.
	ErrInvalidClock = errors.New("rtpdump: clock must not be nil")
	// ErrInvalidCallback is returned when Play is called with a nil packet callback.
	ErrInvalidCallback = errors.New("rtpdump: packet callback must not be nil")
)

// Clock abstracts the passage of time so that tests (or simulations) can
// drive a Replayer with a virtual clock. NewTimer must behave like
// time.NewTimer: the returned channel receives a single value once d has
// elapsed, and the returned stop function prevents the send.
type Clock interface {
	Now() time.Time
	NewTimer(d time.Duration) (ch <-chan time.Time, stop func() bool)
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

func (realClock) NewTimer(d time.Duration) (<-chan time.Time, func() bool) {
	t := time.NewTimer(d)

	return t.C, t.Stop
}

// ReplayerOption configures a Replayer.
type ReplayerOption func(*Replayer) error

// WithPlaybackRate scales the playback speed relative to the recorded
// offsets: 1.0 replays in real time, 2.0 twice as fast, 0.5 at half speed.
// The rate must be a positive finite number, anything else makes NewReplayer
// fail with ErrInvalidRate.
func WithPlaybackRate(rate float64) ReplayerOption {
	return func(r *Replayer) error {
		if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			return fmt.Errorf("%w, got %v", ErrInvalidRate, rate)
		}
		r.rate = rate

		return nil
	}
}

// WithSSRC restricts delivery to RTP packets carrying the given SSRC.
// RTCP packets and RTP packets too short to contain an SSRC are skipped
// when a filter is set. Skipped packets do not count as delivered.
func WithSSRC(ssrc uint32) ReplayerOption {
	return func(r *Replayer) error {
		r.filterSSRC = ssrc
		r.filter = true

		return nil
	}
}

// WithClock sets the Clock used to schedule packet delivery. The default
// is the real wall clock. A nil clock makes NewReplayer fail with
// ErrInvalidClock.
func WithClock(clock Clock) ReplayerOption {
	return func(r *Replayer) error {
		if clock == nil {
			return ErrInvalidClock
		}
		r.clock = clock

		return nil
	}
}

// Replayer streams packets from a Reader and delivers them through a
// callback, paced by the recorded packet offsets. Packets are read from the
// Reader one at a time, so arbitrarily large dumps can be replayed without
// buffering the whole file. Packets that share an offset are delivered in
// file order.
type Replayer struct {
	reader     *Reader
	rate       float64
	filterSSRC uint32
	filter     bool
	clock      Clock

	mu      sync.Mutex
	playing bool
	closed  bool
	closeCh chan struct{}
}

// NewReplayer creates a Replayer that streams packets from r. By default it
// replays every packet in real time against the wall clock.
func NewReplayer(r *Reader, opts ...ReplayerOption) (*Replayer, error) {
	if r == nil {
		return nil, errMalformed
	}

	p := &Replayer{
		reader:  r,
		rate:    1.0,
		clock:   realClock{},
		closeCh: make(chan struct{}),
	}
	for _, opt := range opts {
		if err := opt(p); err != nil {
			return nil, err
		}
	}

	return p, nil
}

// Play reads packets from the Reader and invokes onPacket with each one,
// scheduled so that a packet recorded at offset o is delivered when
// o/rate has elapsed on the configured Clock since Play started. Play
// returns the number of packets delivered to onPacket.
//
// Play returns nil once the Reader reaches EOF. A malformed record aborts
// playback with the Reader's error. A packet whose offset is smaller than a
// preceding packet's aborts playback with ErrOffsetRegression. If onPacket
// returns an error, playback stops and the error is returned (the failed
// packet is not counted as delivered). Cancelling ctx stops scheduling and
// returns the context error; Close stops scheduling and returns ErrClosed.
// Cancellation and Close are observed between packets: a blocked Reader
// read is not interrupted.
//
// Delivery times are computed from the Clock on every packet. If the clock
// jumps forward past a packet's scheduled time the packet is delivered
// immediately; a backward jump simply extends the wait.
func (p *Replayer) Play(ctx context.Context, onPacket func(Packet) error) (int, error) {
	if onPacket == nil {
		return 0, ErrInvalidCallback
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()

		return 0, ErrClosed
	}
	if p.playing {
		p.mu.Unlock()

		return 0, ErrAlreadyPlaying
	}
	p.playing = true
	p.mu.Unlock()

	defer func() {
		p.mu.Lock()
		p.playing = false
		p.mu.Unlock()
	}()

	start := p.clock.Now()
	delivered := 0
	lastOffset := time.Duration(-1)

	for {
		pkt, err := p.reader.Next()
		if errors.Is(err, io.EOF) {
			return delivered, nil
		}
		if err != nil {
			return delivered, err
		}

		if pkt.Offset < lastOffset {
			return delivered, fmt.Errorf("%w: %s follows %s", ErrOffsetRegression, pkt.Offset, lastOffset)
		}
		lastOffset = pkt.Offset

		if p.filter && !matchSSRC(pkt, p.filterSSRC) {
			continue
		}

		if wait := p.waitDuration(start, pkt.Offset); wait > 0 {
			ch, stop := p.clock.NewTimer(wait)
			select {
			case <-ch:
			case <-ctx.Done():
				stop()

				return delivered, ctx.Err() //nolint:wrapcheck // return the caller's own error unwrapped
			case <-p.closeCh:
				stop()

				return delivered, ErrClosed
			}
			stop()
		} else if err := p.interrupted(ctx); err != nil {
			return delivered, err
		}

		if err := onPacket(pkt); err != nil {
			return delivered, fmt.Errorf("rtpdump: packet callback: %w", err)
		}
		delivered++
	}
}

// waitDuration returns how long to wait on the clock before a packet with
// the given recorded offset is due.
func (p *Replayer) waitDuration(start time.Time, offset time.Duration) time.Duration {
	scaled := time.Duration(float64(offset) / p.rate)

	return start.Add(scaled).Sub(p.clock.Now())
}

// interrupted reports whether cancellation or Close is pending, for packets
// that are already due and would otherwise never block.
func (p *Replayer) interrupted(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err() //nolint:wrapcheck // return the caller's own error unwrapped
	case <-p.closeCh:
		return ErrClosed
	default:
		return nil
	}
}

// Close stops any in-progress Play (which returns ErrClosed) and makes
// later Play calls fail with ErrClosed. Close is idempotent.
func (p *Replayer) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}
	p.closed = true
	close(p.closeCh)

	return nil
}

// matchSSRC reports whether pkt is an RTP packet carrying ssrc. Payloads
// too short to contain a full RTP fixed header (e.g. truncated records)
// never match.
func matchSSRC(pkt Packet, ssrc uint32) bool {
	const rtpFixedHeaderLen = 12

	if pkt.IsRTCP || len(pkt.Payload) < rtpFixedHeaderLen {
		return false
	}
	if pkt.Payload[0]>>6 != 2 { // RTP version 2
		return false
	}

	return binary.BigEndian.Uint32(pkt.Payload[8:12]) == ssrc
}
