// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package rtpdump

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// virtualClock is a manually advanced Clock for testing the Replayer.
type virtualClock struct {
	mu     sync.Mutex
	now    time.Time
	timers map[*virtualTimer]struct{}
}

type virtualTimer struct {
	deadline time.Time
	ch       chan time.Time
	stopped  bool
}

func newVirtualClock(start time.Time) *virtualClock {
	return &virtualClock{
		now:    start,
		timers: make(map[*virtualTimer]struct{}),
	}
}

func (c *virtualClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.now
}

func (c *virtualClock) NewTimer(d time.Duration) (<-chan time.Time, func() bool) {
	c.mu.Lock()
	t := &virtualTimer{
		deadline: c.now.Add(d),
		ch:       make(chan time.Time, 1),
	}
	c.timers[t] = struct{}{}
	c.mu.Unlock()

	stop := func() bool {
		c.mu.Lock()
		defer c.mu.Unlock()
		if t.stopped {
			return false
		}
		t.stopped = true
		delete(c.timers, t)

		return true
	}

	return t.ch, stop
}

// Advance moves the clock forward and fires every timer whose deadline has
// been reached.
func (c *virtualClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.now = c.now.Add(d)
	for t := range c.timers {
		if t.deadline.After(c.now) {
			continue
		}
		delete(c.timers, t)
		t.stopped = true
		t.ch <- c.now
	}
}

func (c *virtualClock) timerCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	return len(c.timers)
}

func (c *virtualClock) waitForTimers(t *testing.T, n int) {
	t.Helper()
	require.Eventually(t, func() bool {
		return c.timerCount() == n
	}, 5*time.Second, time.Millisecond)
}

// delivery records one callback invocation.
type delivery struct {
	at  time.Duration // virtual time of delivery, relative to clock start
	pkt Packet
}

type recorder struct {
	mu        sync.Mutex
	delivered []delivery
	errAt     int // if > 0, the n-th call (1-based) returns err
	err       error
	clock     *virtualClock
	start     time.Time
}

func (r *recorder) callback(pkt Packet) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.delivered = append(r.delivered, delivery{
		at:  r.clock.Now().Sub(r.start),
		pkt: pkt,
	})
	if r.err != nil && len(r.delivered) == r.errAt {
		return r.err
	}

	return nil
}

func (r *recorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.delivered)
}

func (r *recorder) waitFor(t *testing.T, n int) {
	t.Helper()
	require.Eventually(t, func() bool {
		return r.count() == n
	}, 5*time.Second, time.Millisecond)
}

func (r *recorder) snapshot() []delivery {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]delivery(nil), r.delivered...)
}

// makeRTPPayload builds a minimal RTP packet (V2, PT 96) with the given
// SSRC and sequence number.
func makeRTPPayload(ssrc uint32, seq uint16) []byte {
	payload := make([]byte, 14)
	payload[0] = 0x80
	payload[1] = 96
	binary.BigEndian.PutUint16(payload[2:], seq)
	binary.BigEndian.PutUint32(payload[4:], 1234)
	binary.BigEndian.PutUint32(payload[8:], ssrc)
	payload[12] = 0xAA
	payload[13] = 0xBB

	return payload
}

func makeDump(t *testing.T, packets ...Packet) []byte {
	t.Helper()

	var buf bytes.Buffer
	writer, err := NewWriter(&buf, Header{
		Start:  time.Unix(0, 0).UTC(),
		Source: net.IPv4(127, 0, 0, 1),
		Port:   5004,
	})
	require.NoError(t, err)
	for _, pkt := range packets {
		require.NoError(t, writer.WritePacket(pkt))
	}

	return buf.Bytes()
}

const (
	ssrcA = 1111
	ssrcB = 2222
)

// dualSSRCDump holds two interleaved SSRCs, including two packets sharing
// the 40ms offset to exercise file-order delivery.
func dualSSRCDump(t *testing.T) []byte {
	t.Helper()

	return makeDump(t,
		Packet{Offset: 0, Payload: makeRTPPayload(ssrcA, 1)},
		Packet{Offset: 20 * time.Millisecond, Payload: makeRTPPayload(ssrcB, 1)},
		Packet{Offset: 40 * time.Millisecond, Payload: makeRTPPayload(ssrcA, 2)},
		Packet{Offset: 40 * time.Millisecond, Payload: makeRTPPayload(ssrcB, 2)},
		Packet{Offset: 100 * time.Millisecond, Payload: makeRTPPayload(ssrcA, 3)},
	)
}

func TestReplayerTimingAndOrder(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	type result struct {
		delivered int
		err       error
	}
	resultCh := make(chan result, 1)
	go func() {
		delivered, err := replayer.Play(context.Background(), rec.callback)
		resultCh <- result{delivered, err}
	}()

	// offset 0 packet is delivered immediately.
	rec.waitFor(t, 1)

	clock.waitForTimers(t, 1)
	clock.Advance(20 * time.Millisecond)
	rec.waitFor(t, 2)

	// both 40ms packets fire on the same advance, in file order.
	clock.waitForTimers(t, 1)
	clock.Advance(20 * time.Millisecond)
	rec.waitFor(t, 4)

	clock.waitForTimers(t, 1)
	clock.Advance(60 * time.Millisecond)

	var res result
	select {
	case res = <-resultCh:
	case <-time.After(5 * time.Second):
		t.Fatal("Play did not return after EOF")
	}
	require.NoError(t, res.err)
	assert.Equal(t, 5, res.delivered)

	got := rec.snapshot()
	require.Len(t, got, 5)

	wantTimes := []time.Duration{0, 20 * time.Millisecond, 40 * time.Millisecond, 40 * time.Millisecond, 100 * time.Millisecond}
	wantSSRC := []uint32{ssrcA, ssrcB, ssrcA, ssrcB, ssrcA}
	for i, d := range got {
		assert.Equal(t, wantTimes[i], d.at, "delivery %d time", i)
		assert.Equal(t, wantSSRC[i], binary.BigEndian.Uint32(d.pkt.Payload[8:12]), "delivery %d SSRC", i)
	}
}

func TestReplayerPlaybackRate(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock), WithPlaybackRate(2.0))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	rec.waitFor(t, 1)
	clock.waitForTimers(t, 1)
	clock.Advance(10 * time.Millisecond) // 20ms at 2x
	rec.waitFor(t, 2)
	clock.waitForTimers(t, 1)
	clock.Advance(10 * time.Millisecond) // 40ms at 2x
	rec.waitFor(t, 4)
	clock.waitForTimers(t, 1)
	clock.Advance(30 * time.Millisecond) // 100ms at 2x

	<-done
	require.NoError(t, playErr)
	assert.Equal(t, 5, delivered)

	wantTimes := []time.Duration{0, 10 * time.Millisecond, 20 * time.Millisecond, 20 * time.Millisecond, 50 * time.Millisecond}
	for i, d := range rec.snapshot() {
		assert.Equal(t, wantTimes[i], d.at, "delivery %d time", i)
	}
}

func TestReplayerSSRCFilter(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock), WithSSRC(ssrcB))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	// only ssrcB packets (offsets 20ms and 40ms) are delivered.
	clock.waitForTimers(t, 1)
	clock.Advance(20 * time.Millisecond)
	rec.waitFor(t, 1)
	clock.waitForTimers(t, 1)
	clock.Advance(20 * time.Millisecond)

	<-done
	require.NoError(t, playErr)
	assert.Equal(t, 2, delivered)

	got := rec.snapshot()
	require.Len(t, got, 2)
	assert.Equal(t, 20*time.Millisecond, got[0].at)
	assert.Equal(t, uint32(ssrcB), binary.BigEndian.Uint32(got[0].pkt.Payload[8:12]))
	assert.Equal(t, 40*time.Millisecond, got[1].at)
	assert.Equal(t, uint32(ssrcB), binary.BigEndian.Uint32(got[1].pkt.Payload[8:12]))
}

func TestReplayerCancel(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(ctx, rec.callback)
	}()

	rec.waitFor(t, 1) // offset 0 delivered
	clock.waitForTimers(t, 1)
	cancel()

	<-done
	require.ErrorIs(t, playErr, context.Canceled)
	assert.Equal(t, 1, delivered)

	// no further scheduling happens after cancellation.
	clock.Advance(time.Hour)
	assert.Equal(t, 1, rec.count())
}

func TestReplayerClose(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	rec.waitFor(t, 1)
	clock.waitForTimers(t, 1)
	require.NoError(t, replayer.Close())

	<-done
	require.ErrorIs(t, playErr, ErrClosed)
	assert.Equal(t, 1, delivered)

	// Play after Close fails immediately.
	_, err = replayer.Play(context.Background(), rec.callback)
	require.ErrorIs(t, err, ErrClosed)

	// Close is idempotent.
	require.NoError(t, replayer.Close())
}

func TestReplayerInvalidOptions(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	for _, rate := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err = NewReplayer(reader, WithPlaybackRate(rate))
		require.ErrorIs(t, err, ErrInvalidRate, "rate %v", rate)
	}

	_, err = NewReplayer(reader, WithClock(nil))
	require.ErrorIs(t, err, ErrInvalidClock)

	_, err = NewReplayer(nil)
	require.ErrorIs(t, err, errMalformed)

	replayer, err := NewReplayer(reader)
	require.NoError(t, err)
	_, err = replayer.Play(context.Background(), nil)
	require.ErrorIs(t, err, ErrInvalidCallback)
}

func TestReplayerOffsetRegression(t *testing.T) {
	dump := makeDump(t,
		Packet{Offset: 10 * time.Millisecond, Payload: makeRTPPayload(ssrcA, 1)},
		Packet{Offset: 5 * time.Millisecond, Payload: makeRTPPayload(ssrcA, 2)},
	)
	reader, _, err := NewReader(bytes.NewReader(dump))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	clock.waitForTimers(t, 1)
	clock.Advance(10 * time.Millisecond)

	<-done
	require.ErrorIs(t, playErr, ErrOffsetRegression)
	assert.Equal(t, 1, delivered)
}

func TestReplayerCallbackError(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	errBoom := errors.New("boom")
	rec := &recorder{clock: clock, start: clock.Now(), err: errBoom, errAt: 2}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	rec.waitFor(t, 1)
	clock.waitForTimers(t, 1)
	clock.Advance(20 * time.Millisecond)

	<-done
	require.ErrorIs(t, playErr, errBoom)
	// the failing packet is not counted as delivered.
	assert.Equal(t, 1, delivered)
}

func TestReplayerMalformedRecord(t *testing.T) {
	// one valid packet followed by a truncated packet header.
	dump := append(makeDump(t,
		Packet{Offset: 0, Payload: makeRTPPayload(ssrcA, 1)},
	), 0x00, 0x14)
	reader, _, err := NewReader(bytes.NewReader(dump))
	require.NoError(t, err)

	replayer, err := NewReplayer(reader)
	require.NoError(t, err)

	delivered, err := replayer.Play(context.Background(), func(Packet) error { return nil })
	require.ErrorIs(t, err, errMalformed)
	assert.Equal(t, 1, delivered)
}

func TestReplayerClockJumpForward(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	var delivered int
	var playErr error
	go func() {
		defer close(done)
		delivered, playErr = replayer.Play(context.Background(), rec.callback)
	}()

	rec.waitFor(t, 1)
	clock.waitForTimers(t, 1)
	// jump far past every scheduled offset: all remaining packets are
	// delivered immediately, still in file order.
	clock.Advance(time.Hour)

	<-done
	require.NoError(t, playErr)
	assert.Equal(t, 5, delivered)

	wantSSRC := []uint32{ssrcA, ssrcB, ssrcA, ssrcB, ssrcA}
	for i, d := range rec.snapshot() {
		assert.Equal(t, wantSSRC[i], binary.BigEndian.Uint32(d.pkt.Payload[8:12]), "delivery %d SSRC", i)
	}
}

func TestReplayerConcurrentPlay(t *testing.T) {
	reader, _, err := NewReader(bytes.NewReader(dualSSRCDump(t)))
	require.NoError(t, err)

	clock := newVirtualClock(time.Unix(0, 0))
	replayer, err := NewReplayer(reader, WithClock(clock))
	require.NoError(t, err)

	rec := &recorder{clock: clock, start: clock.Now()}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = replayer.Play(context.Background(), rec.callback)
	}()

	rec.waitFor(t, 1)
	_, err = replayer.Play(context.Background(), rec.callback)
	require.ErrorIs(t, err, ErrAlreadyPlaying)

	require.NoError(t, replayer.Close())
	<-done
}
