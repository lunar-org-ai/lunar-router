package server

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/OpenTracy/opentracy/go/internal/provider"
)

// TestGenerateSessionID_RebrandedPrefix guards the rebrand: session IDs are
// handed to clients via `X-OpenTracy-Session-Id` and stored in ClickHouse,
// so flipping the prefix back to "lunar-session-" would silently diverge
// live traffic from what's persisted. A single string change is easy to
// revert by accident — this asserts the invariant.
func TestGenerateSessionID_RebrandedPrefix(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 8; i++ {
		id := GenerateSessionID()
		if !strings.HasPrefix(id, "opentracy-session-") {
			t.Fatalf("GenerateSessionID() = %q, want prefix %q", id, "opentracy-session-")
		}
		if strings.HasPrefix(id, "lunar-session-") {
			t.Fatalf("GenerateSessionID() returned legacy prefix: %q", id)
		}
		if seen[id] {
			t.Fatalf("GenerateSessionID() returned duplicate %q — randomness broken", id)
		}
		seen[id] = true
	}
}

func TestToolCallSessionTryFinalizeConcurrent(t *testing.T) {
	session := &ToolCallSession{}

	const workers = 64
	var wg sync.WaitGroup
	var successCount int32

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			if session.TryFinalize() {
				atomic.AddInt32(&successCount, 1)
			}
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt32(&successCount); got != 1 {
		t.Fatalf("TryFinalize success count = %d, want 1", got)
	}
	if !session.Finalized {
		t.Fatal("session.Finalized = false, want true")
	}
}

func TestToolCallSessionSnapshotIncludesAggregates(t *testing.T) {
	now := time.Now().UTC()
	providerName := "openai"
	modelName := "gpt-4o-mini"
	toolCallID := "call_123"
	toolName := "get_weather"
	toolOutput := `{"temp":18}`

	s := &ToolCallSession{
		OriginalMessages: []provider.Message{{Role: "user", Content: []byte(`"hello"`)}},
		RequestToolsJSON: `[{"type":"function"}]`,
		CreatedAt:        now,
		LastTouchAt:      now,
		Timeline: []ExecutionTimelineStep{
			{
				Step:        1,
				Phase:       "inference",
				StartedAt:   now.Format(time.RFC3339Nano),
				CompletedAt: now.Add(100 * time.Millisecond).Format(time.RFC3339Nano),
				DurationMs:  100,
				Status:      "completed",
				Provider:    &providerName,
				Model:       &modelName,
			},
			{
				Step:        2,
				Phase:       "tool_execution",
				StartedAt:   now.Add(100 * time.Millisecond).Format(time.RFC3339Nano),
				CompletedAt: now.Add(300 * time.Millisecond).Format(time.RFC3339Nano),
				DurationMs:  200,
				Status:      "completed",
				ToolCallID:  &toolCallID,
				ToolName:    &toolName,
				ToolOutput:  &toolOutput,
			},
		},
		InferenceTurns: 2,
		AllTokensIn:    100,
		AllTokensOut:   50,
		AllInputCost:   0.001,
		AllOutputCost:  0.002,
		AllTotalCost:   0.003,
	}

	snap := s.Snapshot()

	if !snap.HasToolCalls {
		t.Fatal("snap.HasToolCalls = false, want true")
	}
	if snap.ToolCallCount != 1 {
		t.Fatalf("snap.ToolCallCount = %d, want 1", snap.ToolCallCount)
	}
	if snap.InferenceTurns != 2 {
		t.Fatalf("snap.InferenceTurns = %d, want 2", snap.InferenceTurns)
	}
	if snap.AllTokensIn != 100 || snap.AllTokensOut != 50 {
		t.Fatalf("unexpected token aggregates: in=%d out=%d", snap.AllTokensIn, snap.AllTokensOut)
	}
	if len(snap.Timeline) != 2 {
		t.Fatalf("len(snap.Timeline) = %d, want 2", len(snap.Timeline))
	}
}

func TestSessionStoreConcurrentSetGetDelete(t *testing.T) {
	store := NewSessionStore()
	defer store.Close()

	const workers = 16
	const perWorker = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		w := w
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				id := fmt.Sprintf("session-%d-%d", w, i)
				s := &ToolCallSession{CreatedAt: time.Now(), LastTouchAt: time.Now()}
				store.Set(id, s)
				if got := store.Get(id); got == nil {
					// t.Fatalf is illegal from a non-test goroutine; use t.Errorf
					// + return to fail the test without goroutine-unsafe FailNow.
					t.Errorf("store.Get(%q) = nil, want session", id)
					return
				}
				store.Delete(id)
			}
		}()
	}

	wg.Wait()

	store.mu.RLock()
	defer store.mu.RUnlock()
	if len(store.sessions) != 0 {
		t.Fatalf("len(store.sessions) = %d, want 0", len(store.sessions))
	}
}
