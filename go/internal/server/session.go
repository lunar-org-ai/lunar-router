package server

import (
	"crypto/rand"
	"fmt"
	"sync"
	"time"

	"github.com/lunar-org-ai/lunar-router/go/internal/provider"
)

type ToolCallSession struct {
	mu sync.Mutex

	ID string

	// Accumulated execution timeline
	Timeline []ExecutionTimelineStep

	// Running token / cost totals across all inference turns
	AllTokensIn  int
	AllTokensOut int
	AllInputCost  float64
	AllOutputCost float64
	AllTotalCost  float64

	// Snapshot of the original request tools list (JSON, for trace)
	RequestToolsJSON string

	// Original user messages (for trace input_text)
	OriginalMessages []provider.Message

	// Timestamp when the last inference step completed — used to compute
	// client-side tool execution duration.
	LastInferenceCompletedAt time.Time

	// Step counter (monotonically increasing across all turns)
	StepCounter int

	// Number of inference turns completed
	InferenceTurns int

	// LastMessageCount is the number of messages in the request at the END of the
	// last inference turn. Used to identify new role="tool" messages on the next
	// request from the client.
	LastMessageCount int

	CreatedAt   time.Time
	LastTouchAt time.Time
}

func (s *ToolCallSession) nextStep() int {
	s.StepCounter++
	return s.StepCounter
}

// Lock locks the session mutex. Callers must call Unlock when done.
func (s *ToolCallSession) Lock() {
	s.mu.Lock()
}

// Unlock unlocks the session mutex.
func (s *ToolCallSession) Unlock() {
	s.mu.Unlock()
}

func (s *ToolCallSession) AddInferenceStep(providerName, modelName string, startedAt time.Time) *ExecutionTimelineStep {
	n := s.nextStep()
	pn := providerName
	mn := modelName
	step := ExecutionTimelineStep{
		Step:      n,
		Phase:     "inference",
		StartedAt: startedAt.UTC().Format(time.RFC3339Nano),
		Status:    "pending",
		Provider:  &pn,
		Model:     &mn,
	}
	s.Timeline = append(s.Timeline, step)
	return &s.Timeline[len(s.Timeline)-1]
}

func (s *ToolCallSession) AddToolResultSteps(
	messages []provider.Message,
	lastKnownLen int,
	requestArrivedAt time.Time,
) {
	startedAt := s.LastInferenceCompletedAt
	if startedAt.IsZero() {
		startedAt = requestArrivedAt
	}

	for i := lastKnownLen; i < len(messages); i++ {
		msg := messages[i]
		if msg.Role != "tool" {
			continue
		}

		n := s.nextStep()
		completedAt := requestArrivedAt
		durationMs := completedAt.Sub(startedAt).Seconds() * 1000
		if durationMs < 0 {
			durationMs = 0
		}

		tcID := msg.ToolCallID
		fnName := msg.Name
		content := msg.TextContent()
		output := content

		step := ExecutionTimelineStep{
			Step:        n,
			Phase:       "tool_execution",
			StartedAt:   startedAt.UTC().Format(time.RFC3339Nano),
			CompletedAt: completedAt.UTC().Format(time.RFC3339Nano),
			DurationMs:  durationMs,
			Status:      "completed",
			ToolCallID:  &tcID,
			ToolName:    &fnName,
			ToolOutput:  &output,
		}
		s.Timeline = append(s.Timeline, step)
	}
}

func (s *ToolCallSession) Touch() {
	s.LastTouchAt = time.Now()
}

func (s *ToolCallSession) ToolCallCount() int {
	n := 0
	for _, step := range s.Timeline {
		if step.Phase == "tool_execution" {
			n++
		}
	}
	return n
}

func (s *ToolCallSession) HasToolCalls() bool {
	return s.ToolCallCount() > 0
}

const sessionTTL = 30 * time.Minute

type SessionStore struct {
	mu       sync.RWMutex
	sessions map[string]*ToolCallSession
	stopGC   chan struct{}
}

func NewSessionStore() *SessionStore {
	st := &SessionStore{
		sessions: make(map[string]*ToolCallSession),
		stopGC:   make(chan struct{}),
	}
	go st.gc()
	return st
}

func (st *SessionStore) New(id string) *ToolCallSession {
	now := time.Now()
	s := &ToolCallSession{
		ID:          id,
		CreatedAt:   now,
		LastTouchAt: now,
	}
	st.mu.Lock()
	st.sessions[id] = s
	st.mu.Unlock()
	return s
}

func (st *SessionStore) Get(id string) *ToolCallSession {
	if id == "" {
		return nil
	}
	st.mu.RLock()
	s := st.sessions[id]
	st.mu.RUnlock()
	return s
}

func (st *SessionStore) Delete(id string) {
	st.mu.Lock()
	delete(st.sessions, id)
	st.mu.Unlock()
}

func (st *SessionStore) Close() {
	select {
	case <-st.stopGC:
		// already closed
	default:
		close(st.stopGC)
	}
}

func (st *SessionStore) gc() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			cutoff := time.Now().Add(-sessionTTL)
			st.mu.Lock()
			for id, s := range st.sessions {
				if s.LastTouchAt.Before(cutoff) {
					delete(st.sessions, id)
				}
			}
			st.mu.Unlock()
		case <-st.stopGC:
			return
		}
	}
}

func GenerateSessionID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("lunar-session-%x", b)
}
