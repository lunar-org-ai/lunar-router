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


func (s *ToolCallSession) AddInferenceStep(providerName, modelName string, startedAt time.Time) int {
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
	return len(s.Timeline) - 1
}

// CompleteInferenceStep updates the inference step at the given index.
// Caller must hold s.mu.
func (s *ToolCallSession) CompleteInferenceStep(idx int, status string, completedAt time.Time, durationMs float64, tokensIn, tokensOut *int, ttftMs *float64, toolError *string) {
	if idx < 0 || idx >= len(s.Timeline) {
		return
	}
	step := &s.Timeline[idx]
	step.Status = status
	step.CompletedAt = completedAt.UTC().Format(time.RFC3339Nano)
	step.DurationMs = durationMs
	step.TokensIn = tokensIn
	step.TokensOut = tokensOut
	step.TTFTMs = ttftMs
	step.ToolError = toolError
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
	s.mu.Lock()
	s.LastTouchAt = time.Now()
	s.mu.Unlock()
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

type SessionSnapshot struct {
	Timeline         []ExecutionTimelineStep
	OriginalMessages []provider.Message
	RequestToolsJSON string
	HasToolCalls     bool
	ToolCallCount    int
	InferenceTurns   int
	AllTokensIn      int
	AllTokensOut     int
	AllInputCost     float64
	AllOutputCost    float64
	AllTotalCost     float64
	CreatedAt        time.Time
}

// Snapshot takes a consistent snapshot of the session under lock.
func (s *ToolCallSession) Snapshot() SessionSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	tl := make([]ExecutionTimelineStep, len(s.Timeline))
	copy(tl, s.Timeline)
	tcCount := 0
	for _, step := range tl {
		if step.Phase == "tool_execution" {
			tcCount++
		}
	}
	return SessionSnapshot{
		Timeline:         tl,
		OriginalMessages: s.OriginalMessages,
		RequestToolsJSON: s.RequestToolsJSON,
		HasToolCalls:     tcCount > 0,
		ToolCallCount:    tcCount,
		InferenceTurns:   s.InferenceTurns,
		AllTokensIn:      s.AllTokensIn,
		AllTokensOut:     s.AllTokensOut,
		AllInputCost:     s.AllInputCost,
		AllOutputCost:    s.AllOutputCost,
		AllTotalCost:     s.AllTotalCost,
		CreatedAt:        s.CreatedAt,
	}
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

func (st *SessionStore) Set(id string, session *ToolCallSession) {
	st.mu.Lock()
	st.sessions[id] = session
	st.mu.Unlock()
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
				s.mu.Lock()
				expired := s.LastTouchAt.Before(cutoff)
				s.mu.Unlock()
				if expired {
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
