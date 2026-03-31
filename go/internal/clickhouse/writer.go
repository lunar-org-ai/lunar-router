package clickhouse

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	ch "github.com/ClickHouse/clickhouse-go/v2"
	chdriver "github.com/ClickHouse/clickhouse-go/v2/lib/driver"

	"github.com/lunar-org-ai/lunar-router/go/internal/metrics"
)

// clampInt clamps v to [min, max].
func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// TraceExtra carries routing-specific fields not in RequestMetrics.
type TraceExtra struct {
	RequestType          string             // "route", "chat", "chat_stream"
	CacheHit             bool
	ExpectedError        float64
	CostAdjustedScore    float64
	AllScores            map[string]float64
	ClusterProbabilities []float64
	InputText            string             // last user message or full prompt
	OutputText           string             // assistant response text
	InputMessages        string             // JSON of full messages array
	OutputMessage        string             // JSON of assistant message

	// Tool-call tracking (migration 006)
	FinishReason      string // "stop", "tool_calls", "length", etc.
	RequestTools      string // JSON array of tools sent in the request
	ResponseToolCalls string // JSON array of tool_calls from assistant response
	HasToolCalls      bool
	ToolCallsCount    int
	ExecutionTimeline string  // JSON array of ExecutionTimelineStep
	TokensPerS        float64 // total_tokens / latency_s
}

// traceRow is the combined data for one ClickHouse row.
type traceRow struct {
	Metrics metrics.RequestMetrics
	Extra   TraceExtra
}

// Writer is an async, non-blocking ClickHouse trace writer.
// It buffers rows in a channel and flushes them in batches from a background goroutine.
// Record() never blocks the caller — if the buffer is full, the row is dropped.
type Writer struct {
	conn chdriver.Conn
	cfg  Config
	ch   chan traceRow
	done chan struct{}
	wg   sync.WaitGroup

	// Observability counters (atomic for lock-free reads)
	written     atomic.Int64
	dropped     atomic.Int64
	flushErrors atomic.Int64
}

// openConn creates a new ClickHouse native-protocol connection.
func openConn(cfg Config) (chdriver.Conn, error) {
	conn, err := ch.Open(&ch.Options{
		Addr: []string{fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)},
		Auth: ch.Auth{
			Database: cfg.Database,
			Username: cfg.Username,
			Password: cfg.Password,
		},
		Settings: ch.Settings{
			"async_insert":          1,
			"wait_for_async_insert": 1,
		},
		MaxOpenConns: 4,
		DialTimeout:  5 * time.Second,
	})
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := conn.Ping(ctx); err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}

// NewWriter creates and starts a new Writer.
// Returns nil if the config is disabled.
func NewWriter(cfg Config) (*Writer, error) {
	if !cfg.Enabled {
		return nil, nil
	}

	conn, err := openConn(cfg)
	if err != nil {
		return nil, err
	}

	bufSize := cfg.BatchSize * 2
	if bufSize < 100 {
		bufSize = 100
	}

	w := &Writer{
		conn: conn,
		cfg:  cfg,
		ch:   make(chan traceRow, bufSize),
		done: make(chan struct{}),
	}

	w.wg.Add(1)
	go w.run()

	log.Printf("[clickhouse] Writer started (batch=%d, flush=%s, buffer=%d)",
		cfg.BatchSize, cfg.FlushInterval, bufSize)

	return w, nil
}

// Record enqueues a trace row for async insertion.
// Non-blocking: drops the row (and increments Dropped counter) if the buffer is full.
func (w *Writer) Record(m metrics.RequestMetrics, extra TraceExtra) {
	if w == nil {
		return
	}
	select {
	case w.ch <- traceRow{Metrics: m, Extra: extra}:
	default:
		w.dropped.Add(1)
	}
}

// Stats returns writer observability counters.
func (w *Writer) Stats() (written, dropped, flushErrors int64) {
	if w == nil {
		return 0, 0, 0
	}
	return w.written.Load(), w.dropped.Load(), w.flushErrors.Load()
}

// Close drains remaining rows and shuts down the writer.
func (w *Writer) Close() {
	if w == nil {
		return
	}
	close(w.ch)

	// Wait for drain with timeout
	done := make(chan struct{})
	go func() {
		w.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		log.Println("[clickhouse] Writer drain timed out after 10s")
	}

	w.conn.Close()
	log.Printf("[clickhouse] Writer closed (written=%d, dropped=%d, errors=%d)",
		w.written.Load(), w.dropped.Load(), w.flushErrors.Load())
}

// run is the background goroutine that batches and flushes rows.
func (w *Writer) run() {
	defer w.wg.Done()

	batch := make([]traceRow, 0, w.cfg.BatchSize)
	ticker := time.NewTicker(w.cfg.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case row, ok := <-w.ch:
			if !ok {
				// Channel closed — flush remaining and exit
				if len(batch) > 0 {
					w.flush(batch)
				}
				return
			}
			batch = append(batch, row)
			if len(batch) >= w.cfg.BatchSize {
				w.flush(batch)
				batch = make([]traceRow, 0, w.cfg.BatchSize)
			}

		case <-ticker.C:
			if len(batch) > 0 {
				w.flush(batch)
				batch = make([]traceRow, 0, w.cfg.BatchSize)
			}
		}
	}
}

// flush sends a batch of rows to ClickHouse with retries.
func (w *Writer) flush(rows []traceRow) {
	var lastErr error

	for attempt := 0; attempt <= w.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(100<<uint(attempt-1)) * time.Millisecond
			time.Sleep(backoff)
		}

		err := w.insertBatch(rows)
		if err == nil {
			w.written.Add(int64(len(rows)))
			return
		}
		lastErr = err
	}

	// All retries exhausted — drop the batch
	w.flushErrors.Add(1)
	log.Printf("[clickhouse] Flush failed after %d retries (%d rows dropped): %v",
		w.cfg.MaxRetries, len(rows), lastErr)
	w.dropped.Add(int64(len(rows)))
}

// InsertClusterMappings inserts trace → cluster mappings into trace_cluster_map.
func (w *Writer) InsertClusterMappings(runID string, clusterID int, requestIDs []string) error {
	if w == nil || len(requestIDs) == 0 {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	batch, err := w.conn.PrepareBatch(ctx, `INSERT INTO trace_cluster_map (run_id, request_id, cluster_id)`)
	if err != nil {
		return err
	}

	for _, reqID := range requestIDs {
		if err := batch.Append(runID, reqID, uint32(clusterID)); err != nil {
			return err
		}
	}

	return batch.Send()
}

// insertBatch performs a single batch insert into llm_traces.
func (w *Writer) insertBatch(rows []traceRow) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	batch, err := w.conn.PrepareBatch(ctx, `INSERT INTO llm_traces (
		request_id, timestamp, selected_model, provider, cluster_id,
		expected_error, cost_adjusted_score,
		latency_ms, ttft_ms, routing_ms, embedding_ms,
		tokens_in, tokens_out, total_tokens,
		input_cost_usd, output_cost_usd, cache_input_cost_usd, total_cost_usd,
		is_error, error_category, error_message,
		request_type, is_stream, cache_hit,
		fallback_count, provider_attempts,
		all_scores, cluster_probabilities,
		input_text, output_text, input_messages, output_message,
		finish_reason, request_tools, response_tool_calls,
		has_tool_calls, tool_calls_count, execution_timeline, tokens_per_s
	)`)
	if err != nil {
		return err
	}

	for _, row := range rows {
		m := row.Metrics
		e := row.Extra

		// Resolve timestamp
		ts := time.Now().UTC()
		if m.Timestamp > 0 {
			ts = time.UnixMilli(m.Timestamp).UTC()
		}

		// Resolve request ID
		requestID := m.RequestID
		if requestID == "" {
			requestID = "" // Let ClickHouse generate via DEFAULT
		}

		// Resolve selected model
		selectedModel := m.SelectedModel
		if selectedModel == "" {
			selectedModel = m.Model
		}

		// JSON encode provider attempts
		attemptsJSON := "[]"
		if len(m.ProviderAttempts) > 0 {
			if b, err := json.Marshal(m.ProviderAttempts); err == nil {
				attemptsJSON = string(b)
			}
		}

		// JSON encode all_scores
		scoresJSON := "{}"
		if len(e.AllScores) > 0 {
			if b, err := json.Marshal(e.AllScores); err == nil {
				scoresJSON = string(b)
			}
		}

		// JSON encode cluster probabilities
		probsJSON := "[]"
		if len(e.ClusterProbabilities) > 0 {
			if b, err := json.Marshal(e.ClusterProbabilities); err == nil {
				probsJSON = string(b)
			}
		}

		// is_error as UInt8
		var isError uint8
		if m.Error > 0 {
			isError = 1
		}

		// is_stream as UInt8
		var isStream uint8
		if m.Stream {
			isStream = 1
		}

		// cache_hit as UInt8
		var cacheHit uint8
		if e.CacheHit {
			cacheHit = 1
		}

		// Request type
		reqType := e.RequestType
		if reqType == "" {
			reqType = "chat"
		}

		// Derived tool-call fields
		var hasToolCalls uint8
		if e.HasToolCalls {
			hasToolCalls = 1
		}
		requestTools := e.RequestTools
		if requestTools == "" {
			requestTools = "[]"
		}
		responseToolCalls := e.ResponseToolCalls
		if responseToolCalls == "" {
			responseToolCalls = "[]"
		}
		executionTimeline := e.ExecutionTimeline
		if executionTimeline == "" {
			executionTimeline = "[]"
		}

		err := batch.Append(
			requestID,
			ts,
			selectedModel,
			m.Provider,
			int32(m.ClusterID),
			e.ExpectedError,
			e.CostAdjustedScore,
			m.LatencyMs,
			m.TTFTMs,
			m.RoutingMs,
			m.EmbeddingMs,
			uint32(m.TokensIn),
			uint32(m.TokensOut),
			uint32(m.TotalTokens),
			m.InputCostUSD,
			m.OutputCostUSD,
			m.CacheInputCostUSD,
			m.TotalCostUSD,
			isError,
			m.ErrorCategory,
			m.ErrorMessage,
			reqType,
			isStream,
			cacheHit,
			uint8(m.FallbackCount),
			attemptsJSON,
			scoresJSON,
			probsJSON,
			e.InputText,
			e.OutputText,
			e.InputMessages,
			e.OutputMessage,
			e.FinishReason,
			requestTools,
			responseToolCalls,
			hasToolCalls,
			uint16(clampInt(e.ToolCallsCount, 0, 65535)),
			executionTimeline,
			e.TokensPerS,
		)
		if err != nil {
			return err
		}
	}

	return batch.Send()
}
