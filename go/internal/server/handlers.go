package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	chw "github.com/lunar-org-ai/lunar-router/go/internal/clickhouse"
	"github.com/lunar-org-ai/lunar-router/go/internal/metrics"
	"github.com/lunar-org-ai/lunar-router/go/internal/provider"
	"github.com/lunar-org-ai/lunar-router/go/internal/router"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleListModels)
	mux.HandleFunc("GET /v1/models/{id}", s.handleGetModel)
	mux.HandleFunc("POST /v1/route", s.handleRoute)
	mux.HandleFunc("GET /v1/metrics", s.handleMetrics)
	mux.HandleFunc("GET /v1/metrics/recent", s.handleMetricsRecent)
	mux.HandleFunc("POST /v1/metrics/reset", s.handleMetricsReset)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /v1/cache", s.handleCacheStats)
	mux.HandleFunc("POST /v1/cache/clear", s.handleCacheClear)
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("POST /stats/reset", s.handleStatsReset)
	mux.HandleFunc("POST /v1/config/keys", s.handleSetKey)
	mux.HandleFunc("GET /v1/config/keys", s.handleListKeys)
	mux.HandleFunc("DELETE /v1/config/keys/{provider}", s.handleDeleteKey)
	mux.HandleFunc("POST /v1/config/reload", s.handleReloadKeys)
	mux.HandleFunc("POST /v1/traces", s.handleIngestTraces)
	mux.HandleFunc("POST /v1/datasets/{runId}/{clusterId}/traces", s.handleAddTracesToDataset)
	mux.HandleFunc("POST /v1/datasets/{runId}/{clusterId}/assign", s.handleAssignTracesToDataset)
}

// --- Health ---

type healthResponse struct {
	Status            string `json:"status"`
	RouterInitialized bool   `json:"router_initialized"`
	NumModels         int    `json:"num_models"`
	NumClusters       int    `json:"num_clusters"`
	EmbedderReady     bool   `json:"embedder_ready"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := healthResponse{
		Status:            "healthy",
		RouterInitialized: s.Router != nil,
		NumModels:         s.Registry.Len(),
		NumClusters:       s.Router.ClusterAssigner.NumClusters(),
		EmbedderReady:     s.Router.Embedder != nil,
	}
	writeJSON(w, http.StatusOK, resp)
}

// --- Route ---

type routeRequest struct {
	Prompt          string    `json:"prompt,omitempty"`
	Embedding       []float64 `json:"embedding,omitempty"`
	AvailableModels []string  `json:"available_models,omitempty"`
	CostWeight      *float64  `json:"cost_weight,omitempty"`
}

type routeResponse struct {
	*router.RoutingDecision
	CacheHit bool       `json:"cache_hit"`
	Usage    *usageInfo `json:"usage,omitempty"`
}

type usageInfo struct {
	RoutingMs   float64 `json:"routing_ms"`
	EmbeddingMs float64 `json:"embedding_ms,omitempty"`
}

type costInfo struct {
	InputCostUSD  float64 `json:"input_cost_usd"`
	OutputCostUSD float64 `json:"output_cost_usd"`
	TotalCostUSD  float64 `json:"total_cost_usd"`
}

func (s *Server) handleRoute(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var req routeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.Embedding) == 0 && req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "either 'prompt' or 'embedding' is required")
		return
	}

	var opts []router.RouteOption
	if req.AvailableModels != nil {
		opts = append(opts, router.WithAvailableModels(req.AvailableModels))
	}
	if req.CostWeight != nil {
		opts = append(opts, router.WithCostWeight(*req.CostWeight))
	}

	var decision *router.RoutingDecision
	var err error
	var embeddingMs float64

	var cacheHit bool

	if len(req.Embedding) > 0 {
		decision, err = s.Router.RouteEmbedding(req.Embedding, opts...)
	} else {
		hitsBefore := s.Router.Cache().Stats().Hits
		embStart := time.Now()
		decision, err = s.Router.Route(req.Prompt, opts...)
		embeddingMs = float64(time.Since(embStart).Microseconds()) / 1000.0
		cacheHit = s.Router.Cache().Stats().Hits > hitsBefore
	}

	routingMs := float64(time.Since(start).Microseconds()) / 1000.0

	if err != nil {
		// Record failed routing request
		m := metrics.RequestMetrics{
			LatencyMs:     routingMs,
			EmbeddingMs:   embeddingMs,
			RoutingMs:     routingMs,
			Error:         1.0,
			ErrorCategory: metrics.ErrCategoryInvalidReq,
			ErrorMessage:  err.Error(),
		}
		s.Metrics.Record(m)
		s.CHWriter.Record(m, chw.TraceExtra{RequestType: "route"})
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Record successful routing request
	tokensIn := 0
	if req.Prompt != "" {
		tokensIn = metrics.EstimateTokensIn(req.Prompt)
	}

	m := metrics.RequestMetrics{
		LatencyMs:     routingMs,
		EmbeddingMs:   embeddingMs,
		RoutingMs:     routingMs,
		TokensIn:      tokensIn,
		Error:         0,
		SelectedModel: decision.SelectedModel,
		ClusterID:     decision.ClusterID,
	}
	s.Metrics.Record(m)
	s.CHWriter.Record(m, chw.TraceExtra{
		RequestType:       "route",
		CacheHit:          cacheHit,
		ExpectedError:     decision.ExpectedError,
		CostAdjustedScore: decision.CostAdjustedScore,
		AllScores:         decision.AllScores,
	})

	writeJSON(w, http.StatusOK, routeResponse{
		RoutingDecision: decision,
		CacheHit:        cacheHit,
		Usage: &usageInfo{
			RoutingMs:   routingMs,
			EmbeddingMs: embeddingMs,
		},
	})
}

// --- Models ---

type modelInfo struct {
	ModelID         string  `json:"model_id"`
	CostPer1kTokens float64 `json:"cost_per_1k_tokens"`
	NumClusters     int     `json:"num_clusters"`
	OverallAccuracy float64 `json:"overall_accuracy"`
}

type modelListResponse struct {
	Models       []modelInfo `json:"models"`
	DefaultModel string      `json:"default_model"`
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	profiles := s.Registry.GetAll()
	models := make([]modelInfo, len(profiles))
	for i, p := range profiles {
		models[i] = modelInfo{
			ModelID:         p.ModelID,
			CostPer1kTokens: p.CostPer1kTokens,
			NumClusters:     p.NumClusters(),
			OverallAccuracy: p.OverallAccuracy(),
		}
	}
	writeJSON(w, http.StatusOK, modelListResponse{
		Models:       models,
		DefaultModel: s.Registry.DefaultModelID(),
	})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	p := s.Registry.Get(id)
	if p == nil {
		writeError(w, http.StatusNotFound, "model not found: "+id)
		return
	}
	writeJSON(w, http.StatusOK, modelInfo{
		ModelID:         p.ModelID,
		CostPer1kTokens: p.CostPer1kTokens,
		NumClusters:     p.NumClusters(),
		OverallAccuracy: p.OverallAccuracy(),
	})
}

// --- Metrics ---

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.Metrics.Summary())
}

func (s *Server) handleMetricsRecent(w http.ResponseWriter, r *http.Request) {
	n := 20
	if v := r.URL.Query().Get("n"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			n = parsed
		}
	}
	writeJSON(w, http.StatusOK, s.Metrics.RecentRequests(n))
}

func (s *Server) handleMetricsReset(w http.ResponseWriter, r *http.Request) {
	s.Metrics.Reset()
	writeJSON(w, http.StatusOK, map[string]string{"message": "metrics reset"})
}

// --- Cache ---

func (s *Server) handleCacheStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.Router.Cache().Stats())
}

func (s *Server) handleCacheClear(w http.ResponseWriter, r *http.Request) {
	s.Router.Cache().Clear()
	writeJSON(w, http.StatusOK, map[string]string{"message": "cache cleared"})
}

// --- Stats (legacy, kept for backwards compat) ---

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	snap := s.Router.Stats().Snapshot()
	clusterDist := make(map[string]int, len(snap.ClusterDistributions))
	for k, v := range snap.ClusterDistributions {
		clusterDist[strconv.Itoa(k)] = v
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"total_requests":        snap.TotalRequests,
		"model_selections":      snap.ModelSelections,
		"cluster_distributions": clusterDist,
		"avg_expected_error":    snap.AvgExpectedError,
		"avg_cost_score":        snap.AvgCostScore,
	})
}

func (s *Server) handleStatsReset(w http.ResponseWriter, r *http.Request) {
	s.Router.Stats().Reset()
	s.Router.Cache().Clear()
	s.Metrics.Reset()
	writeJSON(w, http.StatusOK, map[string]string{"message": "statistics reset"})
}

// --- Runtime Key Management ---

type setKeyRequest struct {
	Provider string `json:"provider"`
	APIKey   string `json:"api_key"`
}

func (s *Server) handleSetKey(w http.ResponseWriter, r *http.Request) {
	var req setKeyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid body: "+err.Error())
		return
	}
	if req.Provider == "" || req.APIKey == "" {
		writeError(w, http.StatusBadRequest, "provider and api_key are required")
		return
	}
	if err := s.Providers.SetProviderKey(req.Provider, req.APIKey); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"message": "key updated", "provider": req.Provider})
}

func (s *Server) handleListKeys(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"configured_providers": s.Providers.ConfiguredProviders(),
	})
}

func (s *Server) handleDeleteKey(w http.ResponseWriter, r *http.Request) {
	provider := r.PathValue("provider")
	if err := s.Providers.SetProviderKey(provider, ""); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"message": "key removed", "provider": provider})
}

func (s *Server) handleReloadKeys(w http.ResponseWriter, r *http.Request) {
	loaded := s.ReloadSecretsFile()
	writeJSON(w, http.StatusOK, map[string]any{
		"message":              "keys reloaded",
		"loaded":               loaded,
		"configured_providers": s.Providers.ConfiguredProviders(),
	})
}

// --- Trace Ingestion ---

// ingestTrace is the flexible input format for manual trace import.
type ingestTrace struct {
	// Messages (OpenAI format) — the primary content
	Messages []map[string]any `json:"messages,omitempty"`

	// Simple input/output (alternative to messages)
	Input  string `json:"input,omitempty"`
	Output string `json:"output,omitempty"`

	// Metadata (all optional, auto-enriched if missing)
	Model     string  `json:"model,omitempty"`
	Provider  string  `json:"provider,omitempty"`
	TokensIn  *int    `json:"tokens_in,omitempty"`
	TokensOut *int    `json:"tokens_out,omitempty"`
	CostUSD   *float64 `json:"cost_usd,omitempty"`
	LatencyMs *float64 `json:"latency_ms,omitempty"`
	IsError   bool    `json:"is_error,omitempty"`
	Source    string  `json:"source,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	Timestamp *string  `json:"timestamp,omitempty"` // ISO 8601
}

type ingestRequest struct {
	// Single trace
	*ingestTrace

	// Batch of traces
	Traces []ingestTrace `json:"traces,omitempty"`
}

func (s *Server) handleIngestTraces(w http.ResponseWriter, r *http.Request) {
	if s.CHWriter == nil {
		writeError(w, http.StatusServiceUnavailable, "ClickHouse not enabled. Start with: make start-full")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}

	// Try to detect JSONL (newline-delimited JSON)
	var traces []ingestTrace
	if len(body) > 0 && body[0] == '{' && bytes.Contains(body, []byte("\n{")) {
		// JSONL format
		for _, line := range bytes.Split(body, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var t ingestTrace
			if err := json.Unmarshal(line, &t); err == nil {
				traces = append(traces, t)
			}
		}
	} else {
		// JSON format — single object or {traces: [...]}
		var req ingestRequest
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Traces) > 0 {
			traces = req.Traces
		} else if req.ingestTrace != nil && (len(req.Messages) > 0 || req.Input != "") {
			traces = []ingestTrace{*req.ingestTrace}
		}
	}

	if len(traces) == 0 {
		writeError(w, http.StatusBadRequest, "no traces found. Send {messages:[...]} or {traces:[...]} or JSONL")
		return
	}

	ingested := 0
	for _, t := range traces {
		inputText, outputText, messagesJSON, outputMsgJSON := extractTraceContent(&t)

		tokensIn := 0
		if t.TokensIn != nil {
			tokensIn = *t.TokensIn
		} else {
			tokensIn = estimateTokens(inputText)
		}

		tokensOut := 0
		if t.TokensOut != nil {
			tokensOut = *t.TokensOut
		} else {
			tokensOut = estimateTokens(outputText)
		}

		var totalCost float64
		if t.CostUSD != nil {
			totalCost = *t.CostUSD
		} else if t.Model != "" {
			pricing := provider.GetPricing(t.Model)
			if pricing != nil {
				_, _, totalCost = pricing.ComputeCost(tokensIn, tokensOut)
			}
		}

		latency := 0.0
		if t.LatencyMs != nil {
			latency = *t.LatencyMs
		}

		source := t.Source
		if source == "" {
			source = "import"
		}

		reqType := source
		if len(t.Tags) > 0 {
			if b, err := json.Marshal(t.Tags); err == nil {
				reqType = source + ":" + string(b)
			}
		}

		var errVal float64
		if t.IsError {
			errVal = 1.0
		}

		m := metrics.RequestMetrics{
			Provider:    t.Provider,
			Model:       t.Model,
			TokensIn:    tokensIn,
			TokensOut:   tokensOut,
			TotalTokens: tokensIn + tokensOut,
			TotalCostUSD: totalCost,
			LatencyMs:   latency,
			Error:       errVal,
			ClusterID:   -1,
		}

		extra := chw.TraceExtra{
			RequestType:   reqType,
			InputText:     inputText,
			OutputText:    outputText,
			InputMessages: messagesJSON,
			OutputMessage: outputMsgJSON,
		}

		s.CHWriter.Record(m, extra)
		ingested++
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"message":  fmt.Sprintf("ingested %d traces", ingested),
		"ingested": ingested,
	})
}

// extractTraceContent normalizes messages or input/output into trace content fields.
func extractTraceContent(t *ingestTrace) (inputText, outputText, messagesJSON, outputMsgJSON string) {
	if len(t.Messages) > 0 {
		// Extract last user message as input_text
		for i := len(t.Messages) - 1; i >= 0; i-- {
			if role, _ := t.Messages[i]["role"].(string); role == "user" {
				if content, ok := t.Messages[i]["content"].(string); ok {
					inputText = content
				}
				break
			}
		}
		// Extract last assistant message as output_text
		for i := len(t.Messages) - 1; i >= 0; i-- {
			if role, _ := t.Messages[i]["role"].(string); role == "assistant" {
				if content, ok := t.Messages[i]["content"].(string); ok {
					outputText = content
				}
				outMsg := t.Messages[i]
				if b, err := json.Marshal(outMsg); err == nil {
					outputMsgJSON = string(b)
				}
				break
			}
		}
		if b, err := json.Marshal(t.Messages); err == nil {
			messagesJSON = string(b)
		}
	} else {
		inputText = t.Input
		outputText = t.Output
		if t.Input != "" {
			msgs := []map[string]string{{"role": "user", "content": t.Input}}
			if t.Output != "" {
				msgs = append(msgs, map[string]string{"role": "assistant", "content": t.Output})
			}
			if b, err := json.Marshal(msgs); err == nil {
				messagesJSON = string(b)
			}
		}
		if t.Output != "" {
			outMsg := map[string]string{"role": "assistant", "content": t.Output}
			if b, err := json.Marshal(outMsg); err == nil {
				outputMsgJSON = string(b)
			}
		}
	}
	return
}

func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	// Rough estimate: ~4 chars per token
	return len(text) / 4
}

// --- Add Traces to Dataset ---

func (s *Server) handleAddTracesToDataset(w http.ResponseWriter, r *http.Request) {
	if s.CHWriter == nil {
		writeError(w, http.StatusServiceUnavailable, "ClickHouse not enabled")
		return
	}

	runId := r.PathValue("runId")
	clusterIdStr := r.PathValue("clusterId")
	clusterId, err := strconv.Atoi(clusterIdStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid cluster_id")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}

	// Parse traces (same format as /v1/traces)
	var traces []ingestTrace
	if len(body) > 0 && body[0] == '{' && bytes.Contains(body, []byte("\n{")) {
		for _, line := range bytes.Split(body, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var t ingestTrace
			if json.Unmarshal(line, &t) == nil {
				traces = append(traces, t)
			}
		}
	} else {
		var req ingestRequest
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Traces) > 0 {
			traces = req.Traces
		} else if req.ingestTrace != nil && (len(req.Messages) > 0 || req.Input != "") {
			traces = []ingestTrace{*req.ingestTrace}
		}
	}

	if len(traces) == 0 {
		writeError(w, http.StatusBadRequest, "no traces provided")
		return
	}

	// Insert each trace and map to the cluster
	ingested := 0
	var requestIDs []string

	for _, t := range traces {
		inputText, outputText, messagesJSON, outputMsgJSON := extractTraceContent(&t)

		// Generate a deterministic request_id from content
		reqID := fmt.Sprintf("manual-%x", sha256Short(inputText+outputText))

		tokensIn := 0
		if t.TokensIn != nil {
			tokensIn = *t.TokensIn
		} else {
			tokensIn = estimateTokens(inputText)
		}
		tokensOut := 0
		if t.TokensOut != nil {
			tokensOut = *t.TokensOut
		} else {
			tokensOut = estimateTokens(outputText)
		}

		var totalCost float64
		if t.CostUSD != nil {
			totalCost = *t.CostUSD
		} else if t.Model != "" {
			if pricing := provider.GetPricing(t.Model); pricing != nil {
				_, _, totalCost = pricing.ComputeCost(tokensIn, tokensOut)
			}
		}

		latency := 0.0
		if t.LatencyMs != nil {
			latency = *t.LatencyMs
		}

		source := t.Source
		if source == "" {
			source = "manual"
		}

		var errVal float64
		if t.IsError {
			errVal = 1.0
		}

		m := metrics.RequestMetrics{
			RequestID:    reqID,
			Provider:     t.Provider,
			Model:        t.Model,
			TokensIn:     tokensIn,
			TokensOut:    tokensOut,
			TotalTokens:  tokensIn + tokensOut,
			TotalCostUSD: totalCost,
			LatencyMs:    latency,
			Error:        errVal,
			ClusterID:    clusterId,
		}

		extra := chw.TraceExtra{
			RequestType:   source,
			InputText:     inputText,
			OutputText:    outputText,
			InputMessages: messagesJSON,
			OutputMessage: outputMsgJSON,
		}

		s.CHWriter.Record(m, extra)
		requestIDs = append(requestIDs, reqID)
		ingested++
	}

	// Insert mappings into trace_cluster_map
	if err := s.CHWriter.InsertClusterMappings(runId, clusterId, requestIDs); err != nil {
		// Traces are inserted, but mapping failed — log but don't fail
		writeJSON(w, http.StatusOK, map[string]any{
			"message":      fmt.Sprintf("ingested %d traces (mapping error: %v)", ingested, err),
			"ingested":     ingested,
			"run_id":       runId,
			"cluster_id":   clusterId,
			"mapping_error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"message":    fmt.Sprintf("added %d traces to dataset %s/%d", ingested, runId, clusterId),
		"ingested":   ingested,
		"run_id":     runId,
		"cluster_id": clusterId,
	})
}

// --- Assign Existing Traces to Dataset ---

func (s *Server) handleAssignTracesToDataset(w http.ResponseWriter, r *http.Request) {
	if s.CHWriter == nil {
		writeError(w, http.StatusServiceUnavailable, "ClickHouse not enabled")
		return
	}

	runId := r.PathValue("runId")
	clusterIdStr := r.PathValue("clusterId")
	clusterId, err := strconv.Atoi(clusterIdStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid cluster_id")
		return
	}

	var req struct {
		RequestIDs []string `json:"request_ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid body: "+err.Error())
		return
	}

	if len(req.RequestIDs) == 0 {
		writeError(w, http.StatusBadRequest, "request_ids is required")
		return
	}

	if err := s.CHWriter.InsertClusterMappings(runId, clusterId, req.RequestIDs); err != nil {
		writeError(w, http.StatusInternalServerError, "mapping failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"message":    fmt.Sprintf("assigned %d traces to dataset %s/%d", len(req.RequestIDs), runId, clusterId),
		"assigned":   len(req.RequestIDs),
		"run_id":     runId,
		"cluster_id": clusterId,
	})
}

func sha256Short(s string) uint64 {
	h := uint64(14695981039346656037) // FNV offset basis
	for _, c := range s {
		h ^= uint64(c)
		h *= 1099511628211 // FNV prime
	}
	return h
}

// --- Chat Completions (Gateway / Proxy) ---

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Internal calls (clustering pipeline, labeling) are not traced
	isInternal := r.Header.Get("X-Lunar-Internal") == "true"

	// Read raw body (for pass-through)
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}

	var req provider.ChatRequest
	if err := json.Unmarshal(rawBody, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	req.RawBody = rawBody

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}

	selectedModel := req.Model
	var routingMs float64

	// model="auto" → use semantic router to pick best model
	if req.Model == "auto" || req.Model == "" {
		if s.Router == nil || s.Router.Embedder == nil {
			writeError(w, http.StatusBadRequest, "semantic routing not available; specify a model name")
			return
		}

		// Use last user message for routing
		prompt := lastUserMessage(req.Messages)
		routeStart := time.Now()
		decision, err := s.Router.Route(prompt)
		routingMs = float64(time.Since(routeStart).Microseconds()) / 1000.0
		if err != nil {
			writeError(w, http.StatusInternalServerError, "routing failed: "+err.Error())
			return
		}
		selectedModel = decision.SelectedModel
		req.Model = selectedModel

		// Set routing headers
		w.Header().Set("X-Lunar-Selected-Model", selectedModel)
		w.Header().Set("X-Lunar-Cluster-ID", strconv.Itoa(decision.ClusterID))
		w.Header().Set("X-Lunar-Expected-Error", fmt.Sprintf("%.4f", decision.ExpectedError))
		w.Header().Set("X-Lunar-Routing-Ms", fmt.Sprintf("%.2f", routingMs))
	}

	// Find provider for the model
	if s.Providers == nil {
		writeError(w, http.StatusServiceUnavailable, "no providers configured")
		return
	}

	prov, err := s.Providers.ForModel(selectedModel)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Strip "provider/" prefix so upstream API gets the bare model name
	if idx := strings.IndexByte(req.Model, '/'); idx > 0 {
		req.Model = req.Model[idx+1:]
		req.RawBody = nil // force re-serialization with the stripped model name
	}

	// Forward request
	if req.Stream {
		s.handleStreamingProxy(w, r, prov, &req, start, routingMs, isInternal)
	} else {
		s.handleNonStreamingProxy(w, prov, &req, start, routingMs, isInternal)
	}
}

func (s *Server) handleNonStreamingProxy(
	w http.ResponseWriter,
	prov provider.Provider,
	req *provider.ChatRequest,
	start time.Time,
	routingMs float64,
	isInternal bool,
) {
	ctx := context.Background()
	resp, err := prov.Send(ctx, req)
	latencyMs := float64(time.Since(start).Microseconds()) / 1000.0

	if err != nil {
		em := metrics.RequestMetrics{
			LatencyMs:     latencyMs,
			RoutingMs:     routingMs,
			Error:         1.0,
			ErrorCategory: metrics.ErrCategoryServer,
			ErrorMessage:  err.Error(),
			Provider:      prov.Name(),
			Model:         req.Model,
		}
		s.Metrics.Record(em)
		if !isInternal {
			s.CHWriter.Record(em, chw.TraceExtra{RequestType: "chat"})
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	// Record metrics with cost
	tokensIn, tokensOut, totalTokens := 0, 0, 0
	if resp.Usage != nil {
		tokensIn = resp.Usage.PromptTokens
		tokensOut = resp.Usage.CompletionTokens
		totalTokens = resp.Usage.TotalTokens
	}

	var inputCost, outputCost, totalCost float64
	pricing := provider.GetPricing(req.Model)
	if pricing != nil {
		inputCost, outputCost, totalCost = pricing.ComputeCost(tokensIn, tokensOut)
	}

	sm := metrics.RequestMetrics{
		LatencyMs:     latencyMs,
		TTFTMs:        latencyMs,
		RoutingMs:     routingMs,
		TokensIn:      tokensIn,
		TokensOut:     tokensOut,
		TotalTokens:   totalTokens,
		InputCostUSD:  inputCost,
		OutputCostUSD: outputCost,
		TotalCostUSD:  totalCost,
		Error:         0,
		Provider:      prov.Name(),
		Model:         req.Model,
	}
	s.Metrics.Record(sm)

	// Extract content for trace storage
	inputText := lastUserMessage(req.Messages)
	var outputText, outputMsg string
	if len(resp.Choices) > 0 && resp.Choices[0].Message != nil {
		outputText = resp.Choices[0].Message.TextContent()
		if b, err := json.Marshal(resp.Choices[0].Message); err == nil {
			outputMsg = string(b)
		}
	}
	inputMsgsJSON := ""
	if b, err := json.Marshal(req.Messages); err == nil {
		inputMsgsJSON = string(b)
	}

	if !isInternal {
		s.CHWriter.Record(sm, chw.TraceExtra{
			RequestType:   "chat",
			InputText:     inputText,
			OutputText:    outputText,
			InputMessages: inputMsgsJSON,
			OutputMessage: outputMsg,
		})
	}

	// Inject cost into response
	type costAwareResponse struct {
		*provider.ChatResponse
		Cost *costInfo `json:"cost,omitempty"`
	}
	var costField *costInfo
	if pricing != nil {
		costField = &costInfo{
			InputCostUSD:  inputCost,
			OutputCostUSD: outputCost,
			TotalCostUSD:  totalCost,
		}
	}

	writeJSON(w, http.StatusOK, costAwareResponse{
		ChatResponse: resp,
		Cost:         costField,
	})
}

func (s *Server) handleStreamingProxy(
	w http.ResponseWriter,
	r *http.Request,
	prov provider.Provider,
	req *provider.ChatRequest,
	start time.Time,
	routingMs float64,
	isInternal bool,
) {
	ctx := r.Context()
	stream, err := prov.SendStream(ctx, req)
	if err != nil {
		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		sem := metrics.RequestMetrics{
			LatencyMs:     latencyMs,
			RoutingMs:     routingMs,
			Error:         1.0,
			ErrorCategory: metrics.ErrCategoryServer,
			ErrorMessage:  err.Error(),
			Provider:      prov.Name(),
			Model:         req.Model,
			Stream:        true,
		}
		s.Metrics.Record(sem)
		if !isInternal {
			s.CHWriter.Record(sem, chw.TraceExtra{RequestType: "chat_stream"})
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer stream.Close()

	// Stream SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ttftRecorded := false
	var ttftMs float64
	buf := make([]byte, 4096)

	for {
		n, err := stream.Read(buf)
		if n > 0 {
			if !ttftRecorded {
				ttftMs = float64(time.Since(start).Microseconds()) / 1000.0
				ttftRecorded = true
			}
			w.Write(buf[:n])
			flusher.Flush()
		}
		if err != nil {
			break
		}
	}

	latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
	ssm := metrics.RequestMetrics{
		LatencyMs: latencyMs,
		TTFTMs:    ttftMs,
		RoutingMs: routingMs,
		Error:     0,
		Provider:  prov.Name(),
		Model:     req.Model,
		Stream:    true,
	}
	s.Metrics.Record(ssm)
	if !isInternal {
		s.CHWriter.Record(ssm, chw.TraceExtra{RequestType: "chat_stream"})
	}
}

func lastUserMessage(messages []provider.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].TextContent()
		}
	}
	if len(messages) > 0 {
		return messages[len(messages)-1].TextContent()
	}
	return ""
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
