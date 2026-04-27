package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// AzureProvider forwards requests to Azure OpenAI.
//
// Azure differs from the other OpenAI-compatible providers in three ways
// that justify a dedicated type rather than reusing OpenAIProvider:
//
//  1. The endpoint is per-resource (lives in AZURE_OPENAI_ENDPOINT, not in
//     ProviderConfig.BaseURL — there is no single fixed URL).
//  2. The model field in the request *path* is the deployment name (whatever
//     the user named the deployment when creating it), and the api-version
//     is a query parameter:
//       {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
//  3. Auth uses an "api-key" header, not "Authorization: Bearer".
//
// Body shape and request fields are identical to OpenAI, so we reuse
// OpenAIProvider.buildBody to avoid duplication.
type AzureProvider struct {
	name       string
	endpoint   string // from AZURE_OPENAI_ENDPOINT (or ProviderConfig.BaseURL if set explicitly)
	apiKey     string
	apiVersion string
	openai     *OpenAIProvider // body-building helper only
	client     *http.Client
}

// NewAzureProvider creates a provider for Azure OpenAI.
//
// Env vars consulted at construction time:
//   - AZURE_OPENAI_API_KEY      → required (or set later via SetAPIKey)
//   - AZURE_OPENAI_ENDPOINT     → required (per-resource endpoint URL)
//   - AZURE_OPENAI_API_VERSION  → optional, defaults to "2024-10-21"
func NewAzureProvider(cfg ProviderConfig) *AzureProvider {
	endpoint := cfg.BaseURL
	if endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	endpoint = strings.TrimRight(endpoint, "/")

	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		apiVersion = "2024-10-21"
	}

	apiKeyEnv := cfg.APIKeyEnv
	if apiKeyEnv == "" {
		apiKeyEnv = "AZURE_OPENAI_API_KEY"
	}

	return &AzureProvider{
		name:       cfg.Name,
		endpoint:   endpoint,
		apiKey:     os.Getenv(apiKeyEnv),
		apiVersion: apiVersion,
		openai:     &OpenAIProvider{name: cfg.Name},
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

func (p *AzureProvider) Name() string        { return p.name }
func (p *AzureProvider) SetAPIKey(key string) { p.apiKey = key }
func (p *AzureProvider) HasAPIKey() bool      { return p.apiKey != "" }

// chatURL builds the per-deployment chat-completions URL with api-version.
// The deployment name comes from req.Model — by the time the request reaches
// us, the registry has stripped the "azure/" prefix.
func (p *AzureProvider) chatURL(deployment string) (string, error) {
	if p.endpoint == "" {
		return "", fmt.Errorf("azure: endpoint not configured (AZURE_OPENAI_ENDPOINT)")
	}
	if deployment == "" {
		return "", fmt.Errorf("azure: model (deployment name) is required")
	}
	u := fmt.Sprintf(
		"%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.endpoint,
		url.PathEscape(deployment),
		url.QueryEscape(p.apiVersion),
	)
	return u, nil
}

func (p *AzureProvider) Send(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	body, err := p.openai.buildBody(req)
	if err != nil {
		return nil, err
	}

	endpoint, err := p.chatURL(req.Model)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("provider %s error (status %d): %s", p.name, resp.StatusCode, respBody)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	return &chatResp, nil
}

func (p *AzureProvider) SendStream(ctx context.Context, req *ChatRequest) (io.ReadCloser, error) {
	req.Stream = true

	body, err := p.openai.buildBody(req)
	if err != nil {
		return nil, err
	}

	endpoint, err := p.chatURL(req.Model)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("provider %s error (status %d): %s", p.name, resp.StatusCode, body)
	}

	return resp.Body, nil
}

func (p *AzureProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("api-key", p.apiKey)
	}
}
