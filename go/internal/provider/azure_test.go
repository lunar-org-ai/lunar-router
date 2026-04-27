package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAzureProviderRegisteredViaFormat(t *testing.T) {
	cfg := ProviderConfig{
		Name:      "azure",
		BaseURL:   "https://res.openai.azure.com",
		APIKeyEnv: "AZURE_OPENAI_API_KEY",
		Format:    "azure",
	}
	t.Setenv("AZURE_OPENAI_API_KEY", "az-key")

	r := NewRegistry([]ProviderConfig{cfg})

	p := r.Get("azure")
	if p == nil {
		t.Fatal("azure provider not registered")
	}
	if _, ok := p.(*AzureProvider); !ok {
		t.Fatalf("expected *AzureProvider, got %T", p)
	}
}

func TestAzureProviderURLBuild(t *testing.T) {
	p := NewAzureProvider(ProviderConfig{
		Name:    "azure",
		BaseURL: "https://res.openai.azure.com",
	})
	t.Setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
	// Re-construct after setting env so apiVersion picks it up.
	p = NewAzureProvider(ProviderConfig{
		Name:    "azure",
		BaseURL: "https://res.openai.azure.com",
	})

	got, err := p.chatURL("my-gpt5")
	if err != nil {
		t.Fatalf("chatURL: %v", err)
	}
	want := "https://res.openai.azure.com/openai/deployments/my-gpt5/chat/completions?api-version=2025-01-01-preview"
	if got != want {
		t.Errorf("URL = %q\n want %q", got, want)
	}
}

func TestAzureProviderDefaultAPIVersion(t *testing.T) {
	t.Setenv("AZURE_OPENAI_API_VERSION", "")
	p := NewAzureProvider(ProviderConfig{
		Name:    "azure",
		BaseURL: "https://res.openai.azure.com",
	})
	if p.apiVersion != "2024-10-21" {
		t.Errorf("apiVersion = %q, want 2024-10-21", p.apiVersion)
	}
}

func TestAzureProviderEndpointFromEnv(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://from-env.openai.azure.com")
	p := NewAzureProvider(ProviderConfig{
		Name:      "azure",
		APIKeyEnv: "AZURE_OPENAI_API_KEY",
	})
	if p.endpoint != "https://from-env.openai.azure.com" {
		t.Errorf("endpoint = %q, want from-env URL", p.endpoint)
	}
}

func TestAzureProviderMissingEndpointFails(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	p := NewAzureProvider(ProviderConfig{
		Name:      "azure",
		APIKeyEnv: "AZURE_OPENAI_API_KEY",
	})
	_, err := p.chatURL("my-deploy")
	if err == nil {
		t.Fatal("expected error when endpoint not configured")
	}
	if !strings.Contains(err.Error(), "AZURE_OPENAI_ENDPOINT") {
		t.Errorf("error should mention AZURE_OPENAI_ENDPOINT, got: %v", err)
	}
}

// TestAzureProviderSendUsesApiKeyHeader spins up a fake Azure server and
// verifies that AzureProvider hits the right path, sends api-key (not
// Authorization Bearer), and round-trips the response.
func TestAzureProviderSendUsesApiKeyHeader(t *testing.T) {
	var gotPath, gotAuthHeader, gotApiKey string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotAuthHeader = r.Header.Get("Authorization")
		gotApiKey = r.Header.Get("api-key")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-az",
			"object": "chat.completion",
			"model": "gpt-5",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
		}`))
	}))
	defer srv.Close()

	t.Setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
	p := NewAzureProvider(ProviderConfig{
		Name:    "azure",
		BaseURL: srv.URL,
	})
	p.SetAPIKey("az-secret")

	resp, err := p.Send(context.Background(), &ChatRequest{
		Model: "my-gpt5",
		Messages: []Message{
			TextMessage("user", "ping"),
		},
	})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}

	wantPath := "/openai/deployments/my-gpt5/chat/completions?api-version=2024-10-21"
	if gotPath != wantPath {
		t.Errorf("path = %q\n want %q", gotPath, wantPath)
	}
	if gotApiKey != "az-secret" {
		t.Errorf("api-key header = %q, want az-secret", gotApiKey)
	}
	if gotAuthHeader != "" {
		t.Errorf("Authorization header should be empty on Azure, got %q", gotAuthHeader)
	}
	if gotBody["model"] != "my-gpt5" {
		t.Errorf("body.model = %v, want my-gpt5", gotBody["model"])
	}
	if len(resp.Choices) != 1 || string(resp.Choices[0].Message.Content) != `"hi"` {
		t.Errorf("response not parsed: %+v", resp)
	}
}
