package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/lunar-org-ai/lunar-router/go/internal/clickhouse"
	"github.com/lunar-org-ai/lunar-router/go/internal/config"
	"github.com/lunar-org-ai/lunar-router/go/internal/metrics"
	"github.com/lunar-org-ai/lunar-router/go/internal/provider"
	"github.com/lunar-org-ai/lunar-router/go/internal/router"
	"github.com/lunar-org-ai/lunar-router/go/internal/weights"
)

// Server is the Lunar Router HTTP server.
type Server struct {
	Router       *router.Router
	Registry     *weights.Registry
	Providers    *provider.Registry
	Metrics      *metrics.Collector
	CHWriter     *clickhouse.Writer
	Sessions     *SessionStore
	Addr         string

	httpServer *http.Server
}

// New creates a new Server.
func New(r *router.Router, reg *weights.Registry, providers *provider.Registry, cfg *config.Config) *Server {
	s := &Server{
		Router:       r,
		Registry:     reg,
		Providers:    providers,
		Metrics:      metrics.NewCollector(10000),
		Sessions:     NewSessionStore(),
		Addr:         fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port),
	}

	// Initialize ClickHouse writer if enabled
	if cfg.ClickHouse.Enabled {
		w, err := clickhouse.NewWriter(cfg.ClickHouse)
		if err != nil {
			log.Printf("WARNING: ClickHouse writer init failed: %v (traces disabled)", err)
		} else if w != nil {
			// Run schema migrations
			if err := clickhouse.RunMigrationsFromConfig(cfg.ClickHouse); err != nil {
				log.Printf("WARNING: ClickHouse migrations failed: %v", err)
			}
			s.CHWriter = w
			log.Println("ClickHouse trace writer enabled")
		}
	}

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.httpServer = &http.Server{
		Addr:         s.Addr,
		Handler:      corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return s
}

// Run starts the server and blocks until interrupted.
func (s *Server) Run() error {
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Lunar Engine listening on %s", s.Addr)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	select {
	case <-stop:
		log.Println("Shutting down...")
		s.Sessions.Close()
		if s.CHWriter != nil {
			s.CHWriter.Close()
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return s.httpServer.Shutdown(ctx)
	case err := <-errCh:
		return err
	}
}

// ReloadSecretsFile re-reads ~/.lunar/secrets.json and updates provider keys.
// Returns the number of keys loaded.
func (s *Server) ReloadSecretsFile() int {
	secretsPath := os.Getenv("LUNAR_SECRETS_FILE")
	if secretsPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return 0
		}
		secretsPath = filepath.Join(home, ".lunar", "secrets.json")
	}

	data, err := os.ReadFile(secretsPath)
	if err != nil {
		return 0
	}

	var secrets map[string]string
	if err := json.Unmarshal(data, &secrets); err != nil {
		return 0
	}

	loaded := 0
	for providerName, apiKey := range secrets {
		if apiKey == "" {
			continue
		}
		if err := s.Providers.SetProviderKey(providerName, apiKey); err == nil {
			loaded++
		}
	}
	if loaded > 0 {
		log.Printf("Reloaded %d API keys from %s", loaded, secretsPath)
	}
	return loaded
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}
