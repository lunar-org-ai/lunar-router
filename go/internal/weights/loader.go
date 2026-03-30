package weights

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/lunar-org-ai/lunar-router/go/internal/clustering"
)

// LoadedWeights contains all components loaded from a weights directory.
type LoadedWeights struct {
	ClusterAssigner clustering.ClusterAssigner
	Registry        *Registry
}

// LoadWeights loads cluster assigner and model profiles from a weights directory.
//
// Expected structure:
//
//	weights_dir/
//	  clusters/
//	    mmlu_full.npz | default.npz  (cluster centroids/theta)
//	  profiles/
//	    model-name.json              (one per model)
func LoadWeights(weightsDir string) (*LoadedWeights, error) {
	// 1. Load cluster assigner
	assigner, err := loadClusterAssigner(weightsDir)
	if err != nil {
		return nil, fmt.Errorf("load clusters: %w", err)
	}

	// 2. Load profiles
	registry, err := loadProfiles(weightsDir)
	if err != nil {
		return nil, fmt.Errorf("load profiles: %w", err)
	}

	return &LoadedWeights{
		ClusterAssigner: assigner,
		Registry:        registry,
	}, nil
}

func loadClusterAssigner(weightsDir string) (clustering.ClusterAssigner, error) {
	// Try candidate paths in order (matches Python loader.py)
	candidates := []string{
		filepath.Join(weightsDir, "clusters", "mmlu_full.npz"),
		filepath.Join(weightsDir, "clusters", "default.npz"),
		filepath.Join(weightsDir, "clusters.npz"),
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err != nil {
			continue
		}
		return loadClusterFromNpz(path)
	}

	// Try first .npz in clusters/ directory
	clustersDir := filepath.Join(weightsDir, "clusters")
	entries, err := os.ReadDir(clustersDir)
	if err == nil {
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".npz") {
				return loadClusterFromNpz(filepath.Join(clustersDir, e.Name()))
			}
		}
	}

	return nil, fmt.Errorf("no cluster file found in %s (tried: %v)", weightsDir, candidates)
}

func loadClusterFromNpz(path string) (clustering.ClusterAssigner, error) {
	npz, err := ReadNpz(path)
	if err != nil {
		return nil, err
	}

	// Determine type (default to "kmeans")
	assignerType := "kmeans"
	if typeArr := npz.Get("type"); typeArr != nil {
		assignerType = typeArr.AsString()
	}

	centroidsArr := npz.Get("centroids")
	if centroidsArr == nil {
		return nil, fmt.Errorf("npz %s: missing 'centroids' array", path)
	}
	centroids, err := centroidsArr.AsFloat64_2D()
	if err != nil {
		return nil, fmt.Errorf("npz %s: parse centroids: %w", path, err)
	}

	switch assignerType {
	case "kmeans":
		return clustering.NewKMeansAssigner(centroids), nil

	case "learned_map":
		thetaArr := npz.Get("theta")
		if thetaArr == nil {
			return nil, fmt.Errorf("npz %s: learned_map missing 'theta' array", path)
		}
		theta, err := thetaArr.AsFloat64_2D()
		if err != nil {
			return nil, fmt.Errorf("npz %s: parse theta: %w", path, err)
		}

		temperature := 1.0
		if tempArr := npz.Get("temperature"); tempArr != nil {
			tempFlat, err := tempArr.AsFloat64Flat()
			if err == nil && len(tempFlat) > 0 {
				temperature = tempFlat[0]
			}
		}

		return clustering.NewLearnedMapAssigner(centroids, theta, temperature), nil

	default:
		return nil, fmt.Errorf("npz %s: unknown assigner type %q", path, assignerType)
	}
}

func loadProfiles(weightsDir string) (*Registry, error) {
	profilesDir := filepath.Join(weightsDir, "profiles")
	entries, err := os.ReadDir(profilesDir)
	if err != nil {
		return nil, fmt.Errorf("read profiles dir %s: %w", profilesDir, err)
	}

	registry := NewRegistry()
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		// Skip registry metadata file
		if e.Name() == "_registry.json" {
			continue
		}

		path := filepath.Join(profilesDir, e.Name())
		profile, err := LoadProfile(path)
		if err != nil {
			log.Printf("warning: skipping profile %s: %v", e.Name(), err)
			continue
		}
		registry.Register(profile)
	}

	if registry.Len() == 0 {
		return nil, fmt.Errorf("no valid profiles found in %s", profilesDir)
	}

	return registry, nil
}
