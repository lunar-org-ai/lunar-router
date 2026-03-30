package weights

import (
	"encoding/json"
	"fmt"
	"os"
)

// LLMProfile stores the Psi vector representation of a model.
// The Psi vector contains the average error rate per cluster
// computed during profiling.
type LLMProfile struct {
	ModelID              string         `json:"model_id"`
	PsiVector            []float64      `json:"psi_vector"`
	CostPer1kTokens      float64        `json:"cost_per_1k_tokens"`
	NumValidationSamples int            `json:"num_validation_samples"`
	ClusterSampleCounts  []float64      `json:"cluster_sample_counts"`
	Metadata             map[string]any `json:"metadata,omitempty"`
}

// NumClusters returns the number of clusters K.
func (p *LLMProfile) NumClusters() int {
	return len(p.PsiVector)
}

// OverallErrorRate computes the weighted average error rate across all clusters.
func (p *LLMProfile) OverallErrorRate() float64 {
	var totalSamples float64
	for _, c := range p.ClusterSampleCounts {
		totalSamples += c
	}
	if totalSamples == 0 {
		return 0
	}
	var weightedSum float64
	for i, psi := range p.PsiVector {
		weightedSum += psi * p.ClusterSampleCounts[i]
	}
	return weightedSum / totalSamples
}

// OverallAccuracy returns 1 - overall error rate.
func (p *LLMProfile) OverallAccuracy() float64 {
	return 1.0 - p.OverallErrorRate()
}

// ExpectedError computes γ(x, h) = Φ(x)ᵀ · Ψ(h).
func (p *LLMProfile) ExpectedError(phi []float64) float64 {
	var sum float64
	for i, v := range phi {
		sum += v * p.PsiVector[i]
	}
	return sum
}

// LoadProfile reads a model profile from a JSON file.
func LoadProfile(path string) (*LLMProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read profile %s: %w", path, err)
	}
	var p LLMProfile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse profile %s: %w", path, err)
	}
	if p.ModelID == "" {
		return nil, fmt.Errorf("profile %s: missing model_id", path)
	}
	if len(p.PsiVector) == 0 {
		return nil, fmt.Errorf("profile %s: empty psi_vector", path)
	}
	return &p, nil
}
