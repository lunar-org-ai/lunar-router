package weights

// Registry holds a collection of LLM profiles for routing.
type Registry struct {
	profiles     map[string]*LLMProfile
	defaultModel string
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{
		profiles: make(map[string]*LLMProfile),
	}
}

// Register adds a profile to the registry.
// The first registered model becomes the default.
func (r *Registry) Register(p *LLMProfile) {
	r.profiles[p.ModelID] = p
	if r.defaultModel == "" {
		r.defaultModel = p.ModelID
	}
}

// Get returns a profile by model ID, or nil if not found.
func (r *Registry) Get(modelID string) *LLMProfile {
	return r.profiles[modelID]
}

// GetAll returns all registered profiles.
func (r *Registry) GetAll() []*LLMProfile {
	result := make([]*LLMProfile, 0, len(r.profiles))
	for _, p := range r.profiles {
		result = append(result, p)
	}
	return result
}

// GetAvailable returns profiles for the given model IDs.
// If modelIDs is nil, returns all profiles.
func (r *Registry) GetAvailable(modelIDs []string) []*LLMProfile {
	if modelIDs == nil {
		return r.GetAll()
	}
	result := make([]*LLMProfile, 0, len(modelIDs))
	for _, id := range modelIDs {
		if p, ok := r.profiles[id]; ok {
			result = append(result, p)
		}
	}
	return result
}

// ModelIDs returns all registered model IDs.
func (r *Registry) ModelIDs() []string {
	ids := make([]string, 0, len(r.profiles))
	for id := range r.profiles {
		ids = append(ids, id)
	}
	return ids
}

// Len returns the number of registered models.
func (r *Registry) Len() int {
	return len(r.profiles)
}

// DefaultModelID returns the default model ID.
func (r *Registry) DefaultModelID() string {
	return r.defaultModel
}
