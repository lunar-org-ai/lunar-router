-- 008: Add output_text to trace_cluster_map for distillation training data.

ALTER TABLE trace_cluster_map ADD COLUMN IF NOT EXISTS output_text String DEFAULT '';
