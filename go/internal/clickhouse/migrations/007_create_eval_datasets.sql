-- 007: Evaluation datasets and samples.

CREATE TABLE IF NOT EXISTS eval_datasets
(
    id              String,
    name            String,
    description     String,
    source          LowCardinality(String),
    samples_count   UInt32,
    created_at      DateTime64(3, 'UTC'),
    updated_at      DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (id);

CREATE TABLE IF NOT EXISTS eval_dataset_samples
(
    id              String,
    dataset_id      String,
    input           String,
    output          String,
    expected_output String,
    metadata        String,
    created_at      DateTime64(3, 'UTC')
)
ENGINE = MergeTree
ORDER BY (dataset_id, id);
