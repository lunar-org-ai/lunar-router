"""
Learned Map Trainer: Trains the supervised cluster assignment function.

Implements Section 5.2 of the paper: learns parameter matrix θ to optimize
the cluster assignment function Φ(x; θ) using labeled data from training LLMs.
"""

from typing import Optional
import numpy as np
from tqdm import tqdm

from ..core.embeddings import PromptEmbedder
from ..core.clustering import KMeansClusterAssigner, LearnedMapClusterAssigner
from ..core.metrics import MetricType, get_metric
from ..models.llm_client import LLMClient
from ..data.dataset import PromptDataset


class LearnedMapTrainer:
    """
    Trains the learned cluster map θ.

    Uses supervision from training LLMs (H_tr) to optimize θ such that
    Φ(x; θ)·Ψ(h) approximates the actual error of model h on prompt x.

    The optimization minimizes the log-loss:
        L(θ) = Σ_{(x,y)∈S_tr} Σ_{h∈H_tr} -[ℓ·log(γ) + (1-ℓ)·log(1-γ)]

    where:
        - ℓ = 1{y ≠ h(x)} is the actual error
        - γ = Φ(x; θ)·Ψ(h) is the predicted error

    Attributes:
        embedder: Prompt embedder for φ(x).
        base_assigner: K-Means assigner used to define clusters.
        lr: Learning rate.
        temperature: Temperature for softmax.
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        base_assigner: KMeansClusterAssigner,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
    ):
        """
        Initialize the trainer.

        Args:
            embedder: PromptEmbedder for generating embeddings.
            base_assigner: K-Means assigner with pre-computed centroids.
            learning_rate: Learning rate for gradient descent.
            temperature: Temperature parameter τ for softmax.
        """
        self.embedder = embedder
        self.base_assigner = base_assigner
        self.lr = learning_rate
        self.temperature = temperature

    def train(
        self,
        training_set: PromptDataset,
        training_llms: list[LLMClient],
        epochs: int = 50,
        metric: MetricType = MetricType.EXACT_MATCH,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        early_stopping_patience: int = 5,
    ) -> LearnedMapClusterAssigner:
        """
        Train θ using training data and LLMs.

        Args:
            training_set: S_tr dataset with prompts and ground truths.
            training_llms: H_tr - LLMs used for supervision.
            epochs: Number of training epochs.
            metric: Metric for computing errors.
            batch_size: Optional batch size (None = full batch).
            verbose: Whether to print progress.
            early_stopping_patience: Stop if loss doesn't improve for this many epochs.

        Returns:
            LearnedMapClusterAssigner with trained θ.
        """
        loss_fn = get_metric(metric)
        K = self.base_assigner.num_clusters

        # Step 1: Collect training data (embeddings and error labels)
        if verbose:
            print("Collecting training data...")

        training_data = self._collect_training_data(
            training_set, training_llms, loss_fn, verbose
        )

        # Step 2: Compute Ψ vectors for training LLMs
        psi_train = self._compute_training_psi(training_data, K)

        # Step 3: Initialize θ
        d = training_data[0]["embedding"].shape[0]
        theta = np.random.randn(K, d) * 0.01

        # Step 4: Train θ via gradient descent
        if verbose:
            print(f"Training θ with {len(training_data)} samples, {len(training_llms)} LLMs...")

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(training_data))

            # Process in batches or full batch
            if batch_size is None:
                batches = [indices]
            else:
                batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

            epoch_loss = 0.0

            for batch_idx in batches:
                grad = np.zeros_like(theta)
                batch_loss = 0.0

                for idx in batch_idx:
                    data = training_data[idx]
                    phi_x = data["embedding"]

                    # Forward pass: compute Φ(x; θ) via softmax
                    logits = theta @ phi_x / self.temperature
                    logits_stable = logits - np.max(logits)
                    exp_logits = np.exp(logits_stable)
                    probs = exp_logits / exp_logits.sum()

                    # Compute loss and gradients for each training LLM
                    for h_idx, error_true in enumerate(data["errors"]):
                        psi_h = psi_train[h_idx]

                        # Predicted error: γ = Φ(x; θ)·Ψ(h)
                        gamma = np.dot(probs, psi_h)
                        gamma = np.clip(gamma, 1e-7, 1 - 1e-7)

                        # Log-loss
                        loss = -(error_true * np.log(gamma) + (1 - error_true) * np.log(1 - gamma))
                        batch_loss += loss

                        # Gradient of loss w.r.t. gamma
                        d_loss_d_gamma = (gamma - error_true) / (gamma * (1 - gamma))

                        # Gradient of gamma w.r.t. probs
                        # γ = Σ_k probs_k * Ψ_k(h)
                        # ∂γ/∂probs_k = Ψ_k(h)

                        # Gradient of probs w.r.t. logits (softmax derivative)
                        # ∂probs_k/∂logits_j = probs_k * (δ_kj - probs_j)

                        # Gradient of logits w.r.t. θ_k
                        # ∂logits_k/∂θ_k = φ(x) / τ

                        # Chain rule for each θ_k:
                        for k in range(K):
                            # ∂γ/∂θ_k = Σ_j Ψ_j(h) * probs_j * (δ_jk - probs_k) * φ(x) / τ
                            softmax_grad = probs[k] * (psi_h[k] - np.dot(psi_h, probs))
                            grad[k] += d_loss_d_gamma * softmax_grad * phi_x / self.temperature

                epoch_loss += batch_loss

                # Update θ
                theta -= self.lr * grad / len(batch_idx)

            epoch_loss /= (len(training_data) * len(training_llms))

            # Early stopping check
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"Training complete. Final loss: {epoch_loss:.6f}")

        return LearnedMapClusterAssigner(
            centroids=self.base_assigner.centroids,
            theta=theta,
            temperature=self.temperature,
        )

    def _collect_training_data(
        self,
        training_set: PromptDataset,
        training_llms: list[LLMClient],
        loss_fn,
        verbose: bool,
    ) -> list[dict]:
        """Collect embeddings and error labels for all training samples."""
        data = []

        iterator = training_set
        if verbose:
            iterator = tqdm(iterator, desc="Collecting data")

        for prompt, ground_truth in iterator:
            embedding = self.embedder.embed(prompt)

            # Get error label for each training LLM
            errors = []
            for llm in training_llms:
                response = llm.generate(prompt)
                error = loss_fn(response.text, ground_truth)
                errors.append(error)

            data.append({
                "embedding": embedding,
                "errors": np.array(errors),
            })

        return data

    def _compute_training_psi(self, training_data: list[dict], K: int) -> np.ndarray:
        """
        Compute Ψ vectors for training LLMs using the base K-Means assigner.

        Args:
            training_data: List of dicts with embeddings and errors.
            K: Number of clusters.

        Returns:
            Array of shape (num_llms, K) with error rates per cluster.
        """
        num_llms = len(training_data[0]["errors"])
        psi = np.zeros((num_llms, K))
        counts = np.zeros(K)

        for data in training_data:
            cluster_result = self.base_assigner.assign(data["embedding"])
            k = cluster_result.cluster_id
            counts[k] += 1

            for h_idx, error in enumerate(data["errors"]):
                psi[h_idx, k] += error

        # Normalize
        for k in range(K):
            if counts[k] > 0:
                psi[:, k] /= counts[k]
            else:
                # Use global average for empty clusters
                psi[:, k] = psi.sum() / counts.sum() if counts.sum() > 0 else 0.5

        return psi


class LearnedMapValidator:
    """
    Utilities for validating learned map quality.
    """

    @staticmethod
    def evaluate_routing_accuracy(
        assigner: LearnedMapClusterAssigner,
        kmeans_assigner: KMeansClusterAssigner,
        embedder: PromptEmbedder,
        test_set: PromptDataset,
        test_llms: list[LLMClient],
        metric: MetricType = MetricType.EXACT_MATCH,
    ) -> dict:
        """
        Compare learned map vs K-Means routing accuracy.

        Args:
            assigner: Learned map assigner.
            kmeans_assigner: Base K-Means assigner.
            embedder: Prompt embedder.
            test_set: Test dataset.
            test_llms: Test LLMs to route between.
            metric: Error metric.

        Returns:
            Dictionary with comparison statistics.
        """
        loss_fn = get_metric(metric)

        # Compute Ψ for test LLMs using K-Means
        K = kmeans_assigner.num_clusters
        psi_test = np.zeros((len(test_llms), K))
        counts = np.zeros(K)

        # First pass: collect Ψ
        for prompt, ground_truth in test_set:
            embedding = embedder.embed(prompt)
            cluster_result = kmeans_assigner.assign(embedding)
            k = cluster_result.cluster_id
            counts[k] += 1

            for h_idx, llm in enumerate(test_llms):
                response = llm.generate(prompt)
                error = loss_fn(response.text, ground_truth)
                psi_test[h_idx, k] += error

        # Normalize Ψ
        for k in range(K):
            if counts[k] > 0:
                psi_test[:, k] /= counts[k]

        # Second pass: compare routing decisions
        kmeans_correct = 0
        learned_correct = 0
        total = 0

        for prompt, ground_truth in test_set:
            embedding = embedder.embed(prompt)

            # Get actual best LLM
            actual_errors = []
            for llm in test_llms:
                response = llm.generate(prompt)
                error = loss_fn(response.text, ground_truth)
                actual_errors.append(error)
            best_actual = int(np.argmin(actual_errors))

            # K-Means routing
            kmeans_result = kmeans_assigner.assign(embedding)
            kmeans_scores = [psi_test[h, kmeans_result.cluster_id] for h in range(len(test_llms))]
            kmeans_choice = int(np.argmin(kmeans_scores))

            # Learned map routing
            learned_result = assigner.assign(embedding)
            learned_scores = [np.dot(learned_result.probabilities, psi_test[h]) for h in range(len(test_llms))]
            learned_choice = int(np.argmin(learned_scores))

            if kmeans_choice == best_actual:
                kmeans_correct += 1
            if learned_choice == best_actual:
                learned_correct += 1
            total += 1

        return {
            "total_samples": total,
            "kmeans_accuracy": kmeans_correct / total if total > 0 else 0,
            "learned_accuracy": learned_correct / total if total > 0 else 0,
            "improvement": (learned_correct - kmeans_correct) / total if total > 0 else 0,
        }
