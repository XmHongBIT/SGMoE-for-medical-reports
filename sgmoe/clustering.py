from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpertAssignment:
    cluster_categories: dict[int, list[str]]
    expert_categories: dict[int, list[str]]
    center_to_expert_mapping: dict[int, int]


def choose_cluster_count(scores_by_k):
    """Return the k with the best silhouette score."""
    if not scores_by_k:
        raise ValueError("scores_by_k must not be empty")
    return max(scores_by_k, key=lambda k: (scores_by_k[k], -k))


def build_expert_assignment(labels, cluster_ids, num_clusters):
    """Build paper-style cluster-to-expert assignment.

    A class is kept in cluster j when n_i,j / n_i >= 1 / N, where
    N is the number of semantic clusters. If denoising would remove
    a class from every cluster, the class is restored to its best
    cluster. Clusters containing only one class are merged into one
    expert; each remaining multi-class cluster receives its own expert.
    """
    labels = [str(label) for label in labels]
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if len(labels) != len(cluster_ids):
        raise ValueError("labels and cluster_ids must have the same length")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")

    unique_labels = sorted(set(labels), key=_label_sort_key)
    threshold = 1.0 / float(num_clusters)
    label_totals = {label: labels.count(label) for label in unique_labels}
    counts = {
        cluster_idx: {label: 0 for label in unique_labels}
        for cluster_idx in range(num_clusters)
    }
    for label, cluster_idx in zip(labels, cluster_ids):
        if cluster_idx < 0 or cluster_idx >= num_clusters:
            raise ValueError(f"cluster id {cluster_idx} is outside [0, {num_clusters})")
        counts[int(cluster_idx)][label] += 1

    cluster_categories = {}
    assigned_by_label = {label: [] for label in unique_labels}
    for cluster_idx in range(num_clusters):
        kept = []
        for label in unique_labels:
            ratio = counts[cluster_idx][label] / max(label_totals[label], 1)
            if counts[cluster_idx][label] > 0 and ratio >= threshold:
                kept.append(label)
                assigned_by_label[label].append(cluster_idx)
        if kept:
            cluster_categories[cluster_idx] = kept

    for label in unique_labels:
        if assigned_by_label[label]:
            continue
        best_cluster = max(
            range(num_clusters),
            key=lambda cluster_idx: counts[cluster_idx][label],
        )
        if counts[best_cluster][label] == 0:
            continue
        cluster_categories.setdefault(best_cluster, [])
        if label not in cluster_categories[best_cluster]:
            cluster_categories[best_cluster].append(label)

    for cluster_idx in list(cluster_categories):
        cluster_categories[cluster_idx] = sorted(
            cluster_categories[cluster_idx],
            key=_label_sort_key,
        )

    expert_categories = {}
    center_to_expert_mapping = {}
    single_cluster_labels = []
    single_cluster_ids = []

    for cluster_idx in sorted(cluster_categories):
        categories = cluster_categories[cluster_idx]
        if len(categories) == 1:
            single_cluster_ids.append(cluster_idx)
            single_cluster_labels.extend(categories)
            continue
        expert_idx = len(expert_categories)
        expert_categories[expert_idx] = categories
        center_to_expert_mapping[cluster_idx] = expert_idx

    if single_cluster_ids:
        expert_idx = len(expert_categories)
        expert_categories[expert_idx] = sorted(set(single_cluster_labels), key=_label_sort_key)
        for cluster_idx in single_cluster_ids:
            center_to_expert_mapping[cluster_idx] = expert_idx

    return ExpertAssignment(
        cluster_categories=dict(sorted(cluster_categories.items())),
        expert_categories=expert_categories,
        center_to_expert_mapping=dict(sorted(center_to_expert_mapping.items())),
    )


def fit_kmeans_with_silhouette(embeddings, min_clusters, max_clusters, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    embeddings = np.asarray(embeddings, dtype=np.float32)
    max_allowed = min(int(max_clusters), len(embeddings) - 1)
    min_allowed = max(2, int(min_clusters))
    if max_allowed < min_allowed:
        raise ValueError("not enough samples to select a cluster count")

    scores = {}
    fitted = {}
    for k in range(min_allowed, max_allowed + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        cluster_ids = model.fit_predict(embeddings)
        if len(set(cluster_ids.tolist())) < 2:
            continue
        scores[k] = float(silhouette_score(embeddings, cluster_ids))
        fitted[k] = model
    best_k = choose_cluster_count(scores)
    return best_k, fitted[best_k], scores


def write_cluster_centers(centers, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in np.asarray(centers, dtype=np.float32):
            handle.write(",".join(f"{float(value):.8f}" for value in row))
            handle.write("\n")


def _label_sort_key(label):
    try:
        return (0, int(label))
    except ValueError:
        return (1, str(label))
