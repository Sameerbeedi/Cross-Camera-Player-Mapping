import numpy as np
from scipy.spatial.distance import cosine

def compute_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def match_features(features1, features2):
    matches = []
    used_indices = set()

    for idx1, obj1 in enumerate(features1):
        best_score = -1
        best_match_idx = -1

        for idx2, obj2 in enumerate(features2):
            if idx2 in used_indices:
                continue

            sim = compute_cosine_similarity(obj1["feature"], obj2["feature"])

            if sim > best_score:
                best_score = sim
                best_match_idx = idx2

        if best_match_idx != -1:
            used_indices.add(best_match_idx)
            matches.append((idx1, best_match_idx, best_score))

    return matches
