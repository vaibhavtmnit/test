import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process

def _preprocess_custom(s: str) -> str:
    """Helper to preprocess strings for the custom metric."""
    # 1. Convert to lower
    s = s.lower()
    # 2. Keep only letters and numbers
    s = re.sub(r'[^a-z0-9]', '', s)
    # 3. Ignore vowels
    s = re.sub(r'[aeiou]', '', s)
    return s

def _lcs_length(s1: str, s2: str) -> int:
    """Calculates the length of the Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    # Create a DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def _calculate_custom_score(s1: str, s2: str) -> float:
    """
    Calculates the custom similarity score based on the Longest Common Subsequence
    of the processed strings.
    """
    # Preprocess both strings according to the rules
    p1 = _preprocess_custom(s1)
    p2 = _preprocess_custom(s2)

    # Ensure we compare the smaller to the larger
    if len(p1) > len(p2):
        p1, p2 = p2, p1 # p1 is now the shorter string

    if not p1: # If the shorter string is empty after processing
        return 10.0 if not p2 else 0.0

    # Find the length of the longest common subsequence
    lcs_len = _lcs_length(p1, p2)

    # Score is the ratio of the LCS length to the length of the shorter string
    # This correctly handles the "subsequence" match rule.
    # If p1 is a subsequence of p2, lcs_len will be len(p1).
    score = (lcs_len / len(p1)) * 10.0
    return score


def get_similarity_report(array1: np.ndarray, array2: np.ndarray) -> pd.DataFrame:
    """
    Compares two numpy arrays of strings and returns a DataFrame with multiple
    similarity scores, all scaled from 0 (dissimilar) to 10 (identical).

    Args:
        array1: A numpy array of strings.
        array2: A numpy array of strings to compare against array1.

    Returns:
        A pandas DataFrame with original strings and their similarity scores.
    """
    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # --- 1. Calculate Standard Scores (already vectorized by rapidfuzz) ---
    # rapidfuzz returns scores from 0-100, so we divide by 10 for a 0-10 scale.
    lev_scores = fuzz.ratio(array1, array2) / 10.0
    jaro_scores = fuzz.jaro_winkler_similarity(array1, array2) / 10.0
    jaccard_scores = fuzz.QRatio(array1, array2) / 10.0 # QRatio is a good Jaccard-like metric

    # --- 2. Calculate Custom Score (using np.vectorize) ---
    vectorized_custom_scorer = np.vectorize(_calculate_custom_score)
    custom_scores = vectorized_custom_scorer(array1, array2)

    # --- 3. Assemble the DataFrame ---
    report = pd.DataFrame({
        'string1': array1,
        'string2': array2,
        'levenshtein_score': lev_scores,
        'jaro_winkler_score': jaro_scores,
        'jaccard_score': jaccard_scores,
        'custom_score': custom_scores
    })

    return report

# --- Example Usage ---
if __name__ == '__main__':
    # Create two sample NumPy arrays of strings
    arr1 = np.array([
        "WKND",
        "Python",
        "Computer",
        "Street",
        "Apple Inc.",
        "international"
    ])

    arr2 = np.array([
        "WAKANDA",
        "path",
        "computation",
        "Saint",
        "Apple",
        "national"
    ])

    # Generate the similarity report
    similarity_df = get_similarity_report(arr1, arr2)

    # Print the results
    print("--- String Similarity Report ---")
    print(similarity_df.round(2))
