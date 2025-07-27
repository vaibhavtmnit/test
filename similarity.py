import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz

# --- Helper functions from the previous answer remain the same ---
# (These are included here for a complete, runnable script)

def _preprocess_custom(s: str) -> str:
    """Helper to preprocess strings for the custom metric."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    s = re.sub(r'[aeiou]', '', s)
    return s

def _lcs_length(s1: str, s2: str) -> int:
    """Calculates the length of the Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def _calculate_custom_score(s1: str, s2: str) -> float:
    """Calculates the custom similarity score for a single pair of strings."""
    p1 = _preprocess_custom(s1)
    p2 = _preprocess_custom(s2)
    if len(p1) > len(p2):
        p1, p2 = p2, p1
    if not p1:
        return 10.0 if not p2 else 0.0
    lcs_len = _lcs_length(p1, p2)
    return (lcs_len / len(p1)) * 10.0

# --- UPDATED MAIN FUNCTION ---

def get_cross_similarity_report(array1: np.ndarray, array2: np.ndarray) -> pd.DataFrame:
    """
    Performs a many-to-many comparison between two arrays of strings and
    returns a DataFrame with multiple similarity scores scaled from 0 to 10.

    Args:
        array1: A numpy array of strings.
        array2: A numpy array of strings. Can be a different size than array1.

    Returns:
        A pandas DataFrame with all string pairs and their similarity scores.
    """
    # 1. Create a DataFrame with all possible pairs (Cartesian product)
    df1 = pd.DataFrame({'string1': array1})
    df2 = pd.DataFrame({'string2': array2})
    report_df = pd.merge(df1, df2, how='cross')

    # Extract the paired columns to pass to scoring functions
    s1_col = report_df['string1'].to_numpy()
    s2_col = report_df['string2'].to_numpy()

    # 2. Calculate all scores in a vectorized manner
    # rapidfuzz returns 0-100, so divide by 10 for a 0-10 scale
    report_df['levenshtein_score'] = fuzz.ratio(s1_col, s2_col) / 10.0
    report_df['jaro_winkler_score'] = fuzz.jaro_winkler_similarity(s1_col, s2_col) / 10.0
    report_df['jaccard_score'] = fuzz.QRatio(s1_col, s2_col) / 10.0

    # Vectorize and run the custom scorer
    vectorized_custom_scorer = np.vectorize(_calculate_custom_score)
    report_df['custom_score'] = vectorized_custom_scorer(s1_col, s2_col)

    return report_df

# --- Example Usage ---
if __name__ == '__main__':
    # Create two sample NumPy arrays of different sizes
    arr1 = np.array([
        "WKND",
        "Computer",
    ])

    arr2 = np.array([
        "WAKANDA",
        "computation",
        "Apple"
    ])

    # Generate the similarity report (will have 2 * 3 = 6 rows)
    similarity_df = get_cross_similarity_report(arr1, arr2)

    print("--- Cross-Comparison Similarity Report ---")
    print(similarity_df.round(2))
