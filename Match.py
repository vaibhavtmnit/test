import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union
import multiprocessing

def _preprocess_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Helper to convert datetime columns to int64 for vectorised math."""
    df_copy = df.copy()
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            # Convert to nanoseconds (int64)
            df_copy[col] = df_copy[col].astype('int64')
            # Revert NaT (which become large negative ints) back to NaN equivalent for math
            # However, for int64, NaNs are not supported. 
            # We treat them carefully in the matching logic, but here we keep as numeric.
    return df_copy

def _worker_match(
    left_chunk: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
    tolerances: Dict[str, float],
    match_nulls: bool
) -> pd.DataFrame:
    """
    The worker function running on a specific core.
    """
    
    # 1. Identify Exact vs Fuzzy columns
    # We map left_col -> right_col for easy lookup
    col_map = dict(zip(left_on, right_on))
    
    exact_pairs = []
    fuzzy_pairs = []

    for l_col, r_col in zip(left_on, right_on):
        # If tolerance is defined for this column (check both L and R names)
        tol = tolerances.get(l_col, tolerances.get(r_col))
        if tol is not None and tol >= 0:
            fuzzy_pairs.append((l_col, r_col, tol))
        else:
            exact_pairs.append((l_col, r_col))

    # 2. Perform the Base Merge (Blocking Phase)
    
    # If we have exact columns, we use a hash merge (fastest)
    if exact_pairs:
        l_exact = [p[0] for p in exact_pairs]
        r_exact = [p[1] for p in exact_pairs]
        
        # Handling Null Matches in Merge:
        # Pandas merge drops NaNs. To match NaNs, we fill them with a sentinel.
        if match_nulls:
            # We create temp views to avoid modifying original data in the loop
            l_temp = left_chunk.copy()
            r_temp = right_df.copy()
            
            sentinel = "___NULL_MATCH_SENTINEL___" 
            # Use a numeric sentinel for numeric cols if needed, but string safe for obj
            # For simplicity in this general function, we focus on object/string cols for sentinels
            # or rely on the user knowing exact numeric matches on NaN are rare.
            # Below handles object/string/category types:
            for c in l_exact:
                if l_temp[c].dtype == 'object' or l_temp[c].dtype.name == 'category':
                    l_temp[c] = l_temp[c].fillna(sentinel)
            for c in r_exact:
                 if r_temp[c].dtype == 'object' or r_temp[c].dtype.name == 'category':
                    r_temp[c] = r_temp[c].fillna(sentinel)
            
            merged = pd.merge(l_temp, r_temp, left_on=l_exact, right_on=r_exact, how='inner')
            
            # Restore Nulls if needed or just discard temp frames (merged has result)
        else:
            merged = pd.merge(left_chunk, right_df, left_on=l_exact, right_on=r_exact, how='inner')

    else:
        # EDGE CASE: No exact columns. Pure fuzzy match.
        # We must do a Cross Join (Cartesian Product). 
        # WARNING: This can be memory intensive.
        merged = pd.merge(left_chunk.assign(key=1), right_df.assign(key=1), on='key').drop('key', axis=1)

    # 3. Apply Fuzzy Filtering (Vectorised Phase)
    if fuzzy_pairs and not merged.empty:
        mask = np.ones(len(merged), dtype=bool)
        
        for l_col, r_col, tol in fuzzy_pairs:
            # Extract series from the merged df
            # Note: Merge suffixes might apply if names are identical.
            # Logic to find actual column names in merged result:
            curr_l = l_col
            curr_r = r_col
            
            # If names collide, Pandas adds suffixes. Assumes default _x, _y
            if l_col == r_col:
                curr_l = f"{l_col}_x"
                curr_r = f"{r_col}_y"
            
            series_l = merged[curr_l]
            series_r = merged[curr_r]
            
            # Handle Dates (Convert to numeric for delta calc)
            is_date = pd.api.types.is_datetime64_any_dtype(series_l)
            if is_date:
                # Convert to int64 for subtraction
                vals_l = series_l.values.astype('int64')
                vals_r = series_r.values.astype('int64')
                # Tolerance for dates should be passed as time unit (e.g. nanoseconds for np.datetime64)
                # Or user passes float. We assume user passed compatible float/int tolerance.
            else:
                vals_l = series_l
                vals_r = series_r

            # Calculate Delta
            # We use numpy optimization here
            delta = np.abs(vals_l - vals_r)
            
            # Check Tolerance
            col_mask = (delta <= tol)

            # Check Nulls (if Nulls should match)
            if match_nulls:
                null_mask = (pd.isna(series_l) & pd.isna(series_r))
                col_mask = col_mask | null_mask
            
            mask = mask & col_mask
        
        merged = merged[mask]

    return merged

def flexible_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[List[str]] = None,
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    tolerances: Dict[str, float] = {},
    match_nulls: bool = True,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Master function to coordinate parallel fuzzy merging.
    
    Parameters:
    - left: Left DataFrame
    - right: Right DataFrame
    - on: List of columns to match on (if names are same)
    - left_on: List of columns in Left to match
    - right_on: List of columns in Right to match
    - tolerances: Dict {col_name: tolerance_value}. 
                  For Dates: tolerance is in Nanoseconds (if standard pd.datetime).
                  For Floats/Ints: numeric difference.
    - match_nulls: If True, NaN in left matches NaN in right.
    - n_jobs: Number of CPU cores (-1 for all).
    """
    
    # 1. Argument Normalization
    if on:
        left_on = on
        right_on = on
    
    if not left_on or not right_on:
        raise ValueError("Must provide 'on', or both 'left_on' and 'right_on'")
    
    if len(left_on) != len(right_on):
        raise ValueError("left_on and right_on must have same length")

    # 2. Parallel Execution Setup
    # We split the LEFT dataframe into chunks. 
    # The RIGHT dataframe is broadcasted whole (assumed to fit in memory).
    
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    
    # Calculate chunk size
    chunk_size = int(np.ceil(len(left) / n_cores))
    chunks = [left.iloc[i:i + chunk_size] for i in range(0, len(left), chunk_size)]
    
    # 3. Execute Parallel Merge
    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker_match)(
            chunk, 
            right, 
            left_on, 
            right_on, 
            tolerances, 
            match_nulls
        ) for chunk in chunks
    )
    
    # 4. Concatenate Results
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Create Dummy Data
    data_left = {
        'id_l': ['A', 'B', 'C', 'D', 'E'],
        'date_l': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', pd.NaT, '2023-01-05']),
        'val_l': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat_l': ['foo', 'bar', None, 'baz', 'foo']
    }
    
    data_right = {
        'id_r': ['A', 'A', 'B', 'C', 'X'], # Note duplicates in A
        'date_r': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-04', '2023-01-05']),
        'val_r': [1.05, 1.1, 2.0, 3.1, 5.0], # Slight deviations
        'cat_r': ['foo', 'foo', 'bar', None, 'foo']
    }
    
    df_l = pd.DataFrame(data_left)
    df_r = pd.DataFrame(data_right)

    print("--- Left DF ---")
    print(df_l)
    print("\n--- Right DF ---")
    print(df_r)

    # Define Tolerances
    # 1. 'val' columns: Match if difference <= 0.15
    # 2. 'date' columns: Match if difference <= 1 Day
    # Note: 1 Day in nanoseconds = 86400 * 10^9. 
    # Alternatively, convert dates to days before passing, but here we use raw ns.
    one_day_ns = pd.Timedelta(days=1).value
    
    tols = {
        'val_l': 0.15,
        'date_l': one_day_ns 
    }

    # Execute Merge
    # We map: id_l->id_r (Exact), date_l->date_r (Fuzzy), val_l->val_r (Fuzzy), cat_l->cat_r (Exact + Null)
    result = flexible_merge(
        left=df_l,
        right=df_r,
        left_on=['cat_l', 'val_l', 'date_l'],
        right_on=['cat_r', 'val_r', 'date_r'],
        tolerances=tols,
        match_nulls=True,
        n_jobs=2
    )

    print("\n--- Match Result ---")
    # Expected:
    # A (foo) matches A(foo) because dates and vals are within tolerance.
    # B (bar) matches B(bar).
    # C (None) matches C(None) because match_nulls=True for 'cat'.
    print(result)
