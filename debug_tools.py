import numpy as np
import pandas as pd
import matplotlib as plt

def analyze_label_threshold(y_true, thresholds, returns=None, verbose=True, plot=True):
    """
    Prints label distributions for a list of label_thresholds.
    If returns are passed, computes the actual labels for each threshold.
    """
    if returns is None:
        returns = y_true
        make_labels = False
    else:
        make_labels = True

    records = []
    for thresh in thresholds:
        if make_labels:
            # This assumes you do 0: short, 1: neutral, 2: long
            labels = np.where(returns > thresh, 1, np.where(returns < -thresh, -1, 0))
            labels = pd.Series(labels)
        else:
            labels = pd.Series(y_true)
        counts = labels.value_counts().sort_index()
        if verbose:
            print(f"--- label_threshold={thresh:.6f} ---")
            print(counts)
        records.append(counts)
    if plot:
        # Plot as stacked bar
        df = pd.DataFrame(records, index=[f"{t:.5f}" for t in thresholds]).fillna(0)
        df.plot(kind='bar', stacked=True)
        plt.xlabel("label_threshold")
        plt.ylabel("Count")
        plt.title("Label distribution vs threshold")
        plt.show()


def compare_and_align_indices(df1, df2, name1="Model1", name2="Model2"):
    """
    Align two DataFrames on their indices, print index differences, and return aligned DataFrames.
    """
    idx1 = set(df1.index)
    idx2 = set(df2.index)
    only1 = idx1 - idx2
    only2 = idx2 - idx1
    print(f"\n[Diagnostic] {name1} rows not in {name2}: {len(only1)}")
    print(f"[Diagnostic] {name2} rows not in {name1}: {len(only2)}")
    if only1:
        print(f"First 3 indices in {name1} but not in {name2}: {list(only1)[:3]}")
    if only2:
        print(f"First 3 indices in {name2} but not in {name1}: {list(only2)[:3]}")
    common = idx1 & idx2
    df1_aligned = df1.loc[df1.index.intersection(common)].sort_index()
    df2_aligned = df2.loc[df2.index.intersection(common)].sort_index()
    return df1_aligned, df2_aligned

def print_feature_stats(df, label):
    print(f"\n[DEBUG] {label} feature stats:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    

def validate_result_shape(result, source=""):
    if isinstance(result, tuple) and len(result) in (12, 14):
        return True
    print(f"❌ Invalid result shape from {source}: got tuple of length {len(result)}")
    return False

def compute_composite_score(
    sharpe, drawdown, f1_macro, active_rate, geo_mean_ann,
    weight_scheme="return_focus"
):
    """
    Legacy-style score, patched to accept newer arguments for compatibility.
    """
    sharpe = np.nan_to_num(np.clip(sharpe, -5, 5), nan=0.0)
    drawdown = np.clip(drawdown, 0, 1)
    f1_macro = np.clip(f1_macro, 0, 1)
    active_rate = np.clip(active_rate, 0, 1)
    geo_mean_ann = np.nan_to_num(np.clip(geo_mean_ann, -1, 5), nan=0.0)

    norm_return = np.log1p(geo_mean_ann)
    norm_drawdown = -drawdown
    norm_sharpe = sharpe / 2
    norm_f1 = f1_macro
    norm_active = np.sqrt(active_rate)

    score_components = [norm_return, norm_drawdown, norm_sharpe, norm_f1, norm_active]

    weights_dict = {
        "return_focus": [0.8, 0.05, 0.05, 0.05, 0.05],
        "balanced":     [0.3, 0.2, 0.2, 0.15, 0.15],
    }
    weights = weights_dict.get(weight_scheme, weights_dict["return_focus"])
    
    final_score = sum(w * m for w, m in zip(weights, score_components))
    return round(final_score, 6)

def verify_mtf_no_future_leak(df, column, long_tf, long_window):
    """
    Checks that each row's MTF feature does NOT use future data.
    Prints rows where leakage is detected.
    """
    # Calculate the true rolling mean on the resampled series
    resampled = df[column].resample(long_tf).mean()
    rolling = resampled.rolling(long_window, min_periods=1).mean()
    # Map each original timestamp to the most recent resample timestamp (never future)
    resample_map = resampled.index.searchsorted(df.index, side='right') - 1
    resample_map = np.clip(resample_map, 0, len(resampled)-1)
    mapped_index = resampled.index[resample_map]
    # For each row, get the rolling mean value that would be available at that time
    safe_mtf = rolling.reindex(mapped_index).values
    # Compare to your MTF feature
    mtf_col = f"mtf_ma_slow"  # or whatever your column is named
    diff = np.abs(df[mtf_col].values - safe_mtf)
    leakage_rows = np.where(diff > 1e-8)[0]
    if len(leakage_rows) > 0:
        print(f"⚠️ Potential future leakage detected in {mtf_col} at rows: {leakage_rows}")
    else:
        print(f"✅ No future leakage detected in {mtf_col}.")
        
def multi_timeframe_ma_safe(df, column='price', short_window=10, long_window=50, short_tf='30T', long_tf='4H'):
    short_ma = df[column].rolling(short_window).mean()
    # Resample and compute rolling mean, then shift so value at t uses only data up to t
    resampled = df[column].resample(long_tf, label='right', closed='right').mean()
    rolling = resampled.rolling(long_window, min_periods=1).mean().shift(1)
    # Map each original timestamp to the most recent resample timestamp (never future)
    resample_map = resampled.index.searchsorted(df.index, side='right') - 1
    resample_map = np.clip(resample_map, 0, len(resampled)-1)
    mapped_index = resampled.index[resample_map]
    safe_mtf = rolling.reindex(mapped_index).values
    return short_ma, pd.Series(safe_mtf, index=df.index)


def multi_timeframe_ma(df, column='close', short_window=10, long_window=50, short_tf='30T', long_tf='4H'):
    """
    Example of Multi-Timeframe MA: computes short MA on current, long MA on resampled higher TF.
    Returns two columns: MTF_short, MTF_long.
    """
    short_ma = df[column].rolling(short_window).mean()
    # For long timeframe, first resample, then forward-fill, then join back to main index
    long_ma = (df[column]
               .resample(long_tf)
               .mean()
               .rolling(long_window).mean()
               .reindex(df.index, method='ffill'))
    return short_ma, long_ma