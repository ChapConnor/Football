import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os
from tqdm import tqdm  # <--- NEW IMPORT

# ==========================================
# 0. CONFIGURATION
# ==========================================

# --- INSTRUCTIONS ---
# Add as many file pairings as you want to the list below.
JOBS_LIST = [
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_west_Practice_3.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_3.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_west_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_2.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_west_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_1.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_east_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_1.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_east_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_2.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2024_east_Practice_3.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_3.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2022_east_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_east_Practice_1.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2022_west_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_west_Practice_2.snappy.parquet'
    },
    {
        "matchups": r'/Users/connorchapman/Downloads/detected_matchups_v23_VECTOR_FIX_2022_west_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_west_Practice_1.snappy.parquet'
    },
]

# --- ANALYSIS SETTINGS ---
SMOOTHING_SIGMA = 2.0
BREAK_THRESHOLD_DEG = 8.0


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def generate_output_filename(input_path):
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name_only, ext = os.path.splitext(base_name)
    new_name = f"{name_only}_BATCH_RESULTS{ext}"
    return os.path.join(dir_name, new_name)


def calculate_shortest_angle_diff(angle_series):
    diffs = np.diff(angle_series, prepend=angle_series.iloc[0])
    diffs = np.where(diffs > 180, diffs - 360, diffs)
    diffs = np.where(diffs < -180, diffs + 360, diffs)
    return np.abs(diffs)


def identify_smart_break(player_df):
    """
    FIXED LOGIC (v24):
    - Allows players to decelerate during the cut (Dig/Comeback).
    - Checks if max speed *during the route* was high.
    """
    df = player_df.sort_values('frameId').reset_index(drop=True)
    if df.empty: return None

    # 1. Route Validation
    if df['s'].max() < 4.0:
        return {'frame_idx': df.iloc[0]['frameId'], 'cut_type': 'Jog/Walk (Ignored)'}

    # 2. Turn Rate Calculation
    angle_change = calculate_shortest_angle_diff(df['o'])
    smoothed_change = gaussian_filter1d(angle_change, sigma=SMOOTHING_SIGMA)

    # 3. Scan for Peaks
    peaks, _ = find_peaks(smoothed_change, height=BREAK_THRESHOLD_DEG, distance=10)

    valid_peaks = []
    for p in peaks:
        if df['s'].iloc[p] > 1.0:
            valid_peaks.append(p)

    best_cut_idx = None
    cut_type = "None"

    # --- SCENARIO A: Found Sharp Cuts ---
    if valid_peaks:
        first_cut_idx = valid_peaks[0]
        best_cut_idx = first_cut_idx
        cut_type = "Angle Cut"

        # Check for Double Move
        if len(valid_peaks) > 1:
            first_cut_mag = smoothed_change[first_cut_idx]
            for next_cut_idx in valid_peaks[1:]:
                next_cut_mag = smoothed_change[next_cut_idx]
                if next_cut_mag > (first_cut_mag * 0.8):
                    best_cut_idx = next_cut_idx
                    cut_type = "Double Move"

    # --- SCENARIO B: No Sharp Cuts ---
    else:
        start_x = df.iloc[0]['x']
        start_y = df.iloc[0]['y']
        dists = np.sqrt((df['x'] - start_x) ** 2 + (df['y'] - start_y) ** 2)

        move_mask = dists > 2.0
        if move_mask.any():
            best_cut_idx = move_mask.idxmax()
            cut_type = "Speed Release (LOS)"
        else:
            best_cut_idx = 0
            cut_type = "Static Start"

    return {
        'frame_idx': df.iloc[best_cut_idx]['frameId'],
        'cut_type': cut_type
    }


def calculate_blind_spot_score(row):
    if pd.isna(row.get('db_o')): return 0
    db_o_rad = np.radians(row['db_o'])
    target_x = row['x_wr'] - row['x_db']
    target_y = row['y_wr'] - row['y_db']
    dist = np.sqrt(target_x ** 2 + target_y ** 2)
    if dist < 0.1: return 0
    dot = (np.sin(db_o_rad) * (target_x / dist)) + (np.cos(db_o_rad) * (target_y / dist))
    return -dot


# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
def process_dataset(matchups_file, parquet_file, job_index, total_jobs):
    print(f"\n--- JOB {job_index}/{total_jobs} ---")
    print(f"Matchups: {os.path.basename(matchups_file)}")
    print(f"Tracking: {os.path.basename(parquet_file)}")

    output_file = generate_output_filename(matchups_file)

    try:
        df_matchups = pd.read_csv(matchups_file)
        df_matchups['rep_start'] = pd.to_datetime(df_matchups['rep_start'])
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading matchups: {e}")
        return

    print("Loading Parquet Tracking Data (this may take a moment)...")
    try:
        full_tracking = pd.read_parquet(parquet_file, columns=['ts', 'gsis_id', 'x', 'y', 's', 'dir'])
        full_tracking['ts'] = pd.to_datetime(full_tracking['ts'])
        full_tracking['o'] = full_tracking['dir']
        full_tracking['gsis_id'] = full_tracking['gsis_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        full_tracking = full_tracking.sort_values('ts')
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading parquet: {e}")
        return

    results = []

    # --- TQDM WRAPPER HERE ---
    # wrapping iterrows to create the progress bar
    progress_bar = tqdm(df_matchups.iterrows(), total=df_matchups.shape[0],
                        desc="Processing Reps", unit="rep")

    for i, row in progress_bar:
        try:
            t_start = row['rep_start']
            t_end = t_start + timedelta(seconds=8)
            wr_id = str(int(float(row['WR_ID'])))
            db_id = str(int(float(row['DB_ID'])))

            play_window = full_tracking[
                (full_tracking['ts'] >= t_start) &
                (full_tracking['ts'] <= t_end)
                ].copy()

            if play_window.empty: continue

            play_data = play_window[play_window['gsis_id'].isin([wr_id, db_id])].copy()
            unique_times = play_data['ts'].sort_values().unique()
            time_map = {t: idx for idx, t in enumerate(unique_times)}
            play_data['frameId'] = play_data['ts'].map(time_map)

            play_data.loc[play_data['gsis_id'] == wr_id, 'position'] = 'WR'
            play_data.loc[play_data['gsis_id'] == db_id, 'position'] = 'CB'

            wr_data = play_data[play_data['position'] == 'WR'].drop_duplicates('frameId').sort_values('frameId')
            db_data = play_data[play_data['position'] == 'CB'].drop_duplicates('frameId').sort_values('frameId')

            if wr_data.empty or db_data.empty:
                results.append({**row, 'Status': 'Missing Data'})
                continue

            merged = pd.merge(
                wr_data[['frameId', 'x', 'y', 's', 'o']],
                db_data[['frameId', 'x', 'y', 'o']].rename(columns={'x': 'x_db', 'y': 'y_db', 'o': 'db_o'}),
                on='frameId', how='inner'
            )

            merged['x_wr'] = merged['x']
            merged['y_wr'] = merged['y']
            merged['raw_dist'] = np.sqrt(
                (merged['x_wr'] - merged['x_db']) ** 2 + (merged['y_wr'] - merged['y_db']) ** 2)
            merged['effective_sep'] = merged['raw_dist'] * (1 + merged.apply(calculate_blind_spot_score, axis=1))

            # Find Break
            break_event = identify_smart_break(wr_data)

            # Calculate Stats
            stats = {
                'Status': 'Success',
                'Sep_0_1s': np.nan,
                'Sep_1_2s': np.nan,
                'Sep_2_3s': np.nan,
                'Break_Frame': np.nan,
                'Cut_Type': 'None'
            }

            if break_event:
                bf = break_event['frame_idx']
                stats['Break_Frame'] = bf
                stats['Cut_Type'] = break_event['cut_type']

                post_break = merged[merged['frameId'] >= bf]
                if not post_break.empty:
                    stats['Sep_0_1s'] = post_break['effective_sep'].iloc[0:10].mean()
                    stats['Sep_1_2s'] = post_break['effective_sep'].iloc[10:20].mean()
                    stats['Sep_2_3s'] = post_break['effective_sep'].iloc[20:30].mean()

            full_row = {**row, **stats}
            results.append(full_row)

        except Exception as e:
            error_row = {**row, 'Status': f"Error: {str(e)}"}
            results.append(error_row)

    # Wrap up single file
    df_results = pd.DataFrame(results)

    # Rounding
    cols_to_round = ['Sep_0_1s', 'Sep_1_2s', 'Sep_2_3s']
    for col in cols_to_round:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(2)

    df_results.to_csv(output_file, index=False)
    print(f"--> Saved to: {output_file}")


# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    total_jobs = len(JOBS_LIST)
    print(f"--- BATCH PROCESS INITIALIZED: Found {total_jobs} file pairings ---")

    for i, job in enumerate(JOBS_LIST):
        matchups_path = job.get('matchups')
        tracking_path = job.get('tracking')

        if not matchups_path or not tracking_path:
            print(f"Skipping Job {i + 1}: Missing file path configuration.")
            continue

        if not os.path.exists(matchups_path):
            print(f"Skipping Job {i + 1}: Matchups file not found ({matchups_path})")
            continue

        if not os.path.exists(tracking_path):
            print(f"Skipping Job {i + 1}: Parquet file not found ({tracking_path})")
            continue

        process_dataset(matchups_path, tracking_path, i + 1, total_jobs)

    print("\n===========================================")
    print("ALL BATCH JOBS FINISHED")
    print("===========================================")