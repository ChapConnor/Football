import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial.distance import pdist, squareform
import os
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION
# ==========================================

# --- INSTRUCTIONS ---
# Add as many file pairings as you want to the list below.
JOBS_LIST = [
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_west_Practice_3.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_3.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_west_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_2.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_west_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_West_Practice_1.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_east_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_1.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_east_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_2.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2024_east_Practice_3.csv',
        "tracking": r'/Users/connorchapman/Downloads/2024_east_Practice_3.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2022_east_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_east_Practice_1.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2022_west_Practice_2.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_west_Practice_2.snappy.parquet'
    },
    {
        "roster": r'/Users/connorchapman/Downloads/roster_2022_west_Practice_1.csv',
        "tracking": r'/Users/connorchapman/Downloads/2022_west_Practice_1.snappy.parquet'
    },
]

OUTPUT_BASE_NAME = 'detected_matchups_v23_VECTOR_FIX'

# --- POSITIONS ---
OFFENSE_POS = {'WR'}
DEFENSE_POS = {'CB', 'SS', 'FS', 'S', 'DB'}

# --- v23 "VECTOR" FILTERS ---
# 1. Path Similarity (Directional Vector)
MIN_PATH_SIMILARITY = 0.60

# 2. Path Length Ratio (Distance)
MAX_DB_WR_DISP_RATIO = 1.20
MIN_DB_WR_DISP_RATIO = 0.40

# 3. Alignment Limits
MAX_START_SEPARATION = 6.0
MAX_LATERAL_ALIGNMENT = 3.5
MAX_DEPTH_ALIGNMENT = 5.5

# 4. Standard filters
WR_MAX_START_SPEED = 1.5
WR_MIN_DOWNFIELD_DIST = 5.0
MAX_SPEED_CAP = 10.5
FIELD_Y_MIN = 12.0
FIELD_Y_MAX = 41.3

# Settings
SEARCH_WINDOW_SEC = 3.5
COOLDOWN_SEC = 5.0


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_path_similarity(x1_start, y1_start, x1_end, y1_end,
                              x2_start, y2_start, x2_end, y2_end):
    """
    Calculates the cosine similarity of two displacement vectors.
    """
    v1_x = x1_end - x1_start
    v1_y = y1_end - y1_start
    v2_x = x2_end - x2_start
    v2_y = y2_end - y2_start

    dot_product = (v1_x * v2_x) + (v1_y * v2_y)
    mag1 = np.sqrt(v1_x ** 2 + v1_y ** 2)
    mag2 = np.sqrt(v2_x ** 2 + v2_y ** 2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def process_matchup_job(roster_file, tracking_file, job_index, total_jobs):
    print(f"\n--- JOB {job_index}/{total_jobs} ---")
    print(f"Roster: {os.path.basename(roster_file)}")
    print(f"Tracking: {os.path.basename(tracking_file)}")

    # Setup Output Filename
    try:
        base_filename = os.path.basename(roster_file)
        name_only = os.path.splitext(base_filename)[0]
        # Try to keep the suffix (e.g. "_Practice_3") if present
        suffix = name_only.split('roster')[-1] if 'roster' in name_only.lower() else ""
        output_filename = f"{OUTPUT_BASE_NAME}{suffix}.csv"
        DOWNLOADS_FOLDER = os.path.dirname(roster_file)
        OUTPUT_FILE = os.path.join(DOWNLOADS_FOLDER, output_filename)
    except Exception:
        OUTPUT_FILE = f"detected_matchups_job_{job_index}.csv"

    # Load Data
    print("Loading datasets...")
    try:
        df_players = pd.read_csv(roster_file)
        df_players['gsis_id'] = df_players['gsis_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        player_positions = dict(zip(df_players['gsis_id'], df_players['position']))

        possible_name_cols = ['full_name', 'displayName', 'name', 'Player']
        name_col = next((col for col in possible_name_cols if col in df_players.columns), None)
        player_names = dict(zip(df_players['gsis_id'], df_players[name_col])) if name_col else {}

        df = pd.read_parquet(tracking_file)
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading files: {e}")
        return

    # Prepare Data
    print("Sorting and pivoting tracking data...")
    df = df.sort_values('ts')
    df['ts'] = pd.to_datetime(df['ts'])
    df['gsis_id'] = df['gsis_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df = df.dropna(subset=['gsis_id'])
    df = df.drop_duplicates(subset=['ts', 'gsis_id'])

    pivot_x = df.pivot(index='ts', columns='gsis_id', values='x')
    pivot_y = df.pivot(index='ts', columns='gsis_id', values='y')
    pivot_s = df.pivot(index='ts', columns='gsis_id', values='s')

    # Scan Setup
    timestamps = pivot_x.index
    step_size = 5

    if len(timestamps) > 1:
        avg_delta = (timestamps[1] - timestamps[0]).total_seconds()
        if avg_delta == 0: avg_delta = 0.1
    else:
        avg_delta = 0.1

    cooldown_frames = int(COOLDOWN_SEC / avg_delta)
    search_limit = len(timestamps) - int(SEARCH_WINDOW_SEC / avg_delta)

    candidates_list = []
    i = 0

    print("Scanning with Vector Direction Filter...")

    # --- TQDM PROGRESS BAR ---
    pbar = tqdm(total=search_limit, desc="Scanning Timeline", unit="frame")

    while i < search_limit:
        current_step = 0  # Track how much we advance this iteration

        t_start = timestamps[i]

        # Skip idle frames (high avg speed usually means play is over or chaos)
        ss = pivot_s.iloc[i].values
        if np.nanmean(ss) > 4.0:
            jump = step_size
            i += jump
            pbar.update(jump)
            continue

        # Get valid players
        xs = pivot_x.iloc[i].values
        ys = pivot_y.iloc[i].values
        coords = np.column_stack((xs, ys))

        valid_mask = ~np.isnan(coords).any(axis=1)
        if valid_mask.sum() < 2:
            jump = step_size
            i += jump
            pbar.update(jump)
            continue

        valid_coords = coords[valid_mask]
        valid_ids = pivot_x.columns[valid_mask]

        # Loose Search (Distance Matrix)
        dists = squareform(pdist(valid_coords))
        loose_mask = (dists < 8.0)
        potential_pairs = np.argwhere(loose_mask)

        frame_candidates = []

        for p1_idx, p2_idx in potential_pairs:
            if p1_idx >= p2_idx: continue

            id1 = valid_ids[p1_idx]
            id2 = valid_ids[p2_idx]

            pos1 = player_positions.get(id1, 'Unknown')
            pos2 = player_positions.get(id2, 'Unknown')

            # Position Match
            is_wr_db = (pos1 in OFFENSE_POS and pos2 in DEFENSE_POS) or \
                       (pos2 in OFFENSE_POS and pos1 in DEFENSE_POS)
            if not is_wr_db: continue

            if pos1 in OFFENSE_POS:
                wr_id, wr_pos = id1, pos1
                db_id, db_pos = id2, pos2
                wr_idx = p1_idx
                db_idx = p2_idx
            else:
                wr_id, wr_pos = id2, pos2
                db_id, db_pos = id1, pos1
                wr_idx = p2_idx
                db_idx = p1_idx

            # Alignment Checks
            wr_x, wr_y = valid_coords[wr_idx]
            db_x, db_y = valid_coords[db_idx]

            delta_x = abs(wr_x - db_x)
            delta_y = abs(wr_y - db_y)
            start_sep = np.sqrt(delta_x ** 2 + delta_y ** 2)

            if start_sep > MAX_START_SEPARATION: continue
            if delta_y > MAX_LATERAL_ALIGNMENT: continue
            if delta_x > MAX_DEPTH_ALIGNMENT: continue

            if ss[valid_mask][wr_idx] > WR_MAX_START_SPEED: continue

            # Dynamics & Journey Check
            t_future = t_start + timedelta(seconds=SEARCH_WINDOW_SEC)

            # Slice future data
            fx = pivot_x.loc[t_start:t_future, [wr_id, db_id]].copy().interpolate(limit=5).bfill().ffill()
            fy = pivot_y.loc[t_start:t_future, [wr_id, db_id]].copy().interpolate(limit=5).bfill().ffill()
            fs = pivot_s.loc[t_start:t_future, [wr_id, db_id]].copy().interpolate(limit=5).bfill().ffill()

            if fx.isnull().values.any(): continue

            # 1. Distance Ratio
            wr_disp = np.sqrt(
                (fx[wr_id].iloc[-1] - fx[wr_id].iloc[0]) ** 2 + (fy[wr_id].iloc[-1] - fy[wr_id].iloc[0]) ** 2)
            db_disp = np.sqrt(
                (fx[db_id].iloc[-1] - fx[db_id].iloc[0]) ** 2 + (fy[db_id].iloc[-1] - fy[db_id].iloc[0]) ** 2)

            if wr_disp < WR_MIN_DOWNFIELD_DIST: continue

            ratio = db_disp / wr_disp
            if ratio < MIN_DB_WR_DISP_RATIO: continue
            if ratio > MAX_DB_WR_DISP_RATIO: continue

            # 2. Path Similarity
            path_sim = calculate_path_similarity(
                fx[wr_id].iloc[0], fy[wr_id].iloc[0], fx[wr_id].iloc[-1], fy[wr_id].iloc[-1],
                fx[db_id].iloc[0], fy[db_id].iloc[0], fx[db_id].iloc[-1], fy[db_id].iloc[-1]
            )

            if path_sim < MIN_PATH_SIMILARITY: continue

            # On Field Check
            on_field = (FIELD_Y_MIN < wr_y < FIELD_Y_MAX)
            peak_speed = fs.max().max()
            if peak_speed > MAX_SPEED_CAP: continue

            wr_name = player_names.get(wr_id, wr_id)
            db_name = player_names.get(db_id, db_id)

            frame_candidates.append({
                'rep_start': t_start,  # Changed from 'timestamp' to 'rep_start' to match other script
                'WR_Name': wr_name, 'DB_Name': db_name,
                'WR_ID': wr_id, 'DB_ID': db_id,
                'Start_Sep': round(start_sep, 2),
                'Path_Sim': round(path_sim, 2),
                'Disp_Ratio': round(ratio, 2),
                'WR_Disp': round(wr_disp, 2),
                'DB_Disp': round(db_disp, 2),
                'On_Field': on_field
            })

        # Process Results for this Frame
        if frame_candidates:
            # Sort by best WR movement and dedupe so one WR isn't matched to 2 DBs in same frame
            frame_candidates.sort(key=lambda x: x['WR_Disp'], reverse=True)
            unique_matches = []
            used_ids = set()
            for m in frame_candidates:
                if m['WR_ID'] not in used_ids and m['DB_ID'] not in used_ids:
                    unique_matches.append(m)
                    used_ids.add(m['WR_ID'])
                    used_ids.add(m['DB_ID'])

            candidates_list.extend(unique_matches)
            jump = cooldown_frames
        else:
            jump = step_size

        i += jump
        pbar.update(jump)

    pbar.close()

    # Export
    if candidates_list:
        df_out = pd.DataFrame(candidates_list)
        df_clean = df_out[df_out['On_Field'] == True]
        print(f"--> Found {len(df_clean)} valid reps.")
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print(f"--> Saved to: {OUTPUT_FILE}")
    else:
        print("--> No candidates found in this file.")


# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    total_jobs = len(JOBS_LIST)
    print(f"--- MATCHUP DETECTION BATCH INITIALIZED: Found {total_jobs} pairings ---")

    for i, job in enumerate(JOBS_LIST):
        roster_path = job.get('roster')
        tracking_path = job.get('tracking')

        if not roster_path or not tracking_path:
            print(f"Skipping Job {i + 1}: Missing file path.")
            continue

        if not os.path.exists(roster_path):
            print(f"Skipping Job {i + 1}: Roster file not found ({roster_path})")
            continue

        if not os.path.exists(tracking_path):
            print(f"Skipping Job {i + 1}: Parquet file not found ({tracking_path})")
            continue

        process_matchup_job(roster_path, tracking_path, i + 1, total_jobs)

    print("\n===========================================")
    print("ALL JOBS FINISHED")
    print("===========================================")