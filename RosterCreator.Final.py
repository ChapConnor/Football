import pandas as pd
import os

# ==========================================
# 0. CONFIGURATION
# ==========================================
TRACKING_FILE = r'/Users/connorchapman/Downloads/2024_west_Practice_1.snappy.parquet'
PLAYER_LOOKUP_FILE = r'/Users/connorchapman/Downloads/shrine_bowl_players.parquet'


def generate_roster():
    print(f"--- GENERATING ROSTER (SORTED BY JERSEY) ---")

    if not os.path.exists(TRACKING_FILE):
        print(f"❌ Tracking file not found.")
        return

    # 1. Get IDs and Jersey Numbers from Tracking
    # Based on Diagnostic: Column is 'jersey_number'
    df_tracking = pd.read_parquet(TRACKING_FILE, columns=['gsis_id', 'jersey_number'])
    df_tracking['gsis_id'] = df_tracking['gsis_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Get unique pairs of ID and Jersey #
    df_roster = df_tracking.drop_duplicates(subset=['gsis_id']).copy()

    # 2. Process Name Lookup File
    if os.path.exists(PLAYER_LOOKUP_FILE):
        df_lookup = pd.read_parquet(PLAYER_LOOKUP_FILE)

        # Standardize ID for merging
        df_lookup['gsis_player_id'] = df_lookup['gsis_player_id'].astype(str).str.replace(r'\.0$', '', regex=True)

        # Join names based on your diagnostic columns
        if 'first_name' in df_lookup.columns and 'last_name' in df_lookup.columns:
            df_lookup['full_name'] = df_lookup['first_name'].astype(str) + " " + df_lookup['last_name'].astype(str)

        cols_to_pull = ['gsis_player_id', 'full_name']
        df_sub = df_lookup[cols_to_pull].drop_duplicates(subset=['gsis_player_id'])

        # Merge tracking IDs with Name lookup
        df_roster = pd.merge(df_roster, df_sub, left_on='gsis_id', right_on='gsis_player_id', how='left')

    # 3. Final Column Cleanup
    if 'full_name' not in df_roster.columns:
        df_roster['full_name'] = ""

    df_roster['position'] = ""  # Kept empty for manual WR/DB tagging
    df_roster['full_name'] = df_roster['full_name'].fillna("")

    # 4. SORTING LOGIC
    # Convert jersey_number to numeric for correct sorting (1, 2, 10 instead of 1, 10, 2)
    df_roster['jersey_number'] = pd.to_numeric(df_roster['jersey_number'], errors='coerce')
    df_roster = df_roster.sort_values(by='jersey_number', ascending=True)

    # Convert back to string for the CSV and handle empty values
    df_roster['jersey_number'] = df_roster['jersey_number'].fillna("").astype(str).str.replace(r'\.0$', '', regex=True)

    # 5. Save Output (roster_[PracticeName].csv)
    clean_base = os.path.basename(TRACKING_FILE).replace('.snappy', '').replace('.parquet', '')
    output_name = f"roster_{clean_base}.csv"
    output_path = os.path.join(os.path.dirname(TRACKING_FILE), output_name)

    # Reorder columns: ID, Name, Jersey #, Position
    final_cols = ['gsis_id', 'full_name', 'jersey_number', 'position']
    df_roster = df_roster[final_cols]

    df_roster.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"✅ SUCCESS: Created {output_name}")
    print(f"The roster is now sorted numerically by Jersey Number.")
    print("-" * 30)


if __name__ == "__main__":
    generate_roster()