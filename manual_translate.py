import pandas as pd
import os
import sys
from datetime import datetime

# --- Configuration ---
DATASET_PATH = "evaluated_translations_with_similarity.csv"
OUTPUT_FILE_PATH = "corrected_queries_WIP.csv" # Working file for corrections
COMPLETED_FILE_PATH = "corrected_queries_FINAL.csv" # Final output after all are done

# Define YOUR dataset's column names
ORIGINAL_ID_COL_CANDIDATE = 'utterance' # Let's assume 'utterance' is the best candidate from your data
ENGLISH_UTTERANCE_COL = 'utterance' # Original English text
MT_TAGALOG_COL = 'tagalog' # Your existing MT Tagalog column
SIMILARITY_SCORE_COL = 'similarity' # This column MUST be present

# This column will store the human-corrected Tagalog translation
HUMAN_TAGALOG_COL = 'human_corrected_tagalog' # NEW column for human corrections

# Similarity thresholds for categorization
CRITICAL_THRESHOLD = 0.50 # Below this, assume very broken MT
MEDIUM_THRESHOLD = 0.65 # Between CRITICAL and MEDIUM, assume needs significant work
OVERALL_REVIEW_THRESHOLD = 0.70 # Only review entries below this score

# --- Style Guide for Translator (Hardcoded, but could be loaded from a file) ---
STYLE_GUIDE = """
--- Tagalog Customer Support Style Guide ---
Target Tone: Conversational, empathetic, polite, approachable.
Avoid: Overly formal/literary Tagalog. Prefer common, everyday phrasing.
Politeness: Use 'po' and 'opo' appropriately (e.g., "Salamat po," "Opo, sir/ma'am").
Loanwords: Common English loanwords are acceptable when natural (e.g., 'refund', 'account', 'internet'). Avoid finding pure Tagalog equivalents if the loanword is more common in daily speech.
Clarity: Ensure translations are clear and easily understood by typical Filipino customers.
--------------------------------------------
"""

# --- Helper Functions ---
def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def save_progress(df, file_path=OUTPUT_FILE_PATH):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"\n[INFO] Progress saved to '{file_path}' at {datetime.now().strftime('%H:%M:%S')}.")
    except Exception as e:
        print(f"\n[ERROR] Failed to save progress: {e}")

def load_and_prepare_data(file_path=DATASET_PATH, output_file_path=OUTPUT_FILE_PATH):
    """Loads the dataset, attempts to load existing corrections, and prepares for review."""
    print(f"Loading original dataset from: {file_path}")
    if not os.path.exists(file_path):
        print(f"[ERROR] Dataset file not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)

    # --- IMPORTANT: Validate required columns ---
    required_cols = [ENGLISH_UTTERANCE_COL, MT_TAGALOG_COL, SIMILARITY_SCORE_COL]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: '{col}' in {file_path}. Please ensure your dataset has these columns after similarity scoring.")
            sys.exit(1)

    # Determine the actual ID column to use for merging/resuming
    current_id_col = ORIGINAL_ID_COL_CANDIDATE
    if df[ORIGINAL_ID_COL_CANDIDATE].duplicated().any():
        print(f"[WARNING] Duplicate entries found in '{ORIGINAL_ID_COL_CANDIDATE}' column. Adding a temporary unique ID for robust merging.")
        df['__temp_unique_id__'] = range(len(df))
        current_id_col = '__temp_unique_id__'
    else:
        print(f"[INFO] Using '{ORIGINAL_ID_COL_CANDIDATE}' as the unique ID column for merging.")

    # Initialize the human_corrected_tagalog column if it doesn't exist
    if HUMAN_TAGALOG_COL not in df.columns:
        df[HUMAN_TAGALOG_COL] = pd.NA

    # Attempt to load existing corrections from the output file
    if os.path.exists(output_file_path):
        print(f"Attempting to resume from previous session: {output_file_path}")
        try:
            df_wip = pd.read_csv(output_file_path)
            # Ensure the ID column is the index for combination
            df = df.set_index(current_id_col).combine_first(df_wip.set_index(current_id_col)).reset_index()
            print("[INFO] Resumed previous work successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load or merge '{output_file_path}': {e}. Starting fresh for review session.")
            if HUMAN_TAGALOG_COL not in df.columns:
                 df[HUMAN_TAGALOG_COL] = pd.NA

    # Filter to only include entries below the overall review threshold
    df_filtered = df[df[SIMILARITY_SCORE_COL] < OVERALL_REVIEW_THRESHOLD].copy()

    # Identify categories and assign 'category_type'
    df_filtered['category_type'] = 'Light Edit'
    df_filtered.loc[df_filtered[SIMILARITY_SCORE_COL] < MEDIUM_THRESHOLD, 'category_type'] = 'Medium Edit'
    df_filtered.loc[df_filtered[SIMILARITY_SCORE_COL] < CRITICAL_THRESHOLD, 'category_type'] = 'Heavy Edit'

    print(f"\nTotal entries for review (score < {OVERALL_REVIEW_THRESHOLD}): {len(df_filtered)}")
    print(f"  - Heavy Edit (<{CRITICAL_THRESHOLD}): {len(df_filtered[df_filtered['category_type'] == 'Heavy Edit'])}")
    print(f"  - Medium Edit (<{MEDIUM_THRESHOLD} & >={CRITICAL_THRESHOLD}): {len(df_filtered[df_filtered['category_type'] == 'Medium Edit'])}")
    print(f"  - Light Edit (<{OVERALL_REVIEW_THRESHOLD} & >={MEDIUM_THRESHOLD}): {len(df_filtered[df_filtered['category_type'] == 'Light Edit'])}")

    return df, df_filtered, current_id_col

def run_review_session(full_df, review_df_category, current_id_col, category_name):
    """Runs the interactive review session for a given category DataFrame."""
    reviewed_in_session = 0
    total_in_category = len(review_df_category)

    if total_in_category == 0:
        print(f"\nNo entries found in '{category_name}' category that require review.")
        return full_df # Return dataframe unchanged if nothing to do

    # Find where to start in this specific category
    start_index_in_category_df = 0
    untranslated_indices = review_df_category[review_df_category[HUMAN_TAGALOG_COL].isna()].index
    if not untranslated_indices.empty:
        first_untranslated_original_idx = untranslated_indices[0]
        try:
            start_index_in_category_df = review_df_category.index.get_loc(first_untranslated_original_idx)
            print(f"\nResuming '{category_name}' category from entry {start_index_in_category_df + 1}/{total_in_category}.")
        except KeyError:
            print(f"\nCould not find specific resume point for '{category_name}'. Starting from beginning of this category.")
            start_index_in_category_df = 0
    else:
        print(f"\nAll entries in '{category_name}' category have been reviewed.")
        return full_df # All done for this category

    # Sort this category for consistent progression (lowest sim score first)
    review_df_category = review_df_category.sort_values(by=SIMILARITY_SCORE_COL, ascending=True)

    # Main loop for review
    for i, (original_row_idx, entry) in enumerate(review_df_category.iloc[start_index_in_category_df:].iterrows()):
        current_item_num = start_index_in_category_df + i + 1

        clear_screen()
        print(f"--- Reviewing Entry {current_item_num}/{total_in_category} in '{category_name}' Category ---")
        print(f"ID: {entry[current_id_col]}")
        print(f"Similarity Score: {entry[SIMILARITY_SCORE_COL]:.2f}")
        print("\nOriginal English (Utterance):")
        print(f"  {entry[ENGLISH_UTTERANCE_COL]}")
        print("\nMachine Translation (MT Tagalog):")
        print(f"  {entry[MT_TAGALOG_COL]}")

        if pd.notna(entry[HUMAN_TAGALOG_COL]):
            print("\n(Previously Corrected Translation):")
            print(f"  {entry[HUMAN_TAGALOG_COL]}")

        corrected_tagalog = input("\nCorrected Tagalog (Enter to accept MT, 's' to skip, 'q' to quit & save): ").strip()

        if corrected_tagalog.lower() == 'q':
            save_progress(full_df)
            print("Session ended by user. Exiting.")
            sys.exit(0)
        elif corrected_tagalog.lower() == 's':
            print(f"Skipping entry {entry[current_id_col]}.")
            continue
        elif corrected_tagalog == "":
            final_translation = entry[MT_TAGALOG_COL] if pd.notna(entry[MT_TAGALOG_COL]) else ""
            print(f"Accepted MT as is for entry {entry[current_id_col]}.")
        else:
            final_translation = corrected_tagalog

        # Update the full DataFrame
        full_df.loc[full_df[current_id_col] == entry[current_id_col], HUMAN_TAGALOG_COL] = final_translation
        reviewed_in_session += 1

        if reviewed_in_session % 10 == 0:
            save_progress(full_df)

    print(f"\nAll entries in '{category_name}' category have been reviewed!")
    save_progress(full_df) # Final save for this category
    return full_df

def main():
    clear_screen()
    print("--- Tagalog Customer Support Intent Corrector ---")
    print(STYLE_GUIDE)

    full_df, review_df_all, current_id_col = load_and_prepare_data()

    if review_df_all.empty:
        print("\nNo entries below the similarity threshold for review. All good!")
        full_df.to_csv(COMPLETED_FILE_PATH, index=False, encoding='utf-8')
        sys.exit(0)

    while True:
        clear_screen()
        print("\n--- Choose a Category to Review ---")
        heavy_count = len(review_df_all[(review_df_all['category_type'] == 'Heavy Edit') & (review_df_all[HUMAN_TAGALOG_COL].isna())])
        medium_count = len(review_df_all[(review_df_all['category_type'] == 'Medium Edit') & (review_df_all[HUMAN_TAGALOG_COL].isna())])
        light_count = len(review_df_all[(review_df_all['category_type'] == 'Light Edit') & (review_df_all[HUMAN_TAGALOG_COL].isna())])
        total_remaining = heavy_count + medium_count + light_count

        print(f"1. Heavy Edit (score < {CRITICAL_THRESHOLD}) - Remaining: {heavy_count}")
        print(f"2. Medium Edit (score < {MEDIUM_THRESHOLD} & >={CRITICAL_THRESHOLD}) - Remaining: {medium_count}")
        print(f"3. Light Edit (score < {OVERALL_REVIEW_THRESHOLD} & >={MEDIUM_THRESHOLD}) - Remaining: {light_count}")
        print(f"\nTotal entries remaining to review: {total_remaining}")
        print("0. Quit & Save Final")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            current_category_df = review_df_all[review_df_all['category_type'] == 'Heavy Edit'].copy()
            full_df = run_review_session(full_df, current_category_df, current_id_col, "Heavy Edit")
        elif choice == '2':
            current_category_df = review_df_all[review_df_all['category_type'] == 'Medium Edit'].copy()
            full_df = run_review_session(full_df, current_category_df, current_id_col, "Medium Edit")
        elif choice == '3':
            current_category_df = review_df_all[review_df_all['category_type'] == 'Light Edit'].copy()
            full_df = run_review_session(full_df, current_category_df, current_id_col, "Light Edit")
        elif choice == '0':
            print("\nFinalizing and saving all corrections.")
            save_progress(full_df) # Save WIP one last time
            # Before final save, if we added a temporary ID column, remove it
            if current_id_col == '__temp_unique_id__' and '__temp_unique_id__' in full_df.columns:
                full_df = full_df.drop(columns=['__temp_unique_id__'])
                print("[INFO] Removed temporary unique ID column from final output.")
            full_df.to_csv(COMPLETED_FILE_PATH, index=False, encoding='utf-8')
            print(f"Final dataset (including corrections) saved to '{COMPLETED_FILE_PATH}'")
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

        # After a category session, refresh review_df_all to reflect latest counts
        # This is crucial so the menu updates with correct remaining counts
        full_df, review_df_all, _ = load_and_prepare_data(output_file_path=OUTPUT_FILE_PATH) # Reload with latest changes
        if review_df_all[review_df_all[HUMAN_TAGALOG_COL].isna()].empty:
             print("\nAll entries across all categories have been reviewed!")
             break # Exit the main loop if everything is done

    print("\nAll review categories are complete. Exiting program.")
    sys.exit(0)

if __name__ == "__main__":
    main()