import pandas as pd
import random

def add_random_rows(file1_path, file2_path, output_path, fraction=0.1, seed=42):
    """
    Reads FILE1, samples a fraction of rows, appends them to FILE2, and saves to output_path.
    """
    # Load data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Random selection
    n_rows = len(df1)
    n_to_add = int(n_rows * fraction)
    random.seed(seed)
    indices = random.sample(range(n_rows), n_to_add)
    df1_sample = df1.iloc[indices]

    # Append and save
    df_combined = pd.concat([df2, df1_sample], ignore_index=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Added {n_to_add} rows from '{file1_path}' to '{file2_path}'. Saved to '{output_path}'.")

if __name__ == "__main__":
    # File paths - edit accordingly
    file1 = "raw_calls_file1_dirty.csv"        # Source file path (rows to randomly add)
    file2 = "raw_calls_file1.csv"        # Target file path
    output_file = "raw_calls_file1_with_outliers.csv"  # Output file path

    # Fraction of rows from file1 to add to file2 (e.g. 0.2 = 20%)
    fraction_to_add = 1

    add_random_rows(file1, file2, output_file, fraction=fraction_to_add)
