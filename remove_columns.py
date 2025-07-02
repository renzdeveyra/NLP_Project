import pandas as pd

# Define input and output file names
input_csv_file = 'updated_corrected_queries_WIP.csv'
output_csv_file = 'translated_dataset_tagalog2.csv'

# Columns to remove
columns_to_remove = ['needs_review', 'similarity', 'tagalog']

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_file)

    print(f"Original columns: {df.columns.tolist()}")

    # Remove the specified columns
    # axis=1 indicates that we are dropping columns (axis=0 would be rows)
    # inplace=True modifies the DataFrame directly without needing to reassign
    df.drop(columns=columns_to_remove, inplace=True)

    print(f"Columns after removal: {df.columns.tolist()}")

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(output_csv_file, index=False) # index=False prevents writing the DataFrame index as a column

    print(f"Columns '{columns_to_remove}' removed successfully. New CSV saved to '{output_csv_file}'")

except FileNotFoundError:
    print(f"Error: The file '{input_csv_file}' was not found.")
except KeyError as e:
    print(f"Error: One or more columns to remove were not found in the CSV: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")