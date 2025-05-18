import pandas as pd
import os

# Assuming all your extracted CSV files are in the current directory
# or you can specify the path to the directory containing them
directory = "../processed/mpi_roof"  # Change this to the actual directory if needed

all_files = [f for f in os.listdir(directory) if f.startswith("mpi_roof_") and f.endswith(".csv")]

if not all_files:
    print("No 'mpi_roof_' CSV files found in the specified directory.")
else:
    all_data = []
    for filename in all_files:
        try:
            df = pd.read_csv(os.path.join(directory, filename), encoding='latin-1')
            all_data.append(df)
            print(f"Successfully read: {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if all_data:
        # Concatenate all DataFrames in the list
        merged_df = pd.concat(all_data, ignore_index=True)

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv("../processed/mpi_roof.csv", index=False)
        print("Successfully merged all CSV files into 'mpi_roof.csv'")
    else:
        print("No data was successfully read from the CSV files.")