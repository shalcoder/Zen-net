import os
import pandas as pd
import glob

# Define the root directory
root_dir = r"c:\Users\VISHAL\Downloads\WEDA-FALL-main\dataset\50Hz"
output_file = r"c:\Users\VISHAL\Downloads\WEDA-FALL-main\dataset\merged_sensor_data.csv"

all_data = []
processed_count = 0

print(f"Scanning directory: {root_dir}")

# Walk through all subdirectories
for subdir, dirs, files in os.walk(root_dir):
    # Get the folder name to use as label (e.g., D01, F01)
    folder_name = os.path.basename(subdir)
    
    # Identify accel files
    accel_files = [f for f in files if f.endswith('_accel.csv')]
    
    for acc_file in accel_files:
        # Construct the matching gyro filename
        # Assumption: U01_R01_accel.csv matches U01_R01_gyro.csv
        prefix = acc_file.replace('_accel.csv', '')
        gyro_file = prefix + '_gyro.csv'
        
        if gyro_file in files:
            acc_path = os.path.join(subdir, acc_file)
            gyro_path = os.path.join(subdir, gyro_file)
            
            try:
                # Read the CSV files
                df_acc = pd.read_csv(acc_path)
                df_gyro = pd.read_csv(gyro_path)
                
                # Ensure they have data
                if df_acc.empty or df_gyro.empty:
                    print(f"Skipping empty file: {prefix}")
                    continue

                # Truncate to the minimum length (in case of slight mismatch)
                min_len = min(len(df_acc), len(df_gyro))
                df_acc = df_acc.iloc[:min_len]
                df_gyro = df_gyro.iloc[:min_len]
                
                # Check column structure (sanity check)
                # We expect cols 1,2,3 to be X, Y, Z (skipping col 0 which is time)
                if len(df_acc.columns) < 4 or len(df_gyro.columns) < 4:
                    print(f"Skipping malformed file: {prefix}")
                    continue

                # Create the merged dataframe
                merged_df = pd.DataFrame({
                    'acc_x': df_acc.iloc[:, 1],
                    'acc_y': df_acc.iloc[:, 2],
                    'acc_z': df_acc.iloc[:, 3],
                    'gyro_x': df_gyro.iloc[:, 1],
                    'gyro_y': df_gyro.iloc[:, 2],
                    'gyro_z': df_gyro.iloc[:, 3],
                    'label': folder_name
                })
                
                all_data.append(merged_df)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing {prefix}: {e}")

# Concatenate all data
if all_data:
    print(f"Concatenating {len(all_data)} entries...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Success! Merged data saved to: {output_file}")
    print(f"Total rows: {len(final_df)}")
    print("Head of new file:")
    print(final_df.head())
else:
    print("No matching accel/gyro pair files found.")
