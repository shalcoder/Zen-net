import pandas as pd

# File paths
input_file = r"c:\Users\VISHAL\Downloads\WEDA-FALL-main\dataset\merged_sensor_data.csv"
output_file = r"c:\Users\VISHAL\Downloads\WEDA-FALL-main\dataset\merged_sensor_data_labeled.csv"

# Load the dataset
print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

# Define the mapping based on the README descriptions
# F01-F08: Falls -> 'Fall'
# D01: Walking -> 'Walking'
# D04, D05: Sitting related ADLs -> 'Sitting'
# Rest (D02, D03, D06-D11): -> 'Normal'

# Create a function or dictionary for mapping
def map_label(code):
    # Ensure code is a string and stripped
    code = str(code).strip()
    
    if code.startswith('F'):
        return 'Fall'
    elif code == 'D01':
        return 'Walking'
    elif code in ['D04', 'D05']:
        return 'Sitting'
    else:
        # Includes D02 (Jogging), D03 (Stairs), D06 (Crouching), D07 (Stumble), 
        # D08 (Reach), D09 (Hit table), D10 (Clap), D11 (Door)
        return 'Normal'

# Apply the mapping
print("Mapping labels...")
df['label_mapped'] = df['label'].apply(map_label)

# Check the distribution
print("\nNew Label Distribution:")
print(df['label_mapped'].value_counts())

# Drop the old label column and rename the new one
df = df.drop(columns=['label'])
df = df.rename(columns={'label_mapped': 'label'})

# Save the new dataset
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)
print("Done.")
