import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up plot style
sns.set(style="whitegrid")

# Filenames for the CSV files
calc_csv = "drone_positions.csv"
exif_csv = "image_gps_data.csv"

def extract_common_name(filename):
    """Extract the last part of the filename before the extension."""
    # Get the base filename without path
    base = os.path.basename(filename)
    # Split by common delimiters and take the last part
    parts = base.replace('_', ' ').replace('-', ' ').split()
    return parts[-1]

print("\nDebug Info:")
print("1. Loading CSV files...")

# Read CSV files with error handling
try:
    df_calc = pd.read_csv(calc_csv)
    # Add new column with extracted common names
    df_calc['common_name'] = df_calc['Image'].apply(extract_common_name)
    print(f"Successfully loaded {calc_csv}")
    print(f"Shape: {df_calc.shape}")
    print("First few rows of calculated positions with common names:")
    print(df_calc[['Image', 'common_name']].head())
except Exception as e:
    print(f"Error loading {calc_csv}: {str(e)}")
    exit(1)

try:
    df_exif = pd.read_csv(exif_csv)
    # Add new column with extracted common names
    df_exif['common_name'] = df_exif['Image'].apply(extract_common_name)
    print(f"\nSuccessfully loaded {exif_csv}")
    print(f"Shape: {df_exif.shape}")
    print("First few rows of EXIF data with common names:")
    print(df_exif[['Image', 'common_name']].head())
except Exception as e:
    print(f"Error loading {exif_csv}: {str(e)}")
    exit(1)

# Check for common column names before merge
print("\n2. Checking column names:")
print(f"Calculated positions columns: {df_calc.columns.tolist()}")
print(f"EXIF data columns: {df_exif.columns.tolist()}")

# Merge on common_name instead of Image
print("\n3. Merging dataframes on common names...")
df = pd.merge(df_calc, df_exif, on="common_name", suffixes=("_calc", "_exif"))
print(f"Merged dataframe shape: {df.shape}")
if df.empty:
    print("WARNING: Merged dataframe is empty! Check if common names match between files.")
    print("\nUnique values in calc_csv 'common_name' column:", df_calc['common_name'].unique())
    print("\nUnique values in exif_csv 'common_name' column:", df_exif['common_name'].unique())
    exit(1)

# Check for required columns after merge
required_cols = [
    "Latitude_calc", "Latitude_exif",
    "Longitude_calc", "Longitude_exif",
    "Altitude_calc", "Altitude_exif"
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\nWARNING: Missing required columns: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    exit(1)

# Check for null values
null_counts = df[required_cols].isnull().sum()
if null_counts.any():
    print("\nWARNING: Found null values in required columns:")
    print(null_counts[null_counts > 0])

# Compute error differences with validation
print("\n4. Computing errors...")
df["error_lat"] = df["Latitude_calc"] - df["Latitude_exif"]
df["error_lon"] = df["Longitude_calc"] - df["Longitude_exif"]
df["error_alt"] = df["Altitude_calc"] - df["Altitude_exif"]

# Print error ranges to verify calculations
print("\nError ranges:")
for col in ["error_lat", "error_lon", "error_alt"]:
    print(f"{col}: min={df[col].min():.6f}, max={df[col].max():.6f}")

# Convert to meters with validation
df["mean_lat"] = (df["Latitude_calc"] + df["Latitude_exif"]) / 2.0
df["error_lat_m"] = df["error_lat"] * 111320
df["error_lon_m"] = df["error_lon"] * 111320 * np.cos(np.deg2rad(df["mean_lat"]))
df["error_horizontal_m"] = np.sqrt(df["error_lat_m"]**2 + df["error_lon_m"]**2)

print("\n5. Creating plots...")

# 1. Scatter Plots with data range validation
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Latitude
ax[0].scatter(df["Latitude_exif"], df["Latitude_calc"], color='blue', alpha=0.7)
min_lat = min(df["Latitude_exif"].min(), df["Latitude_calc"].min())
max_lat = max(df["Latitude_exif"].max(), df["Latitude_calc"].max())
if min_lat != max_lat:  # Prevent plotting identical lines
    ax[0].plot([min_lat, max_lat], [min_lat, max_lat], 'r--')
ax[0].set_xlabel("EXIF Latitude")
ax[0].set_ylabel("Calculated Latitude")
ax[0].set_title("Latitude Comparison")

# Longitude
ax[1].scatter(df["Longitude_exif"], df["Longitude_calc"], color='green', alpha=0.7)
min_lon = min(df["Longitude_exif"].min(), df["Longitude_calc"].min())
max_lon = max(df["Longitude_exif"].max(), df["Longitude_calc"].max())
if min_lon != max_lon:
    ax[1].plot([min_lon, max_lon], [min_lon, max_lon], 'r--')
ax[1].set_xlabel("EXIF Longitude")
ax[1].set_ylabel("Calculated Longitude")
ax[1].set_title("Longitude Comparison")

# Altitude
ax[2].scatter(df["Altitude_exif"], df["Altitude_calc"], color='purple', alpha=0.7)
min_alt = min(df["Altitude_exif"].min(), df["Altitude_calc"].min())
max_alt = max(df["Altitude_exif"].max(), df["Altitude_calc"].max())
if min_alt != max_alt:
    ax[2].plot([min_alt, max_alt], [min_alt, max_alt], 'r--')
ax[2].set_xlabel("EXIF Altitude")
ax[2].set_ylabel("Calculated Altitude")
ax[2].set_title("Altitude Comparison")

plt.tight_layout()
plt.savefig("comparison_scatter_plots.png")
plt.show()

# 2. Histograms of Error Distributions
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

sns.histplot(df["error_lat_m"], kde=True, ax=ax[0], color='blue')
ax[0].set_title("Latitude Error (m)")

sns.histplot(df["error_lon_m"], kde=True, ax=ax[1], color='green')
ax[1].set_title("Longitude Error (m)")

sns.histplot(df["error_horizontal_m"], kde=True, ax=ax[2], color='orange')
ax[2].set_title("Horizontal Error (m)")

sns.histplot(df["error_alt"], kde=True, ax=ax[3], color='purple')
ax[3].set_title("Altitude Error (m)")

plt.tight_layout()
plt.savefig("error_histograms.png")
plt.show()

# 3. Error Propagation Plot
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["error_horizontal_m"], marker='o', label="Horizontal Error (m)")
plt.plot(df.index, df["error_alt"], marker='s', label="Altitude Error (m)")
plt.xlabel("Image Index")
plt.ylabel("Error (m)")
plt.title("Error Propagation Across Images")
plt.legend()
plt.savefig("error_propagation.png")
plt.show()

# Save the detailed error data
df.to_csv("detailed_position_errors.csv", index=False)
print("\nDetailed error data saved to detailed_position_errors.csv")