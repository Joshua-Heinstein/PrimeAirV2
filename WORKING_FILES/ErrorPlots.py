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
    base = os.path.basename(filename)
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

# Compute error differences
print("\n4. Computing errors...")
df["error_lat"] = df["Latitude_calc"] - df["Latitude_exif"]
df["error_lon"] = df["Longitude_calc"] - df["Longitude_exif"]
df["error_alt"] = df["Altitude_calc"] - df["Altitude_exif"]

print("\nError ranges:")
for col in ["error_lat", "error_lon", "error_alt"]:
    print(f"{col}: min={df[col].min():.6f}, max={df[col].max():.6f}")

# Convert latitude/longitude differences to meters
df["mean_lat"] = (df["Latitude_calc"] + df["Latitude_exif"]) / 2.0
df["error_lat_m"] = df["error_lat"] * 111320
df["error_lon_m"] = df["error_lon"] * 111320 * np.cos(np.deg2rad(df["mean_lat"]))
df["error_horizontal_m"] = np.sqrt(df["error_lat_m"]**2 + df["error_lon_m"]**2)

print("\n5. Creating plots...")

# --- A. Side-by-Side Comparison Plots for EXIF vs Calculated Values ---
# Create a figure with 3 rows (Latitude, Longitude, Altitude) and 2 columns (Expected vs Calculated)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))
metrics = [("Latitude", "Latitude_exif", "Latitude_calc"), 
           ("Longitude", "Longitude_exif", "Longitude_calc"),
           ("Altitude", "Altitude_exif", "Altitude_calc")]

for i, (metric, exif_col, calc_col) in enumerate(metrics):
    # Expected (EXIF) values
    axes[i, 0].scatter(df.index, df[exif_col], color='blue', alpha=0.7, label="EXIF")
    axes[i, 0].set_title(f"Expected {metric}")
    axes[i, 0].set_xlabel("Image Index")
    axes[i, 0].set_ylabel(metric)
    axes[i, 0].legend()
    
    # Calculated values
    axes[i, 1].scatter(df.index, df[calc_col], color='red', alpha=0.7, label="Calculated")
    axes[i, 1].set_title(f"Calculated {metric}")
    axes[i, 1].set_xlabel("Image Index")
    axes[i, 1].set_ylabel(metric)
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig("side_by_side_comparison.png")
plt.show()

# --- B. Comprehensive Error Distribution Plots ---
# Create subplots with histograms on the top row and corresponding box plots on the bottom row.
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
error_plots = [
    ("Latitude Error (m)", "error_lat_m"),
    ("Longitude Error (m)", "error_lon_m"),
    ("Horizontal Error (m)", "error_horizontal_m"),
    ("Altitude Error", "error_alt")
]

for i, (title, col) in enumerate(error_plots):
    # Histogram with KDE
    sns.histplot(df[col], kde=True, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f"{title} - Histogram")
    axes[0, i].set_xlabel(title)
    # Box Plot
    sns.boxplot(x=df[col], ax=axes[1, i], color='lightgreen')
    axes[1, i].set_title(f"{title} - Box Plot")
    axes[1, i].set_xlabel(title)

plt.tight_layout()
plt.savefig("error_histograms_boxplots.png")
plt.show()

# --- C. Error Propagation Plot ---
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["error_horizontal_m"], marker='o', label="Horizontal Error (m)")
plt.plot(df.index, df["error_alt"], marker='s', label="Altitude Error")
plt.xlabel("Image Index")
plt.ylabel("Error (m)")
plt.title("Error Propagation Across Images")
plt.legend()
plt.savefig("error_propagation.png")
plt.show()

# --- D. Correlation Heatmap of Error Metrics ---
plt.figure(figsize=(8, 6))
error_corr = df[["error_lat_m", "error_lon_m", "error_horizontal_m", "error_alt"]].corr()
sns.heatmap(error_corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Error Metrics")
plt.tight_layout()
plt.savefig("error_correlation_heatmap.png")
plt.show()

# Save the detailed error data
df.to_csv("detailed_position_errors.csv", index=False)
print("\nDetailed error data saved to detailed_position_errors.csv")
