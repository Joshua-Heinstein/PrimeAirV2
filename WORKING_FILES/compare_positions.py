"""
compare_positions.py

This script compares the drone global position calculated from image processing (drone_positions.csv)
with the actual image GPS data extracted from EXIF (image_gps_data.csv).

It performs the following:
  1. Reads both CSV files and merges them on the image filename.
  2. Computes error differences for latitude, longitude, and altitude.
  3. Converts latitude/longitude errors to meters and computes horizontal error.
  4. Generates summary statistics (mean, std) and saves them to a CSV.
  5. Creates plots:
     - Scatter plots comparing calculated vs. EXIF positions.
     - Histograms of error distributions.
     - A plot of error propagation across the image set.
  6. Saves the detailed error data to a CSV file.

Adjust the CSV filenames and paths as needed.
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Set up plot style
sns.set(style="whitegrid")

# Filenames for the CSV files (adjust as needed)
calc_csv = "drone_positions.csv"
exif_csv = "image_gps_data.csv"

# Read CSV files
df_calc = pd.read_csv(calc_csv)
df_exif = pd.read_csv(exif_csv)

# Merge the two dataframes on the "Image" column.
# This will add suffixes to columns that appear in both.
df = pd.merge(df_calc, df_exif, on="Image", suffixes=("_calc", "_exif"))

# Compute error differences between calculated and EXIF values.
df["error_lat"] = df["Latitude_calc"] - df["Latitude_exif"]
df["error_lon"] = df["Longitude_calc"] - df["Longitude_exif"]
df["error_alt"] = df["Altitude_calc"] - df["Altitude_exif"]

# To compute horizontal error in meters, convert degree differences to meters.
# Approximate conversion: 1 deg latitude ≈ 111320 m; longitude conversion depends on latitude.
df["mean_lat"] = (df["Latitude_calc"] + df["Latitude_exif"]) / 2.0
df["error_lat_m"] = df["error_lat"] * 111320
df["error_lon_m"] = df["error_lon"] * 111320 * np.cos(np.deg2rad(df["mean_lat"]))
df["error_horizontal_m"] = np.sqrt(df["error_lat_m"]**2 + df["error_lon_m"]**2)

# Compute summary statistics.
summary_stats = {
    "error_lat_mean": df["error_lat"].mean(),
    "error_lat_std": df["error_lat"].std(),
    "error_lon_mean": df["error_lon"].mean(),
    "error_lon_std": df["error_lon"].std(),
    "error_alt_mean": df["error_alt"].mean(),
    "error_alt_std": df["error_alt"].std(),
    "error_horizontal_mean_m": df["error_horizontal_m"].mean(),
    "error_horizontal_std_m": df["error_horizontal_m"].std()
}

print("Summary Statistics:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")

# Save summary statistics to CSV.
summary_df = pd.DataFrame(list(summary_stats.items()), columns=["Metric", "Value"])
summary_df.to_csv("position_error_summary.csv", index=False)
print("Summary statistics saved to position_error_summary.csv")

# -------------------------
# Plotting
# -------------------------

# 1. Scatter Plots: Calculated vs. EXIF for Latitude, Longitude, and Altitude.
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Latitude
ax[0].scatter(df["Latitude_exif"], df["Latitude_calc"], color='blue', alpha=0.7)
min_lat = min(df["Latitude_exif"].min(), df["Latitude_calc"].min())
max_lat = max(df["Latitude_exif"].max(), df["Latitude_calc"].max())
ax[0].plot([min_lat, max_lat], [min_lat, max_lat], 'r--')
ax[0].set_xlabel("EXIF Latitude")
ax[0].set_ylabel("Calculated Latitude")
ax[0].set_title("Latitude Comparison")

# Longitude
ax[1].scatter(df["Longitude_exif"], df["Longitude_calc"], color='green', alpha=0.7)
min_lon = min(df["Longitude_exif"].min(), df["Longitude_calc"].min())
max_lon = max(df["Longitude_exif"].max(), df["Longitude_calc"].max())
ax[1].plot([min_lon, max_lon], [min_lon, max_lon], 'r--')
ax[1].set_xlabel("EXIF Longitude")
ax[1].set_ylabel("Calculated Longitude")
ax[1].set_title("Longitude Comparison")

# Altitude
ax[2].scatter(df["Altitude_exif"], df["Altitude_calc"], color='purple', alpha=0.7)
min_alt = min(df["Altitude_exif"].min(), df["Altitude_calc"].min())
max_alt = max(df["Altitude_exif"].max(), df["Altitude_calc"].max())
ax[2].plot([min_alt, max_alt], [min_alt, max_alt], 'r--')
ax[2].set_xlabel("EXIF Altitude")
ax[2].set_ylabel("Calculated Altitude")
ax[2].set_title("Altitude Comparison")

plt.tight_layout()
plt.savefig("comparison_scatter_plots.png")
plt.show()

# 2. Histograms of Error Distributions.
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

# 3. Error Propagation Plot: Horizontal and Altitude Errors vs. Image Index.
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["error_horizontal_m"], marker='o', label="Horizontal Error (m)")
plt.plot(df.index, df["error_alt"], marker='s', label="Altitude Error (m)")
plt.xlabel("Image Index")
plt.ylabel("Error (m)")
plt.title("Error Propagation Across Images")
plt.legend()
plt.savefig("error_propagation.png")
plt.show()

# Save the detailed error data for further analysis.
df.to_csv("detailed_position_errors.csv", index=False)
print("Detailed error data saved to detailed_position_errors.csv")
