## READ EXIF TGAS TO GET LLA
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd

def get_exif_data(image):
    """Return a dictionary from the exif data of a PIL Image item."""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def get_decimal_from_dms(dms, ref):
    """Convert degree, minute, second to decimal format."""
    # Convert IFDRational to a tuple (numerator, denominator)
    degrees = float(dms[0].numerator) / float(dms[0].denominator)
    minutes = float(dms[1].numerator) / float(dms[1].denominator) / 60.0
    seconds = float(dms[2].numerator) / float(dms[2].denominator) / 3600.0

    # Adjust for the hemisphere (latitude or longitude)
    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)

def get_lat_lon_alt_from_exif(exif_data):
    """Return the latitude, longitude, and altitude from exif data."""
    lat = None
    lon = None
    alt = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get("GPSLatitudeRef")
        gps_longitude = gps_info.get("GPSLongitude")
        gps_longitude_ref = gps_info.get("GPSLongitudeRef")
        gps_altitude = gps_info.get("GPSAltitude")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
            lon = get_decimal_from_dms(gps_longitude, gps_longitude_ref)

        if gps_altitude:
            alt = float(gps_altitude.numerator) / float(gps_altitude.denominator)

    return lat, lon, alt


# Path to the directory containing images
image_directory = image_directory = "/Users/reinabhatkuly/Desktop/drone video"  # Change this to your images directory
image_names = sorted(os.listdir(image_directory))

# List to hold image GPS data
image_gps_data = []

# Iterate over each image in the directory
for image_name in image_names:
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, image_name)

        image = Image.open(image_path)
        exif_data = get_exif_data(image)
        lat, lon, alt = get_lat_lon_alt_from_exif(exif_data)
        image_gps_data.append([image_name, lat, lon, alt])

# Convert to DataFrame and save to CSV
df = pd.DataFrame(image_gps_data, columns=["Image", "Latitude", "Longitude", "Altitude"])
csv_file = "image_gps_data.csv"  # Name of the output CSV file
df.to_csv(csv_file, index=False)

print(f"Data saved to {csv_file}")