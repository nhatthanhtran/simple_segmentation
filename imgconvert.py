from PIL import Image
import pillow_heif
import os

data_path = "./data/"
pillow_heif.register_heif_opener()
# Convert HEIC to JPEG
for file in os.listdir(data_path):
    if file.endswith(".HEIC"):
        img = Image.open(f"{data_path}{file}")
        img.save(f"{data_path}{file.replace('.HEIC', '.jpg')}")

print(f"Converted HEIC images to JPEG in {data_path}")