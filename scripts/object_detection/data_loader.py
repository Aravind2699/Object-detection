from ultralytics import RTDETR
import os

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Input file path (change as needed)
input_path = "ML_product/data/raw/pumpkin-harvesting-field-different-types-pumpkin.jpg"  # or .mp4
output_dir = "ML_product/data/processed/"
os.makedirs(output_dir, exist_ok=True)

# Determine if input is image or video
ext = os.path.splitext(input_path)[1].lower()

if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
    # Image prediction
    results = model(input_path)
    # Loop through results and save each image
    for i, result in enumerate(results):
        output_file = os.path.join(output_dir, f"output_{i+1}.jpg")
        result.save(filename=output_file)
        print(f"Saved output image to {output_file}")

elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
    # Video prediction
    results = model(input_path)
    # Save output video (ultralytics saves to results.save_dir by default)
    results.save()
    # Move the output video to output_dir if needed
    for file in os.listdir(results.save_dir):
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            src = os.path.join(results.save_dir, file)
            dst = os.path.join(output_dir, file)
            os.rename(src, dst)
            print(f"Saved output video to {dst}")
else:
    print("Unsupported file type.")