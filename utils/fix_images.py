import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

def find_pngs(root_dir):
    """Recursively find all PNG files in the given directory."""
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.png'):
                yield os.path.join(subdir, file)

def crop_image(input_path, size=(1280, 720)):
    """Crop the image from the top left corner to the specified size."""
    try:
        with Image.open(input_path) as img:
            cropped_img = img.crop((0, 0, *size))  # Crop from top left to 1280x720
            cropped_img.save(input_path)
            print(f"Cropped and saved: {input_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main(root_dir):
    png_files = list(find_pngs(root_dir))
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(crop_image, png_file) for png_file in png_files]
        for future in as_completed(futures):
            # Results are processed here if needed
            pass

if __name__ == "__main__":
    root_dir = 'dataset'  # Change this to your directory
    main(root_dir)
