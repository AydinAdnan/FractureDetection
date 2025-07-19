import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def process_single_image(img_path, output_dir=None, show_comparison=False):

    # Read image in grayscale
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None
    
    # 1. Histogram Equalization (basic global)
    equ_img = cv.equalizeHist(img)
    
    # 2. CLAHE (local histogram equalization)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    clahe_img = clahe.apply(img)
    
    # Save CLAHE enhanced image with original filename if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        original_filename = Path(img_path).name
        clahe_path = output_dir / original_filename
        cv.imwrite(str(clahe_path), clahe_img)
    if show_comparison:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original: {Path(img_path).name}')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(equ_img, cmap='gray')
        plt.title('Histogram Equalized')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(clahe_img, cmap='gray')
        plt.title('CLAHE Enhanced')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return clahe_img

def process_folder(folder_path, output_base_dir=None):

    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return
    
    # Get all image files (common extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Set up output directory to maintain same structure
    output_dir = None
    if output_base_dir:
        # Keep the same folder structure: train/images, test/images, valid/images
        relative_path = folder_path.relative_to(folder_path.parents[1])  # Get "train/images" part
        output_dir = Path(output_base_dir) / relative_path
    
    # Process each image
    processed_count = 0
    
    for img_path in image_files:
        result = process_single_image(
            img_path, 
            output_dir=output_dir, 
            show_comparison=False 
        )
        
        if result is not None:
            processed_count += 1
    
    print(f"Successfully processed {processed_count}/{len(image_files)} images from {folder_path}")

def process_all_folders(base_path, output_base_dir=None):
    base_path = Path(base_path)
    
    folders_to_process = [
        base_path / 'test' / 'images',
        base_path / 'train' / 'images', 
        base_path / 'valid' / 'images'
    ]
    
    for folder in folders_to_process:
        print(f"\n{'='*50}")
        print(f"Processing folder: {folder}")
        print(f"{'='*50}")
        
        process_folder(
            folder_path=folder,
            output_base_dir=output_base_dir
        )

# Main execution
if __name__ == '__main__':
    # Set your base directory path here
    base_directory = os.getcwd()  # Current working directory
    
    output_directory = "enhanced_dataset"  # Output base directory
    
    print("Starting CLAHE enhancement for all images...")
    process_all_folders(
        base_path=base_directory,
        output_base_dir=output_directory
    )
    print("CLAHE enhancement completed!")
