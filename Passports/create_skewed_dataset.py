import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def apply_horizontal_skew(image, skew_factor=0.2):
    """Apply horizontal skewing to an image"""
    height, width = image.shape[:2]
    
    # Create transformation matrix for horizontal skewing
    # Points: top-left, top-right, bottom-left, bottom-right
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = np.float32([[0, 0], [width, 0], [skew_factor * height, height], [width + skew_factor * height, height]])
    
    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    skewed_image = cv2.warpPerspective(image, matrix, (int(width + abs(skew_factor * height)), height))
    
    return skewed_image

def apply_vertical_skew(image, skew_factor=0.2):
    """Apply vertical skewing to an image"""
    height, width = image.shape[:2]
    
    # Create transformation matrix for vertical skewing
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = np.float32([[0, 0], [width + skew_factor * width, 0], [0, height], [width, height]])
    
    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    skewed_image = cv2.warpPerspective(image, matrix, (int(width + abs(skew_factor * width)), height))
    
    return skewed_image

def create_skewed_dataset(source_dir, output_dir, h_skew_factors=[-0.3, -0.15, 0.15, 0.3], v_skew_factors=[-0.3, -0.15, 0.15, 0.3]):
    """
    Create skewed versions of all images in the dataset
    
    Args:
        source_dir: Path to the source dataset directory
        output_dir: Path to the output directory for skewed images
        h_skew_factors: List of horizontal skew factors to apply
        v_skew_factors: List of vertical skew factors to apply
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all country directories
    country_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name != 'results']
    
    print(f"Found {len(country_dirs)} country directories")
    print(f"Processing with horizontal skew factors: {h_skew_factors}")
    print(f"Processing with vertical skew factors: {v_skew_factors}")
    
    total_images = 0
    for country_dir in country_dirs:
        jpg_files = list(country_dir.glob("*.jpg"))
        total_images += len(jpg_files)
    
    print(f"Total images to process: {total_images}")
    print(f"Total skewed images to create: {total_images * (len(h_skew_factors) + len(v_skew_factors))}")
    
    processed_images = 0
    
    # Process each country directory
    for country_dir in country_dirs:
        country_code = country_dir.name
        print(f"\nProcessing country: {country_code}")
        
        # Create output directories for this country
        h_skew_output_dir = output_path / f"{country_code}_h_skewed"
        v_skew_output_dir = output_path / f"{country_code}_v_skewed"
        h_skew_output_dir.mkdir(parents=True, exist_ok=True)
        v_skew_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all JPG files in the country directory
        jpg_files = list(country_dir.glob("*.jpg"))
        
        # Process each image
        for jpg_file in tqdm(jpg_files, desc=f"Processing {country_code}"):
            try:
                # Read the original image
                image = cv2.imread(str(jpg_file))
                if image is None:
                    print(f"Warning: Could not read image {jpg_file}")
                    continue
                
                base_name = jpg_file.stem  # filename without extension
                
                # Apply horizontal skewing
                for i, h_skew in enumerate(h_skew_factors):
                    h_skewed_image = apply_horizontal_skew(image, h_skew)
                    h_output_filename = h_skew_output_dir / f"{base_name}_h_skew_{i+1}.jpg"
                    cv2.imwrite(str(h_output_filename), h_skewed_image)
                
                # Apply vertical skewing
                for i, v_skew in enumerate(v_skew_factors):
                    v_skewed_image = apply_vertical_skew(image, v_skew)
                    v_output_filename = v_skew_output_dir / f"{base_name}_v_skew_{i+1}.jpg"
                    cv2.imwrite(str(v_output_filename), v_skewed_image)
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing {jpg_file}: {str(e)}")
                continue
    
    print(f"\nCompleted! Processed {processed_images} original images")
    print(f"Created skewed images in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create skewed versions of passport dataset images")
    parser.add_argument("--source", "-s", type=str, default=".", 
                       help="Source directory containing the dataset (default: current directory)")
    parser.add_argument("--output", "-o", type=str, default="skewed_dataset", 
                       help="Output directory for skewed images (default: skewed_dataset)")
    parser.add_argument("--h-skew", type=float, nargs='+', default=[-0.3, -0.15, 0.15, 0.3],
                       help="Horizontal skew factors to apply (default: -0.3 -0.15 0.15 0.3)")
    parser.add_argument("--v-skew", type=float, nargs='+', default=[-0.3, -0.15, 0.15, 0.3],
                       help="Vertical skew factors to apply (default: -0.3 -0.15 0.15 0.3)")
    
    args = parser.parse_args()
    
    create_skewed_dataset(
        source_dir=args.source,
        output_dir=args.output,
        h_skew_factors=args.h_skew,
        v_skew_factors=args.v_skew
    )

if __name__ == "__main__":
    main() 