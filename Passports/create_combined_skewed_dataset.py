import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def apply_combined_skew(image, h_skew_factor=0.2, v_skew_factor=0.2):
    """Apply both horizontal and vertical skewing to an image"""
    height, width = image.shape[:2]
    
    # Create transformation matrix for combined skewing
    # Start with source points (corners of original image)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Apply both horizontal and vertical skew transformations
    # Horizontal skew affects x-coordinates based on y-position
    # Vertical skew affects y-coordinates based on x-position
    dst_points = np.float32([
        [v_skew_factor * 0, 0],  # top-left: only vertical skew affects this (minimal at y=0)
        [width + v_skew_factor * 0, h_skew_factor * width],  # top-right: vertical skew + horizontal skew
        [v_skew_factor * height, height],  # bottom-left: vertical skew affects x, y stays same
        [width + v_skew_factor * height, height + h_skew_factor * width]  # bottom-right: both skews
    ])
    
    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Calculate output image size to fit the skewed image
    output_width = int(width + abs(v_skew_factor * height) + abs(h_skew_factor * width))
    output_height = int(height + abs(h_skew_factor * width))
    
    # Apply transformation
    skewed_image = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return skewed_image

def apply_horizontal_skew(image, skew_factor=0.2):
    """Apply horizontal skewing to an image"""
    height, width = image.shape[:2]
    
    # Create transformation matrix for horizontal skewing
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

def create_skewed_dataset(source_dir, countries, h_skew_angle, v_skew_angle, mode='combined'):
    """
    Create skewed versions of images for specified countries
    
    Args:
        source_dir: Path to the source dataset directory
        countries: List of country codes to process
        h_skew_angle: Horizontal skew angle
        v_skew_angle: Vertical skew angle
        mode: 'combined', 'horizontal', 'vertical', or 'all'
    """
    
    source_path = Path(source_dir)
    
    # Create main "skewed passport" directory
    main_output_dir = source_path / "skewed passport"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all available country directories
    available_countries = [d.name for d in source_path.iterdir() if d.is_dir() and d.name not in ['results', 'skewed passport']]
    
    # Filter countries based on user input
    if countries and countries != ['all']:
        # Validate that specified countries exist
        invalid_countries = [c for c in countries if c not in available_countries]
        if invalid_countries:
            print(f"Warning: These countries were not found: {invalid_countries}")
        
        countries_to_process = [c for c in countries if c in available_countries]
    else:
        countries_to_process = available_countries
    
    print(f"Available countries: {sorted(available_countries)}")
    print(f"Processing countries: {sorted(countries_to_process)}")
    print(f"Mode: {mode}")
    print(f"Horizontal skew angle: {h_skew_angle}")
    print(f"Vertical skew angle: {v_skew_angle}")
    print(f"Output directory: {main_output_dir}")
    
    total_images = 0
    for country in countries_to_process:
        country_dir = source_path / country
        jpg_files = list(country_dir.glob("*.jpg"))
        total_images += len(jpg_files)
    
    print(f"Total images to process: {total_images}")
    
    processed_images = 0
    
    # Process each country
    for country in countries_to_process:
        country_dir = source_path / country
        print(f"\nProcessing country: {country}")
        
        # Create country directory inside "skewed passport"
        country_output_dir = main_output_dir / country
        country_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all JPG files in the country directory
        jpg_files = list(country_dir.glob("*.jpg"))
        
        # Process each image
        for jpg_file in tqdm(jpg_files, desc=f"Processing {country}"):
            try:
                # Read the original image
                image = cv2.imread(str(jpg_file))
                if image is None:
                    print(f"Warning: Could not read image {jpg_file}")
                    continue
                
                base_name = jpg_file.stem  # filename without extension
                
                if mode == 'combined':
                    # Apply combined horizontal and vertical skewing
                    combined_skewed = apply_combined_skew(image, h_skew_angle, v_skew_angle)
                    output_filename = country_output_dir / f"{base_name}_combined_h{h_skew_angle}_v{v_skew_angle}.jpg"
                    cv2.imwrite(str(output_filename), combined_skewed)
                    
                elif mode == 'horizontal':
                    # Apply only horizontal skewing
                    h_skewed = apply_horizontal_skew(image, h_skew_angle)
                    output_filename = country_output_dir / f"{base_name}_h_skew_{h_skew_angle}.jpg"
                    cv2.imwrite(str(output_filename), h_skewed)
                    
                elif mode == 'vertical':
                    # Apply only vertical skewing
                    v_skewed = apply_vertical_skew(image, v_skew_angle)
                    output_filename = country_output_dir / f"{base_name}_v_skew_{v_skew_angle}.jpg"
                    cv2.imwrite(str(output_filename), v_skewed)
                    
                elif mode == 'all':
                    # Apply all three types of skewing
                    # Combined
                    combined_skewed = apply_combined_skew(image, h_skew_angle, v_skew_angle)
                    combined_filename = country_output_dir / f"{base_name}_combined_h{h_skew_angle}_v{v_skew_angle}.jpg"
                    cv2.imwrite(str(combined_filename), combined_skewed)
                    
                    # Horizontal only
                    h_skewed = apply_horizontal_skew(image, h_skew_angle)
                    h_filename = country_output_dir / f"{base_name}_h_skew_{h_skew_angle}.jpg"
                    cv2.imwrite(str(h_filename), h_skewed)
                    
                    # Vertical only
                    v_skewed = apply_vertical_skew(image, v_skew_angle)
                    v_filename = country_output_dir / f"{base_name}_v_skew_{v_skew_angle}.jpg"
                    cv2.imwrite(str(v_filename), v_skewed)
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing {jpg_file}: {str(e)}")
                continue
    
    print(f"\nCompleted! Processed {processed_images} original images")
    print(f"Created skewed images in: {main_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create skewed versions of passport dataset images with custom angles")
    parser.add_argument("--source", "-s", type=str, default=".", 
                       help="Source directory containing the dataset (default: current directory)")
    parser.add_argument("--countries", "-c", type=str, nargs='+', default=['all'],
                       help="Country codes to process (e.g., USA GBR CAN) or 'all' for all countries")
    parser.add_argument("--h-skew", type=float, default=0.2,
                       help="Horizontal skew angle (default: 0.2)")
    parser.add_argument("--v-skew", type=float, default=0.2,
                       help="Vertical skew angle (default: 0.2)")
    parser.add_argument("--mode", "-m", type=str, choices=['combined', 'horizontal', 'vertical', 'all'], 
                       default='combined',
                       help="Skewing mode: combined (both in single image), horizontal, vertical, or all")
    
    args = parser.parse_args()
    
    # Show help message with examples if no specific arguments provided
    if len(vars(args)) == len(parser.parse_args([]).__dict__):
        print("\nExample usage:")
        print("python3 create_combined_skewed_dataset.py --countries USA GBR --h-skew 0.3 --v-skew -0.2 --mode combined")
        print("python3 create_combined_skewed_dataset.py --countries CAN --h-skew 0.15 --mode horizontal")
        print("python3 create_combined_skewed_dataset.py --countries all --h-skew 0.25 --v-skew 0.25 --mode all")
        print()
    
    create_skewed_dataset(
        source_dir=args.source,
        countries=args.countries,
        h_skew_angle=args.h_skew,
        v_skew_angle=args.v_skew,
        mode=args.mode
    )

if __name__ == "__main__":
    main() 