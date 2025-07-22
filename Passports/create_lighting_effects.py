import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random

def add_glare_effect(image, intensity=0.7, num_glares=1):
    """Add glare/bright spots to simulate camera flash or sunlight reflection"""
    height, width = image.shape[:2]
    glare_image = image.copy().astype(np.float32)
    
    for _ in range(num_glares):
        # Random position for glare
        center_x = random.randint(int(width * 0.2), int(width * 0.8))
        center_y = random.randint(int(height * 0.2), int(height * 0.8))
        
        # Random glare size
        radius = random.randint(int(min(width, height) * 0.05), int(min(width, height) * 0.15))
        
        # Create glare mask with gradient
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        # Create gradient effect for more realistic glare
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        gradient = np.exp(-distance / (radius * 0.5))
        gradient = np.clip(gradient, 0, 1)
        
        # Apply glare effect
        glare_effect = intensity * gradient * 255
        glare_image[mask] = np.clip(glare_image[mask] + glare_effect[mask][:, np.newaxis], 0, 255)
    
    return glare_image.astype(np.uint8)

def add_shadow_effect(image, intensity=0.4, num_shadows=1):
    """Add shadow effects to simulate uneven lighting or objects casting shadows"""
    height, width = image.shape[:2]
    shadow_image = image.copy().astype(np.float32)
    
    for _ in range(num_shadows):
        # Random shadow type: rectangular, circular, or irregular
        shadow_type = random.choice(['rectangular', 'circular', 'irregular'])
        
        if shadow_type == 'rectangular':
            # Rectangle shadow
            x1 = random.randint(0, int(width * 0.7))
            y1 = random.randint(0, int(height * 0.7))
            x2 = random.randint(x1, width)
            y2 = random.randint(y1, height)
            
            # Create soft edge shadow
            mask = np.zeros((height, width), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            
            # Apply Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
        elif shadow_type == 'circular':
            # Circular shadow
            center_x = random.randint(int(width * 0.2), int(width * 0.8))
            center_y = random.randint(int(height * 0.2), int(height * 0.8))
            radius = random.randint(int(min(width, height) * 0.1), int(min(width, height) * 0.3))
            
            y, x = np.ogrid[:height, :width]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).astype(np.float32)
            
            # Apply Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            
        else:  # irregular
            # Create irregular shadow using random points
            mask = np.zeros((height, width), dtype=np.uint8)
            num_points = random.randint(3, 6)
            points = []
            for _ in range(num_points):
                x = random.randint(int(width * 0.1), int(width * 0.9))
                y = random.randint(int(height * 0.1), int(height * 0.9))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            mask = mask.astype(np.float32) / 255.0
            
            # Apply Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (41, 41), 0)
        
        # Apply shadow effect (darken the masked area)
        shadow_factor = 1 - (intensity * mask[:, :, np.newaxis])
        shadow_image = shadow_image * shadow_factor
    
    return np.clip(shadow_image, 0, 255).astype(np.uint8)

def add_uneven_lighting(image, lighting_type='gradient', intensity=0.5):
    """Add uneven lighting effects like gradients or spotlight effects"""
    height, width = image.shape[:2]
    lighting_image = image.copy().astype(np.float32)
    
    if lighting_type == 'gradient':
        # Linear gradient lighting
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            # Left to right gradient
            gradient = np.linspace(1 - intensity, 1 + intensity, width)
            gradient = np.tile(gradient, (height, 1))
        elif direction == 'vertical':
            # Top to bottom gradient
            gradient = np.linspace(1 - intensity, 1 + intensity, height)
            gradient = np.tile(gradient.reshape(-1, 1), (1, width))
        else:  # diagonal
            # Diagonal gradient
            y, x = np.ogrid[:height, :width]
            gradient = (x + y) / (width + height)
            gradient = 1 - intensity + 2 * intensity * gradient
        
    elif lighting_type == 'spotlight':
        # Spotlight effect from center
        center_x = width // 2 + random.randint(-width//4, width//4)
        center_y = height // 2 + random.randint(-height//4, height//4)
        
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(width ** 2 + height ** 2) / 2
        
        gradient = 1 + intensity * (1 - distance / max_distance)
        gradient = np.clip(gradient, 1 - intensity, 1 + intensity)
        
    elif lighting_type == 'vignette':
        # Vignette effect (darker at edges)
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
        
        gradient = 1 - intensity * (distance / max_distance) ** 2
        gradient = np.clip(gradient, 1 - intensity, 1)
    
    # Apply lighting effect
    lighting_image = lighting_image * gradient[:, :, np.newaxis]
    
    return np.clip(lighting_image, 0, 255).astype(np.uint8)

def apply_lighting_effects(image, effect_type='combined', glare_intensity=0.6, shadow_intensity=0.4, 
                          lighting_intensity=0.5, num_glares=1, num_shadows=1):
    """Apply specified lighting effects to an image"""
    
    if effect_type == 'glare':
        return add_glare_effect(image, glare_intensity, num_glares)
    elif effect_type == 'shadow':
        return add_shadow_effect(image, shadow_intensity, num_shadows)
    elif effect_type == 'uneven_light':
        lighting_type = random.choice(['gradient', 'spotlight', 'vignette'])
        return add_uneven_lighting(image, lighting_type, lighting_intensity)
    elif effect_type == 'combined':
        # Apply multiple effects randomly
        result = image.copy()
        
        # Randomly apply each effect
        if random.random() < 0.7:  # 70% chance for uneven lighting
            lighting_type = random.choice(['gradient', 'spotlight', 'vignette'])
            result = add_uneven_lighting(result, lighting_type, lighting_intensity * 0.8)
        
        if random.random() < 0.5:  # 50% chance for shadows
            result = add_shadow_effect(result, shadow_intensity * 0.8, num_shadows)
        
        if random.random() < 0.3:  # 30% chance for glare
            result = add_glare_effect(result, glare_intensity * 0.8, num_glares)
        
        return result
    elif effect_type == 'all':
        # Apply all effects separately
        glare_image = add_glare_effect(image, glare_intensity, num_glares)
        shadow_image = add_shadow_effect(image, shadow_intensity, num_shadows)
        
        lighting_types = ['gradient', 'spotlight', 'vignette']
        lighting_images = []
        for lt in lighting_types:
            lighting_images.append(add_uneven_lighting(image, lt, lighting_intensity))
        
        return glare_image, shadow_image, lighting_images

def create_lighting_dataset(source_dir, countries, effect_type='combined', 
                          glare_intensity=0.6, shadow_intensity=0.4, lighting_intensity=0.5,
                          num_glares=1, num_shadows=1):
    """
    Create lighting effect versions of images for specified countries
    
    Args:
        source_dir: Path to the source dataset directory
        countries: List of country codes to process
        effect_type: 'glare', 'shadow', 'uneven_light', 'combined', or 'all'
        glare_intensity: Intensity of glare effects (0.0 - 1.0)
        shadow_intensity: Intensity of shadow effects (0.0 - 1.0) 
        lighting_intensity: Intensity of uneven lighting (0.0 - 1.0)
        num_glares: Number of glare spots to add
        num_shadows: Number of shadow areas to add
    """
    
    source_path = Path(source_dir)
    
    # Create main "lighting effects" directory
    main_output_dir = source_path / "lighting effects"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all available country directories
    available_countries = [d.name for d in source_path.iterdir() 
                          if d.is_dir() and d.name not in ['results', 'skewed passport', 'lighting effects']]
    
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
    print(f"Effect type: {effect_type}")
    print(f"Glare intensity: {glare_intensity}, Number of glares: {num_glares}")
    print(f"Shadow intensity: {shadow_intensity}, Number of shadows: {num_shadows}")
    print(f"Lighting intensity: {lighting_intensity}")
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
        
        # Create country directory inside "lighting effects"
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
                
                if effect_type == 'all':
                    # Apply all effects and save separately
                    glare_image, shadow_image, lighting_images = apply_lighting_effects(
                        image, effect_type, glare_intensity, shadow_intensity, 
                        lighting_intensity, num_glares, num_shadows
                    )
                    
                    # Save glare image
                    glare_filename = country_output_dir / f"{base_name}_glare_{glare_intensity}.jpg"
                    cv2.imwrite(str(glare_filename), glare_image)
                    
                    # Save shadow image  
                    shadow_filename = country_output_dir / f"{base_name}_shadow_{shadow_intensity}.jpg"
                    cv2.imwrite(str(shadow_filename), shadow_image)
                    
                    # Save lighting images
                    lighting_types = ['gradient', 'spotlight', 'vignette']
                    for i, lighting_image in enumerate(lighting_images):
                        lighting_filename = country_output_dir / f"{base_name}_lighting_{lighting_types[i]}_{lighting_intensity}.jpg"
                        cv2.imwrite(str(lighting_filename), lighting_image)
                
                else:
                    # Apply single or combined effect
                    processed_image = apply_lighting_effects(
                        image, effect_type, glare_intensity, shadow_intensity,
                        lighting_intensity, num_glares, num_shadows
                    )
                    
                    # Generate filename based on effect type
                    if effect_type == 'combined':
                        output_filename = country_output_dir / f"{base_name}_lighting_combined.jpg"
                    else:
                        output_filename = country_output_dir / f"{base_name}_lighting_{effect_type}.jpg"
                    
                    cv2.imwrite(str(output_filename), processed_image)
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing {jpg_file}: {str(e)}")
                continue
    
    print(f"\nCompleted! Processed {processed_images} original images")
    print(f"Created lighting effect images in: {main_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create lighting effect versions of passport dataset images")
    parser.add_argument("--source", "-s", type=str, default=".", 
                       help="Source directory containing the dataset (default: current directory)")
    parser.add_argument("--countries", "-c", type=str, nargs='+', default=['all'],
                       help="Country codes to process (e.g., USA GBR CAN) or 'all' for all countries")
    parser.add_argument("--effect", "-e", type=str, 
                       choices=['glare', 'shadow', 'uneven_light', 'combined', 'all'], 
                       default='combined',
                       help="Lighting effect type to apply (default: combined)")
    parser.add_argument("--glare-intensity", type=float, default=0.6,
                       help="Glare effect intensity 0.0-1.0 (default: 0.6)")
    parser.add_argument("--shadow-intensity", type=float, default=0.4,
                       help="Shadow effect intensity 0.0-1.0 (default: 0.4)")
    parser.add_argument("--lighting-intensity", type=float, default=0.5,
                       help="Uneven lighting intensity 0.0-1.0 (default: 0.5)")
    parser.add_argument("--num-glares", type=int, default=1,
                       help="Number of glare spots to add (default: 1)")
    parser.add_argument("--num-shadows", type=int, default=1,
                       help="Number of shadow areas to add (default: 1)")
    
    args = parser.parse_args()
    
    # Show help message with examples if no specific arguments provided
    if len(vars(args)) == len(parser.parse_args([]).__dict__):
        print("\nExample usage:")
        print("python3 create_lighting_effects.py --countries USA GBR --effect combined")
        print("python3 create_lighting_effects.py --countries CAN --effect glare --glare-intensity 0.8 --num-glares 2")
        print("python3 create_lighting_effects.py --countries all --effect shadow --shadow-intensity 0.6")
        print("python3 create_lighting_effects.py --countries ITA --effect all --lighting-intensity 0.4")
        print()
    
    create_lighting_dataset(
        source_dir=args.source,
        countries=args.countries,
        effect_type=args.effect,
        glare_intensity=args.glare_intensity,
        shadow_intensity=args.shadow_intensity,
        lighting_intensity=args.lighting_intensity,
        num_glares=args.num_glares,
        num_shadows=args.num_shadows
    )

if __name__ == "__main__":
    main() 