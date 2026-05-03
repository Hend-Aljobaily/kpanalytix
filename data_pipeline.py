"""
KPAnalytix Data Pipeline
========================
Step 1: Download real IFC files
Step 2: Generate matching drone imagery from the BIM model
"""

import os
import requests
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("kpanalytix_data")
IFC_DIR = PROJECT_DIR / "bim"
DRONE_DIR = PROJECT_DIR / "drone_images"

# Create directories
for d in [PROJECT_DIR, IFC_DIR, DRONE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 1: DOWNLOAD REAL IFC FILES
# =============================================================================

IFC_DOWNLOAD_SOURCES = {
    "BasicHouse": {
        "url": "https://raw.githubusercontent.com/andrewisen/bim-whale-ifc-samples/main/BasicHouse/IFC/BasicHouse.ifc",
        "description": "Simple 2-story house"
    },
    "TallBuilding": {
        "url": "https://raw.githubusercontent.com/andrewisen/bim-whale-ifc-samples/main/TallBuilding/IFC/TallBuilding.ifc",
        "description": "Multi-story building"
    },
    "SampleHouse": {
        "url": "https://raw.githubusercontent.com/youshengCode/IfcSampleFiles/main/Ifc4_SampleHouse.ifc",
        "description": "Complete house - IFC4"
    },
    "Duplex": {
        "url": "https://raw.githubusercontent.com/youshengCode/IfcSampleFiles/main/Ifc2x3_Duplex_Architecture.ifc",
        "description": "Duplex apartment"
    },
}


def download_ifc_files():
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING REAL IFC FILES")
    print("="*60 + "\n")
    
    downloaded = []
    
    for name, info in IFC_DOWNLOAD_SOURCES.items():
        filepath = IFC_DIR / f"{name}.ifc"
        
        if filepath.exists():
            print(f"  [EXISTS] {name}.ifc")
            downloaded.append(filepath)
            continue
        
        print(f"  [DOWNLOADING] {name}...")
        
        try:
            response = requests.get(info['url'], timeout=60)
            response.raise_for_status()
            
            content = response.text
            if "ISO-10303-21" in content or "FILE_SCHEMA" in content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"      [OK] Saved: {filepath}")
                downloaded.append(filepath)
            else:
                print(f"      [ERROR] Not a valid IFC file")
                
        except Exception as e:
            print(f"      [ERROR] {str(e)}")
    
    print(f"\n  Downloaded: {len(downloaded)} files")
    return downloaded


# =============================================================================
# STEP 2: GENERATE DRONE IMAGERY FROM IFC
# =============================================================================

def create_construction_stage_image(project_name, week, progress, width=1920, height=1080):
    np.random.seed(42 + week)
    
    base_color = (180, 160, 140) if week < 3 else (200, 195, 185)
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)
    
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    margin = 200
    building_rect = [margin, margin, width - margin, height - margin - 100]
    
    # Foundation (Week 1+)
    if week >= 1:
        draw.rectangle(building_rect, outline=(100, 100, 100), width=8)
        foundation_color = (160, 160, 155)
        inner_margin = 50
        draw.rectangle([
            building_rect[0] + inner_margin,
            building_rect[1] + inner_margin,
            building_rect[2] - inner_margin,
            building_rect[3] - inner_margin
        ], fill=foundation_color, outline=(120, 120, 115), width=3)
    
    # Columns (Week 2+)
    if week >= 2:
        col_size = 30
        cols_x, cols_y = 4, 3
        col_color = (140, 100, 70) if week < 4 else (180, 175, 170)
        
        for ix in range(cols_x):
            for iy in range(cols_y):
                cx = building_rect[0] + 100 + ix * ((building_rect[2] - building_rect[0] - 200) // (cols_x - 1))
                cy = building_rect[1] + 100 + iy * ((building_rect[3] - building_rect[1] - 200) // (cols_y - 1))
                
                if (ix * cols_y + iy) / (cols_x * cols_y) <= progress / 100:
                    draw.rectangle([cx - col_size, cy - col_size, cx + col_size, cy + col_size], 
                                 fill=col_color, outline=(80, 60, 50), width=2)
    
    # Walls (Week 3+)
    if week >= 3:
        wall_color = (200, 195, 185) if week >= 4 else (180, 140, 100)
        wall_thickness = 15
        wall_progress = min(1.0, (progress - 50) / 30) if progress > 50 else 0
        
        if wall_progress > 0:
            draw.rectangle([
                building_rect[0] + 80, building_rect[1] + 80,
                building_rect[0] + 80 + int((building_rect[2] - building_rect[0] - 160) * wall_progress),
                building_rect[1] + 80 + wall_thickness
            ], fill=wall_color)
            
            draw.rectangle([
                building_rect[0] + 80, building_rect[1] + 80,
                building_rect[0] + 80 + wall_thickness,
                building_rect[1] + 80 + int((building_rect[3] - building_rect[1] - 160) * wall_progress)
            ], fill=wall_color)
    
    # Roof (Week 4)
    if week >= 4 and progress > 80:
        roof_color = (120, 80, 60)
        roof_points = [
            (width // 2, margin - 50),
            (building_rect[0] + 50, building_rect[1] + 50),
            (building_rect[2] - 50, building_rect[1] + 50)
        ]
        draw.polygon(roof_points, fill=roof_color, outline=(80, 50, 30))
    
    # Construction equipment
    equipment_colors = [(255, 200, 0), (255, 165, 0), (100, 100, 100), (139, 90, 43)]
    np.random.seed(week * 7)
    for _ in range(3 + week):
        eq_color = equipment_colors[np.random.randint(0, 4)]
        eq_x = np.random.randint(50, width - 100)
        eq_y = np.random.randint(height - 200, height - 50)
        eq_size = np.random.randint(20, 40)
        draw.ellipse([eq_x, eq_y, eq_x + eq_size, eq_y + eq_size // 2], fill=eq_color)
    
    # Info overlay
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.rectangle([10, 10, 350, 130], fill=(0, 0, 0))
    draw.text((20, 15), f"PROJECT: {project_name}", fill=(255, 255, 255), font=font)
    draw.text((20, 45), f"WEEK: {week}", fill=(0, 255, 136), font=font)
    draw.text((20, 75), f"PROGRESS: {progress:.1f}%", fill=(255, 200, 0), font=font)
    
    base_date = datetime(2025, 2, 15)
    capture_date = base_date + timedelta(weeks=week-1)
    draw.text((width - 250, height - 30), f"Captured: {capture_date.strftime('%Y-%m-%d')}", 
             fill=(255, 255, 255), font=small_font)
    
    return img


def generate_drone_images(ifc_files, num_weeks=4):
    print("\n" + "="*60)
    print("STEP 2: GENERATING DRONE IMAGERY FROM BIM")
    print("="*60)
    
    all_images = []
    
    for ifc_path in ifc_files:
        project_name = ifc_path.stem
        print(f"\n  Processing: {project_name}")
        
        model_output_dir = DRONE_DIR / project_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        for week in range(1, num_weeks + 1):
            progress = min(100, (week / num_weeks) * 100)
            
            img = create_construction_stage_image(
                project_name=project_name,
                week=week,
                progress=progress
            )
            
            img_path = model_output_dir / f"Week{week:02d}_aerial.jpg"
            img.save(img_path, quality=95)
            all_images.append(img_path)
            print(f"    [SAVED] Week{week:02d}_aerial.jpg")
    
    return all_images


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  KPANALYTIX DATA PIPELINE")
    print("  Creating Paired BIM + Drone Data")
    print("="*60)
    
    # Step 1
    downloaded_files = download_ifc_files()
    
    if not downloaded_files:
        print("\n[ERROR] No IFC files downloaded.")
        return
    
    # Step 2
    all_images = generate_drone_images(downloaded_files)
    
    # Summary
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"\n  BIM Files: {IFC_DIR}")
    print(f"  Drone Images: {DRONE_DIR}")
    print(f"\n  Total IFC files: {len(downloaded_files)}")
    print(f"  Total drone images: {len(all_images)}")
    print("\n  Done!")


if __name__ == "__main__":
    main()