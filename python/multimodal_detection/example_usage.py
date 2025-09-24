#!/usr/bin/env python3
"""
Example usage of the Multi-Modal Detection Script
"""

import os
import sys
from pathlib import Path
import logging

# Add the multimodal_detection module to path
sys.path.insert(0, str(Path(__file__).parent))

from detector import MultiModalDetector
from config import get_config, validate_config
from utils import setup_logging, find_image_files, validate_directory_structure


def example_single_image():
    """Example: Process a single image"""
    print("=== Single Image Detection Example ===")
    
    # Setup
    config = get_config()
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Initialize detector
    detector = MultiModalDetector(
        api_key=config['api_key'],
        api_url=config['api_url'],
        model_name=config['model_name']
    )
    
    # Example image path (you would replace this with a real image)
    image_path = "example_images/sample_rgb.jpg"
    
    if not Path(image_path).exists():
        print(f"Example image not found: {image_path}")
        print("Please provide a valid image path or create the example_images directory")
        return
    
    # Process image
    result = detector.detect_objects(image_path, image_type="RGB")
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Detected {len(result.detections)} objects:")
        for i, detection in enumerate(result.detections):
            print(f"  {i+1}. {detection.get('class', 'unknown')} "
                  f"(confidence: {detection.get('confidence', 0):.2f})")
        
        print(f"Processing time: {result.processing_time:.2f} seconds")


def example_batch_processing():
    """Example: Process multiple images in batch"""
    print("\n=== Batch Processing Example ===")
    
    # Setup
    config = get_config()
    detector = MultiModalDetector(
        api_key=config['api_key'],
        api_url=config['api_url'],
        model_name=config['model_name']
    )
    
    # Example directory with images
    image_dir = Path("example_images")
    output_dir = Path("detection_results")
    
    if not image_dir.exists():
        print(f"Example image directory not found: {image_dir}")
        print("Please create the example_images directory and add some images")
        return
    
    # Find image files
    image_files = find_image_files(image_dir)
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images for processing")
    
    # Process batch
    results = detector.batch_detect(
        image_files,
        output_dir=output_dir,
        save_intermediate=True
    )
    
    # Summary
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    print(f"Batch processing completed:")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    
    if failed:
        print("Failed images:")
        for result in failed:
            print(f"  - {result.image_path}: {result.error}")


def example_with_visualization():
    """Example: Process image and create visualization"""
    print("\n=== Visualization Example ===")
    
    config = get_config()
    detector = MultiModalDetector(
        api_key=config['api_key'],
        api_url=config['api_url'],
        model_name=config['model_name']
    )
    
    image_path = "example_images/sample_rgb.jpg"
    output_dir = Path("detection_results")
    
    if not Path(image_path).exists():
        print(f"Example image not found: {image_path}")
        return
    
    # Process image
    result = detector.detect_objects(image_path)
    
    if result.error:
        print(f"Error: {result.error}")
        return
    
    # Create visualization
    viz_file = detector.visualize_detections(result, output_dir)
    if viz_file:
        print(f"Visualization saved to: {viz_file}")
    
    # Save YOLO annotations
    detector.save_yolo_annotations(result, output_dir)
    print(f"YOLO annotations saved to: {output_dir}")


def example_ir_image_processing():
    """Example: Process IR (thermal) image"""
    print("\n=== IR Image Processing Example ===")
    
    config = get_config()
    detector = MultiModalDetector(
        api_key=config['api_key'],
        api_url=config['api_url'],
        model_name=config['model_name']
    )
    
    # Example IR image path
    ir_image_path = "example_images/sample_ir.jpg"
    
    if not Path(ir_image_path).exists():
        print(f"Example IR image not found: {ir_image_path}")
        print("Please provide a thermal/IR image for this example")
        return
    
    # Process IR image
    result = detector.detect_objects(ir_image_path, image_type="IR")
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Detected {len(result.detections)} objects in IR image:")
        for detection in result.detections:
            print(f"  - {detection.get('class', 'unknown')} "
                  f"(confidence: {detection.get('confidence', 0):.2f})")


def create_example_setup():
    """Create example directory structure and sample files"""
    print("\n=== Setting up example environment ===")
    
    # Create directories
    dirs = validate_directory_structure(Path.cwd(), create_missing=True)
    example_dir = Path("example_images")
    example_dir.mkdir(exist_ok=True)
    
    print(f"Created directories:")
    for name, path in dirs.items():
        print(f"  - {name}: {path}")
    
    print(f"Created example images directory: {example_dir}")
    
    # Create sample configuration file
    config_file = Path("detection_config.json")
    sample_config = {
        "api_key": "your_api_key_here",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "confidence_threshold": 0.3,
        "max_batch_size": 50,
        "timeout": 60
    }
    
    if not config_file.exists():
        import json
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"Created sample configuration: {config_file}")
    
    print("\nTo use the examples:")
    print("1. Set your API key: export QWEN_API_KEY=your_actual_api_key")
    print("2. Add some images to the example_images/ directory")
    print("3. Run the examples")


def main():
    """Main function to run examples"""
    # Setup logging
    setup_logging(level=logging.INFO)
    
    print("Multi-Modal Detection Script - Usage Examples")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('QWEN_API_KEY'):
        print("WARNING: QWEN_API_KEY environment variable not set!")
        print("Please set your API key before running detection examples.")
        print("\nRunning setup example instead...")
        create_example_setup()
        return
    
    try:
        # Run examples
        create_example_setup()
        example_single_image()
        example_batch_processing()
        example_with_visualization()
        example_ir_image_processing()
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
        logging.exception("Exception in main")


if __name__ == '__main__':
    main()