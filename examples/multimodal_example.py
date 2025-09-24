#!/usr/bin/env python3
"""
Example usage of the multi-modal processor with Qwen model
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multimodal_processor import (
    ProcessingConfig, 
    QwenMultiModalClient, 
    BatchProcessor,
    MultiModalLogger
)
from config.multimodal_config import MultiModalConfig, PROMPT_TEMPLATES, BATCH_CONFIGS

def example_single_image_processing():
    """Example: Process a single image with text prompt"""
    print("=== Single Image Processing Example ===")
    
    try:
        # Initialize configuration
        config = ProcessingConfig()
        config.model_name = MultiModalConfig.MODEL_NAME
        
        # Initialize client
        client = QwenMultiModalClient(config)
        logger = MultiModalLogger()
        
        # Example text prompt
        text_prompt = "Analyze this image and describe what you see. Focus on colors, objects, and overall composition."
        
        # Example image paths (you would replace these with actual image paths)
        sample_images = [
            "./examples/sample_rgb.jpg",  # Replace with actual paths
            "./examples/sample_ir.jpg"
        ]
        
        # Process only existing images
        existing_images = [img for img in sample_images if Path(img).exists()]
        
        if not existing_images:
            logger.warning("No sample images found. Please add images to examples/ directory.")
            return
        
        # Process the input
        result = client.process_single_input(
            text=text_prompt,
            image_paths=existing_images,
            temperature=0.7,
            max_tokens=800
        )
        
        if result['success']:
            response = result['response']
            if 'choices' in response and response['choices']:
                print(f"AI Response: {response['choices'][0]['message']['content']}")
            else:
                print(f"Full Response: {response}")
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Example failed: {str(e)}")

def example_batch_processing():
    """Example: Batch process multiple images"""
    print("\n=== Batch Processing Example ===")
    
    try:
        # Initialize
        config = ProcessingConfig()
        config.model_name = MultiModalConfig.MODEL_NAME
        processor = BatchProcessor(config)
        logger = MultiModalLogger()
        
        # Create sample input data
        input_data = [
            {
                "text": "Describe this RGB image in detail.",
                "image_paths": ["./examples/sample_rgb1.jpg"],
                "api_params": {"temperature": 0.5, "max_tokens": 600}
            },
            {
                "text": "Analyze this IR thermal image.",
                "image_paths": ["./examples/sample_ir1.jpg"],
                "api_params": {"temperature": 0.3, "max_tokens": 600}
            },
            {
                "text": "Compare these RGB and IR images of the same scene.",
                "image_paths": ["./examples/sample_rgb2.jpg", "./examples/sample_ir2.jpg"],
                "api_params": {"temperature": 0.4, "max_tokens": 800}
            }
        ]
        
        # Filter to only include existing images
        valid_input_data = []
        for item in input_data:
            existing_paths = [p for p in item["image_paths"] if Path(p).exists()]
            if existing_paths:
                item["image_paths"] = existing_paths
                valid_input_data.append(item)
        
        if not valid_input_data:
            logger.warning("No valid input data with existing images. Please add sample images to examples/ directory.")
            return
        
        # Process batch
        output_dir = Path("./output/batch_example")
        results = processor.process_batch(
            input_data=valid_input_data,
            output_dir=output_dir,
            batch_size=2
        )
        
        print(f"Batch processing completed!")
        print(f"Total processed: {results['total_processed']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Batch example failed: {str(e)}")

def example_directory_processing():
    """Example: Process all images in a directory"""
    print("\n=== Directory Processing Example ===")
    
    try:
        # Initialize
        config = ProcessingConfig()
        config.model_name = MultiModalConfig.MODEL_NAME
        processor = BatchProcessor(config)
        logger = MultiModalLogger()
        
        # Example directory (create if it doesn't exist)
        image_dir = Path("./examples/sample_images")
        image_dir.mkdir(parents=True, exist_ok=True)
        
        if not any(image_dir.iterdir()):
            logger.warning(f"No images found in {image_dir}. Please add sample images.")
            return
        
        # Process directory
        results = processor.process_image_directory(
            image_dir=image_dir,
            prompt_template=PROMPT_TEMPLATES["image_analysis"],
            output_dir="./output/directory_example",
            batch_size=3,
            api_params={"temperature": 0.6, "max_tokens": 700}
        )
        
        print(f"Directory processing completed!")
        print(f"Images processed: {results['total_processed']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Directory example failed: {str(e)}")

def example_with_different_configs():
    """Example: Using different batch configurations"""
    print("\n=== Different Configurations Example ===")
    
    try:
        config = ProcessingConfig()
        config.model_name = MultiModalConfig.MODEL_NAME
        processor = BatchProcessor(config)
        logger = MultiModalLogger()
        
        # Example using surveillance configuration
        surveillance_config = BATCH_CONFIGS["surveillance"]
        
        input_data = [
            {
                "text": surveillance_config["prompt_template"].format(
                    image_type="RGB",
                    filename="security_cam1.jpg",
                    filepath="./examples/security_cam1.jpg"
                ),
                "image_paths": ["./examples/security_cam1.jpg"] if Path("./examples/security_cam1.jpg").exists() else [],
                "api_params": surveillance_config["api_params"]
            }
        ]
        
        # Filter valid data
        valid_data = [item for item in input_data if item["image_paths"]]
        
        if valid_data:
            results = processor.process_batch(
                input_data=valid_data,
                output_dir="./output/surveillance_example",
                batch_size=surveillance_config["batch_size"]
            )
            
            print(f"Surveillance config processing completed!")
            print(f"Success rate: {results['success_rate']:.2%}")
        else:
            logger.warning("No valid images for surveillance example.")
            
    except Exception as e:
        print(f"Configuration example failed: {str(e)}")

def create_sample_structure():
    """Create sample directory structure and README"""
    print("=== Creating Sample Structure ===")
    
    # Create directories
    directories = [
        "./examples",
        "./examples/sample_images",
        "./output",
        "./src/logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create README
    readme_content = """# Multi-Modal Processing Examples

This directory contains example scripts and sample data for the multi-modal processor.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   export QWEN_API_KEY="your_api_key_here"
   # OR
   export DASHSCOPE_API_KEY="your_dashscope_api_key_here"
   ```

3. Add sample images to the `sample_images/` directory:
   - RGB images: name them with 'rgb', 'color', or 'visible' in the filename
   - IR images: name them with 'ir', 'infrared', or 'thermal' in the filename

## Usage

Run the example script:
```bash
python examples/multimodal_example.py
```

## Sample Image Naming Convention

- `sample_rgb_001.jpg` - RGB/color image
- `sample_ir_001.jpg` - Infrared/thermal image
- `scene1_color.png` - Another RGB image
- `scene1_thermal.png` - Corresponding IR image

The processor will automatically categorize images based on filename keywords.
"""
    
    readme_path = Path("./examples/README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README: {readme_path}")

def main():
    """Run all examples"""
    print("Multi-Modal Processing Examples")
    print("=" * 40)
    
    # Ensure MultiModalConfig directories exist
    MultiModalConfig.ensure_directories()
    
    # Validate configuration
    config_errors = MultiModalConfig.validate_config()
    if config_errors:
        print("Configuration errors:")
        for error in config_errors:
            print(f"  - {error}")
        print("\nPlease fix configuration issues before running examples.")
        return
    
    # Create sample structure
    create_sample_structure()
    
    # Run examples
    example_single_image_processing()
    example_batch_processing()
    example_directory_processing()
    example_with_different_configs()
    
    print("\n" + "=" * 40)
    print("Examples completed! Check the output/ directory for results.")

if __name__ == "__main__":
    main()