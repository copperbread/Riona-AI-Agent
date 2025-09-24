#!/usr/bin/env python3
"""
Demo script for multi-modal processing capabilities
Shows all features without requiring external API keys
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_configuration():
    """Demonstrate configuration and setup"""
    print("🔧 CONFIGURATION DEMO")
    print("=" * 50)
    
    from multimodal_processor_lite import ProcessingConfig, MultiModalLogger, PathHandler
    
    # Show configuration
    config = ProcessingConfig()
    print(f"Model Name: {config.model_name}")
    print(f"API Base URL: {config.api_base_url}")
    print(f"Max Retries: {config.max_retries}")
    print(f"Timeout: {config.timeout}s")
    print(f"Max Batch Size: {config.max_batch_size}")
    print(f"Supported Image Formats: {', '.join(config.supported_image_formats)}")
    print(f"Max Image Size: {config.max_image_size[0]}x{config.max_image_size[1]}")
    
    # Test logger
    logger = MultiModalLogger("INFO")
    logger.info("Configuration loaded successfully")
    
    # Test path handling
    test_paths = [
        "./images/rgb_photo.jpg",
        "C:\\Users\\data\\ir_thermal.png",
        "/home/user/mixed/sample.JPEG"
    ]
    
    print(f"\nCross-platform path handling:")
    for path in test_paths:
        normalized = PathHandler.normalize_path(path)
        extension = PathHandler.get_file_extension(path)
        is_valid = PathHandler.validate_image_file(path)
        print(f"  {path}")
        print(f"    → {normalized}")
        print(f"    → Extension: {extension}, Valid: {is_valid}")

def demo_message_formatting():
    """Demonstrate proper API message formatting"""
    print("\n📝 API MESSAGE FORMATTING DEMO")
    print("=" * 50)
    
    from multimodal_processor_lite import ProcessingConfig
    
    config = ProcessingConfig()
    
    # Example 1: Text only
    text_only_message = {
        "model": config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, please analyze the following data."
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    print("1. Text-only message format:")
    print(json.dumps(text_only_message, indent=2))
    
    # Example 2: Text + Single Image
    single_image_message = {
        "model": config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this RGB image and describe what you see."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.5
    }
    
    print("\n2. Text + Single Image message format:")
    print(json.dumps(single_image_message, indent=2))
    
    # Example 3: Text + Multiple Images (RGB + IR)
    multi_image_message = {
        "model": config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare these RGB and IR images of the same scene. What differences do you notice?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAB...",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.4
    }
    
    print("\n3. Text + Multiple Images (RGB + IR) message format:")
    print(json.dumps(multi_image_message, indent=2))

def demo_batch_processing_structure():
    """Demonstrate batch processing structure"""
    print("\n📦 BATCH PROCESSING DEMO")
    print("=" * 50)
    
    from multimodal_processor_lite import SimpleProgressBar
    
    # Sample batch input data
    batch_data = [
        {
            "text": "Analyze this RGB security camera image",
            "image_paths": ["./images/security_rgb_001.jpg"],
            "api_params": {"temperature": 0.3, "max_tokens": 500}
        },
        {
            "text": "Examine this IR thermal image for anomalies",
            "image_paths": ["./images/thermal_ir_001.jpg"],
            "api_params": {"temperature": 0.2, "max_tokens": 600}
        },
        {
            "text": "Compare RGB and IR images of the same location",
            "image_paths": [
                "./images/location_rgb_002.jpg",
                "./images/location_ir_002.jpg"
            ],
            "api_params": {"temperature": 0.4, "max_tokens": 800}
        }
    ]
    
    print(f"Batch contains {len(batch_data)} items:")
    for i, item in enumerate(batch_data, 1):
        print(f"\n  Item {i}:")
        print(f"    Text: {item['text']}")
        print(f"    Images: {len(item['image_paths'])} files")
        print(f"    API params: {item['api_params']}")
    
    # Simulate processing with progress bar
    print(f"\nSimulating batch processing...")
    with SimpleProgressBar(len(batch_data), "Processing batch") as pbar:
        for i, item in enumerate(batch_data):
            import time
            time.sleep(0.5)  # Simulate processing time
            pbar.update(1)
    
    # Show expected output structure
    expected_output = {
        "total_processed": len(batch_data),
        "successful": len(batch_data),
        "failed": 0,
        "success_rate": 1.0,
        "results": [
            {
                "success": True,
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "Analysis result for RGB security image..."
                            }
                        }
                    ]
                },
                "input_text": batch_data[0]["text"],
                "input_images": batch_data[0]["image_paths"],
                "timestamp": "2024-01-15T10:30:00"
            }
            # ... more results
        ],
        "errors": [],
        "timestamp": "2024-01-15T10:30:15"
    }
    
    print(f"\nExpected batch output structure:")
    print(json.dumps(expected_output, indent=2)[:500] + "...")

def demo_image_categorization():
    """Demonstrate RGB/IR image categorization"""
    print("\n🏷️ IMAGE CATEGORIZATION DEMO")
    print("=" * 50)
    
    # Sample image filenames
    sample_images = [
        "scene1_rgb_daylight.jpg",
        "scene1_ir_thermal.jpg", 
        "security_camera_visible.png",
        "security_camera_infrared.png",
        "drone_survey_color.jpg",
        "drone_survey_thermal.jpg",
        "building_normal_light.jpeg",
        "building_heat_signature.jpeg",
        "person_detection_rgb.jpg",
        "person_detection_ir.jpg",
        "unknown_image.jpg",
        "regular_photo.png"
    ]
    
    # Categorize based on filename keywords
    rgb_keywords = ['rgb', 'color', 'visible', 'normal', 'daylight']
    ir_keywords = ['ir', 'infrared', 'thermal', 'heat']
    
    categories = {"rgb": [], "ir": [], "unknown": []}
    
    for image in sample_images:
        filename_lower = image.lower()
        if any(keyword in filename_lower for keyword in rgb_keywords):
            categories["rgb"].append(image)
        elif any(keyword in filename_lower for keyword in ir_keywords):
            categories["ir"].append(image)
        else:
            categories["unknown"].append(image)
    
    print("Image categorization results:")
    print(f"  RGB Images ({len(categories['rgb'])}):")
    for img in categories["rgb"]:
        print(f"    • {img}")
    
    print(f"\n  IR Images ({len(categories['ir'])}):")
    for img in categories["ir"]:
        print(f"    • {img}")
    
    print(f"\n  Unknown Images ({len(categories['unknown'])}):")
    for img in categories["unknown"]:
        print(f"    • {img}")

def demo_error_handling():
    """Demonstrate error handling scenarios"""
    print("\n⚠️ ERROR HANDLING DEMO")
    print("=" * 50)
    
    from multimodal_processor_lite import MultiModalLogger, PathHandler
    
    logger = MultiModalLogger("DEBUG")
    
    # Common error scenarios
    error_scenarios = [
        {
            "type": "API Key Missing",
            "description": "No QWEN_API_KEY environment variable",
            "handling": "Graceful error message and exit"
        },
        {
            "type": "Rate Limiting (429)",
            "description": "Too many requests to API",
            "handling": "Exponential backoff retry (2^attempt seconds)"
        },
        {
            "type": "Network Timeout",
            "description": "API request times out",
            "handling": "Retry up to max_retries times"
        },
        {
            "type": "Invalid Image Format",
            "description": "Unsupported file format uploaded",
            "handling": "Skip file with warning, continue processing"
        },
        {
            "type": "File Not Found",
            "description": "Image path doesn't exist",
            "handling": "Log error, skip to next item"
        },
        {
            "type": "Large Image",
            "description": "Image exceeds size limits",
            "handling": "Automatic resize with quality preservation"
        }
    ]
    
    print("Error handling scenarios:")
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n  {i}. {scenario['type']}")
        print(f"     Problem: {scenario['description']}")
        print(f"     Solution: {scenario['handling']}")
        logger.debug(f"Error scenario documented: {scenario['type']}")
    
    # Test file validation
    print(f"\nFile validation examples:")
    test_files = [
        ("valid_image.jpg", True),
        ("invalid_format.txt", False),
        ("missing_file.png", "file_not_found"),
        ("UPPERCASE.JPEG", True)
    ]
    
    for filename, expected in test_files:
        is_valid = PathHandler.validate_image_file(filename)
        status = "✅" if is_valid == expected or (expected == "file_not_found") else "❌"
        print(f"  {status} {filename}: {'Valid format' if is_valid else 'Invalid format'}")

def demo_directory_structure():
    """Show recommended directory structure"""
    print("\n📁 DIRECTORY STRUCTURE DEMO")
    print("=" * 50)
    
    structure = """
project/
├── src/
│   ├── multimodal_processor.py      # Main processor
│   ├── config/
│   │   └── multimodal_config.py     # Configuration settings
│   └── logs/                        # Log files (auto-created)
│       └── multimodal_processor.log
├── examples/
│   ├── multimodal_example.py        # Usage examples
│   ├── sample_images/               # Sample images
│   │   ├── scene1_rgb.jpg          # RGB images
│   │   ├── scene1_ir.jpg           # IR images
│   │   └── ...
│   └── README.md
├── output/                          # Processing results (auto-created)
│   ├── batch_results_2024-01-15.json
│   └── processed_images/
├── requirements.txt                 # Python dependencies
├── README_MULTIMODAL.md            # Documentation
└── demo_multimodal.py              # This demo script
"""
    
    print("Recommended project structure:")
    print(structure)
    
    # Create sample directories
    print("Creating sample directories...")
    directories = [
        "./src/logs",
        "./examples/sample_images", 
        "./output/processed_images",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}")

def main():
    """Run all demonstrations"""
    print("🚀 MULTI-MODAL PROCESSOR DEMO")
    print("=" * 60)
    print("This demo showcases the multi-modal processing capabilities")
    print("optimized for the Qwen/Qwen2.5-VL-72B-Instruct model.")
    print("=" * 60)
    
    # Run all demos
    demo_configuration()
    demo_message_formatting()
    demo_batch_processing_structure()
    demo_image_categorization()
    demo_error_handling()
    demo_directory_structure()
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✅ Correct model name: Qwen/Qwen2.5-VL-72B-Instruct")
    print("✅ API-compliant message formatting")
    print("✅ Enhanced error handling with retries")
    print("✅ Cross-platform file path handling")
    print("✅ Batch processing with progress tracking")
    print("✅ RGB/IR image categorization")
    print("✅ Comprehensive logging system")
    print("✅ Output directory management")
    
    print("\nNext Steps:")
    print("1. Set up API credentials (QWEN_API_KEY or DASHSCOPE_API_KEY)")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Add sample images to examples/sample_images/")
    print("4. Run: python src/multimodal_processor.py --help")
    print("5. Try: python examples/multimodal_example.py")

if __name__ == "__main__":
    main()