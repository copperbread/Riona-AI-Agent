#!/usr/bin/env python3
"""
Demo script for Multi-Modal Detection functionality
This script demonstrates the features without requiring actual API calls
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from detector import MultiModalDetector, DetectionResult
from config import get_config, validate_config
from utils import (
    setup_logging,
    ensure_cross_platform_path,
    validate_directory_structure,
    save_json_safely,
    normalize_bbox,
    denormalize_bbox,
    calculate_iou
)


def create_sample_detection_result():
    """Create a sample detection result for demonstration"""
    return DetectionResult(
        image_path="demo/sample_image.jpg",
        detections=[
            {
                "class": "person",
                "confidence": 0.95,
                "bbox": [0.5, 0.3, 0.2, 0.4],
                "attributes": {"color": "blue", "size": "medium"}
            },
            {
                "class": "car",
                "confidence": 0.87,
                "bbox": [0.7, 0.6, 0.25, 0.15],
                "attributes": {"color": "red", "type": "sedan"}
            },
            {
                "class": "bicycle",
                "confidence": 0.72,
                "bbox": [0.2, 0.7, 0.15, 0.2],
                "attributes": {"color": "green"}
            }
        ],
        confidence_scores=[0.95, 0.87, 0.72],
        processing_time=2.3
    )


def demo_detection_result():
    """Demonstrate DetectionResult functionality"""
    print("=== DetectionResult Demo ===")
    
    result = create_sample_detection_result()
    
    print(f"Image: {result.image_path}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Number of detections: {len(result.detections)}")
    print(f"Average confidence: {sum(result.confidence_scores) / len(result.confidence_scores):.3f}")
    
    print("\nDetections:")
    for i, detection in enumerate(result.detections):
        print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
        bbox = detection['bbox']
        print(f"      Location: center=({bbox[0]:.2f}, {bbox[1]:.2f}), size=({bbox[2]:.2f}, {bbox[3]:.2f})")
        if 'attributes' in detection:
            attrs = ", ".join([f"{k}={v}" for k, v in detection['attributes'].items()])
            print(f"      Attributes: {attrs}")
    
    print()


def demo_bbox_operations():
    """Demonstrate bounding box operations"""
    print("=== Bounding Box Operations Demo ===")
    
    # Original bbox in pixels
    bbox_pixels = [320, 240, 100, 80]  # x_center, y_center, width, height
    img_width, img_height = 640, 480
    
    print(f"Original bbox (pixels): center=({bbox_pixels[0]}, {bbox_pixels[1]}), size=({bbox_pixels[2]}, {bbox_pixels[3]})")
    print(f"Image size: {img_width} x {img_height}")
    
    # Normalize
    normalized = normalize_bbox(bbox_pixels, img_width, img_height)
    print(f"Normalized bbox: center=({normalized[0]:.3f}, {normalized[1]:.3f}), size=({normalized[2]:.3f}, {normalized[3]:.3f})")
    
    # Denormalize back to corner coordinates
    corners = denormalize_bbox(normalized, img_width, img_height)
    print(f"Corner coordinates: top-left=({corners[0]}, {corners[1]}), bottom-right=({corners[2]}, {corners[3]})")
    
    # Calculate IoU between two boxes
    box1 = [0, 0, 100, 100]    # x1, y1, x2, y2
    box2 = [50, 50, 150, 150]  # x1, y1, x2, y2
    iou = calculate_iou(box1, box2)
    print(f"\nIoU between overlapping boxes: {iou:.3f}")
    
    print()


def demo_file_operations():
    """Demonstrate file operations"""
    print("=== File Operations Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Working in temporary directory: {temp_path}")
        
        # Demonstrate directory structure creation
        dirs = validate_directory_structure(temp_path / "detection_project", create_missing=True)
        print(f"Created directory structure:")
        for name, path in dirs.items():
            print(f"  {name}: {path}")
        
        # Demonstrate JSON saving
        sample_data = {
            "project": "multimodal_detection",
            "version": "1.0.0",
            "detections": [
                {"class": "person", "confidence": 0.95},
                {"class": "car", "confidence": 0.87}
            ]
        }
        
        json_file = dirs['results'] / "sample_results.json"
        success = save_json_safely(sample_data, json_file)
        print(f"\nJSON save {'successful' if success else 'failed'}: {json_file}")
        
        if success and json_file.exists():
            print(f"File size: {json_file.stat().st_size} bytes")
    
    print()


def demo_detector_initialization():
    """Demonstrate detector initialization with various configurations"""
    print("=== Detector Initialization Demo ===")
    
    # Test configuration validation
    test_configs = [
        {
            "name": "Valid Configuration",
            "config": {
                'api_key': 'demo_key_12345',
                'api_url': 'https://api.example.com/v1/chat/completions',
                'model_name': 'Qwen/Qwen2.5-VL-72B-Instruct',
                'confidence_threshold': 0.3,
                'max_batch_size': 50,
                'timeout': 60
            }
        },
        {
            "name": "Invalid Configuration (missing API key)",
            "config": {
                'api_url': 'https://api.example.com',
                'model_name': 'some_model',
                'confidence_threshold': 1.5,  # Invalid threshold
                'max_batch_size': -1,         # Invalid batch size
                'timeout': 0                  # Invalid timeout
            }
        }
    ]
    
    for test_case in test_configs:
        print(f"\nTesting: {test_case['name']}")
        errors = validate_config(test_case['config'])
        
        if errors:
            print("  Configuration errors found:")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  ✓ Configuration is valid")
            
            # Try to initialize detector
            try:
                detector = MultiModalDetector(
                    api_key=test_case['config']['api_key'],
                    api_url=test_case['config']['api_url'],
                    model_name=test_case['config']['model_name']
                )
                print(f"  ✓ Detector initialized successfully")
                print(f"    Model: {detector.model_name}")
                print(f"    API URL: {detector.api_url}")
            except Exception as e:
                print(f"  ✗ Detector initialization failed: {e}")
    
    print()


def demo_mock_api_interaction():
    """Demonstrate API interaction with mocking"""
    print("=== Mock API Interaction Demo ===")
    
    # Mock API response
    mock_response_data = {
        'choices': [
            {
                'message': {
                    'content': json.dumps({
                        'detections': [
                            {
                                'class': 'person',
                                'confidence': 0.92,
                                'bbox': [0.4, 0.3, 0.2, 0.5],
                                'attributes': {'clothing': 'casual', 'posture': 'standing'}
                            },
                            {
                                'class': 'dog',
                                'confidence': 0.88,
                                'bbox': [0.6, 0.7, 0.15, 0.2],
                                'attributes': {'breed': 'labrador', 'action': 'sitting'}
                            }
                        ],
                        'image_info': {
                            'width': 1024,
                            'height': 768,
                            'type': 'RGB'
                        }
                    })
                }
            }
        ]
    }
    
    print("Mock API Response:")
    print(json.dumps(mock_response_data, indent=2))
    
    # Simulate processing the response
    try:
        content = mock_response_data['choices'][0]['message']['content']
        detection_data = json.loads(content)
        detections = detection_data.get('detections', [])
        
        print(f"\nParsed {len(detections)} detections:")
        for detection in detections:
            print(f"  - {detection['class']}: {detection['confidence']:.2f}")
            bbox = detection['bbox']
            print(f"    Location: ({bbox[0]:.2f}, {bbox[1]:.2f}) size: ({bbox[2]:.2f}, {bbox[3]:.2f})")
    
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
    
    print()


def demo_error_handling():
    """Demonstrate error handling scenarios"""
    print("=== Error Handling Demo ===")
    
    error_scenarios = [
        {
            "name": "File Not Found",
            "error": FileNotFoundError("Image file not found: /nonexistent/image.jpg"),
            "context": "Loading image file"
        },
        {
            "name": "Invalid Image Format",
            "error": ValueError("Unsupported image format: .xyz"),
            "context": "Validating image format"
        },
        {
            "name": "API Authentication Error",
            "error": Exception("Authentication failed. Please check your API key. Status: 401"),
            "context": "Making API request"
        },
        {
            "name": "Network Timeout",
            "error": TimeoutError("API request timed out after 60 seconds"),
            "context": "API communication"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Context: {scenario['context']}")
        print(f"Error Type: {type(scenario['error']).__name__}")
        print(f"Error Message: {scenario['error']}")
        
        # Demonstrate how the system would create a DetectionResult with error
        error_result = DetectionResult(
            image_path="error_test.jpg",
            detections=[],
            confidence_scores=[],
            processing_time=0.0,
            error=str(scenario['error'])
        )
        
        print(f"Error Result: {error_result.error}")
    
    print()


def demo_cross_platform_compatibility():
    """Demonstrate cross-platform path handling"""
    print("=== Cross-Platform Compatibility Demo ===")
    
    test_paths = [
        "/home/user/images/photo.jpg",      # Unix-style
        "C:\\Users\\User\\Images\\photo.jpg", # Windows-style
        "images/subfolder/photo.jpg",       # Relative path
        "~/Documents/photos/image.png"      # Home directory
    ]
    
    for path_str in test_paths:
        try:
            normalized_path = ensure_cross_platform_path(path_str)
            print(f"Original: {path_str}")
            print(f"Normalized: {normalized_path}")
            print(f"Is Absolute: {normalized_path.is_absolute()}")
            print(f"Parts: {normalized_path.parts}")
            print()
        except Exception as e:
            print(f"Error processing {path_str}: {e}")
            print()


def main():
    """Run all demonstrations"""
    # Setup logging
    logger = setup_logging(level=20)  # INFO level
    
    print("Multi-Modal Detection System - Feature Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Run all demonstrations
        demo_detection_result()
        demo_bbox_operations()
        demo_file_operations()
        demo_detector_initialization()
        demo_mock_api_interaction()
        demo_error_handling()
        demo_cross_platform_compatibility()
        
        print("=" * 60)
        print("All demonstrations completed successfully!")
        print("\nNext Steps:")
        print("1. Set your API key: export QWEN_API_KEY='your_actual_api_key'")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run example_usage.py for real usage examples")
        print("4. Use detector.py from command line for actual detection")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()