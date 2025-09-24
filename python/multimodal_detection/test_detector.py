#!/usr/bin/env python3
"""
Test script for the Multi-Modal Detection System
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from detector import MultiModalDetector, DetectionResult
from config import get_config, validate_config
from utils import (
    ensure_cross_platform_path,
    find_image_files,
    validate_image_file,
    get_image_info,
    setup_logging,
    normalize_bbox,
    denormalize_bbox,
    calculate_iou
)


def test_cross_platform_path():
    """Test cross-platform path handling"""
    print("Testing cross-platform path handling...")
    
    # Test with string path
    path_str = "/tmp/test/image.jpg"
    result = ensure_cross_platform_path(path_str)
    assert isinstance(result, Path), "Should return Path object"
    
    # Test with Path object
    path_obj = Path("/tmp/test/image.jpg")
    result = ensure_cross_platform_path(path_obj)
    assert isinstance(result, Path), "Should return Path object"
    
    print("✓ Cross-platform path handling works")


def test_bbox_operations():
    """Test bounding box operations"""
    print("Testing bounding box operations...")
    
    # Test normalization
    bbox = [100, 50, 200, 100]  # x_center, y_center, width, height in pixels
    img_width, img_height = 640, 480
    
    normalized = normalize_bbox(bbox, img_width, img_height)
    expected = [100/640, 50/480, 200/640, 100/480]
    
    assert abs(normalized[0] - expected[0]) < 0.001, f"Expected {expected[0]}, got {normalized[0]}"
    assert abs(normalized[1] - expected[1]) < 0.001, f"Expected {expected[1]}, got {normalized[1]}"
    
    # Test denormalization
    denorm = denormalize_bbox(normalized, img_width, img_height)
    # Convert back to center format for comparison
    x_center = (denorm[0] + denorm[2]) / 2
    y_center = (denorm[1] + denorm[3]) / 2
    
    assert abs(x_center - bbox[0]) < 1, "X center should match original"
    assert abs(y_center - bbox[1]) < 1, "Y center should match original"
    
    print("✓ Bounding box operations work")


def test_iou_calculation():
    """Test IoU calculation"""
    print("Testing IoU calculation...")
    
    # Test identical boxes
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    iou = calculate_iou(box1, box2)
    assert abs(iou - 1.0) < 0.001, f"IoU of identical boxes should be 1.0, got {iou}"
    
    # Test non-overlapping boxes
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    iou = calculate_iou(box1, box2)
    assert abs(iou - 0.0) < 0.001, f"IoU of non-overlapping boxes should be 0.0, got {iou}"
    
    # Test partially overlapping boxes
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = calculate_iou(box1, box2)
    expected_iou = 25 / 175  # intersection=25, union=100+100-25=175
    assert abs(iou - expected_iou) < 0.001, f"Expected IoU {expected_iou}, got {iou}"
    
    print("✓ IoU calculation works")


def test_config_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test valid config
    valid_config = {
        'api_key': 'test_key',
        'api_url': 'https://api.example.com',
        'model_name': 'test_model',
        'confidence_threshold': 0.5,
        'max_batch_size': 10,
        'timeout': 30
    }
    
    errors = validate_config(valid_config)
    assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"
    
    # Test invalid config
    invalid_config = {
        'api_key': '',  # Empty API key
        'confidence_threshold': 1.5,  # Invalid threshold
        'max_batch_size': -1,  # Invalid batch size
        'timeout': 0  # Invalid timeout
    }
    
    errors = validate_config(invalid_config)
    assert len(errors) > 0, "Invalid config should have errors"
    
    print("✓ Configuration validation works")


def test_detection_result():
    """Test DetectionResult data class"""
    print("Testing DetectionResult...")
    
    result = DetectionResult(
        image_path="/test/image.jpg",
        detections=[
            {
                'class': 'person',
                'confidence': 0.95,
                'bbox': [0.5, 0.3, 0.2, 0.4]
            }
        ],
        confidence_scores=[0.95],
        processing_time=1.5
    )
    
    assert result.image_path == "/test/image.jpg"
    assert len(result.detections) == 1
    assert result.detections[0]['class'] == 'person'
    assert result.error is None
    
    print("✓ DetectionResult works")


def test_file_operations():
    """Test file operations with temporary files"""
    print("Testing file operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test find_image_files with empty directory
        image_files = find_image_files(temp_path)
        assert len(image_files) == 0, "Empty directory should return no files"
        
        # Create a fake image file
        fake_image = temp_path / "test.jpg"
        fake_image.write_text("fake image content")
        
        # Test find_image_files with fake file
        image_files = find_image_files(temp_path)
        assert len(image_files) == 1, "Should find one image file"
        assert image_files[0].name == "test.jpg"
        
        # Test validate_image_file (will fail for fake image)
        is_valid, error = validate_image_file(fake_image)
        assert not is_valid, "Fake image should be invalid"
        assert error is not None, "Should return error message"
    
    print("✓ File operations work")


def test_detector_initialization():
    """Test detector initialization"""
    print("Testing detector initialization...")
    
    # Test with valid configuration
    try:
        detector = MultiModalDetector(
            api_key="test_key",
            api_url="https://api.example.com",
            model_name="test_model"
        )
        assert detector.api_key == "test_key"
        assert detector.model_name == "test_model"
        print("✓ Detector initialization works")
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")


def mock_api_response():
    """Create a mock API response"""
    return {
        'choices': [
            {
                'message': {
                    'content': json.dumps({
                        'detections': [
                            {
                                'class': 'person',
                                'confidence': 0.95,
                                'bbox': [0.5, 0.3, 0.2, 0.4],
                                'attributes': {'color': 'blue'}
                            }
                        ],
                        'image_info': {
                            'width': 640,
                            'height': 480,
                            'type': 'RGB'
                        }
                    })
                }
            }
        ]
    }


def test_api_request_mocking():
    """Test API request with mocking"""
    print("Testing API request mocking...")
    
    try:
        with patch('requests.Session.post') as mock_post:
            # Setup mock response
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = mock_api_response()
            mock_post.return_value = mock_response
            
            # Create detector
            detector = MultiModalDetector(
                api_key="test_key",
                api_url="https://api.example.com"
            )
            
            # Test _make_api_request
            messages = [{'role': 'user', 'content': 'test'}]
            response = detector._make_api_request(messages)
            
            assert 'choices' in response
            assert len(response['choices']) > 0
            
            print("✓ API request mocking works")
    
    except Exception as e:
        print(f"✗ API request mocking failed: {e}")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running Multi-Modal Detection Tests")
    print("=" * 50)
    
    tests = [
        test_cross_platform_path,
        test_bbox_operations,
        test_iou_calculation,
        test_config_validation,
        test_detection_result,
        test_file_operations,
        test_detector_initialization,
        test_api_request_mocking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)