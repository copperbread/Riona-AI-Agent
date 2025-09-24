#!/usr/bin/env python3
"""
Test script for multi-modal processor functionality
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from multimodal_processor import (
            ProcessingConfig, 
            QwenMultiModalClient, 
            BatchProcessor,
            MultiModalLogger,
            PathHandler,
            ImageProcessor
        )
        from config.multimodal_config import MultiModalConfig, PROMPT_TEMPLATES, BATCH_CONFIGS
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration validation"""
    print("Testing configuration...")
    
    try:
        from config.multimodal_config import MultiModalConfig
        
        # Test directory creation
        MultiModalConfig.ensure_directories()
        
        # Test configuration validation
        errors = MultiModalConfig.validate_config()
        if errors:
            print(f"⚠️  Configuration warnings: {errors}")
        else:
            print("✅ Configuration validation passed")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_path_handler():
    """Test cross-platform path handling"""
    print("Testing path handler...")
    
    try:
        from multimodal_processor import PathHandler
        
        # Test path normalization
        test_path = PathHandler.normalize_path("./test/path")
        assert isinstance(test_path, Path)
        
        # Test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = PathHandler.ensure_directory(Path(temp_dir) / "test_subdir")
            assert test_dir.exists()
        
        # Test file extension
        ext = PathHandler.get_file_extension("test.JPG")
        assert ext == ".jpg"
        
        # Test image validation
        is_valid = PathHandler.validate_image_file("test.jpg")
        assert is_valid == True
        
        is_invalid = PathHandler.validate_image_file("test.txt")
        assert is_invalid == False
        
        print("✅ Path handler tests passed")
        return True
    except Exception as e:
        print(f"❌ Path handler test failed: {e}")
        return False

def test_logger():
    """Test logging functionality"""
    print("Testing logger...")
    
    try:
        from multimodal_processor import MultiModalLogger
        
        logger = MultiModalLogger("DEBUG")
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        # Check if log directory was created
        log_dir = Path(__file__).parent / "src" / "logs"
        if log_dir.exists():
            print("✅ Logger tests passed")
            return True
        else:
            print("⚠️  Log directory not created, but logger initialized")
            return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False

def test_processing_config():
    """Test processing configuration"""
    print("Testing processing configuration...")
    
    try:
        from multimodal_processor import ProcessingConfig
        
        config = ProcessingConfig()
        
        # Test default values
        assert config.model_name == "Qwen/Qwen2.5-VL-72B-Instruct"
        assert config.max_retries > 0
        assert config.timeout > 0
        assert config.max_batch_size > 0
        assert len(config.supported_image_formats) > 0
        
        print("✅ Processing configuration tests passed")
        return True
    except Exception as e:
        print(f"❌ Processing configuration test failed: {e}")
        return False

def test_message_formatting():
    """Test API message formatting without making actual API calls"""
    print("Testing message formatting...")
    
    try:
        from multimodal_processor import ProcessingConfig, QwenMultiModalClient
        from unittest.mock import Mock
        import tempfile
        from PIL import Image
        
        # Create a small test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(temp_file.name, 'JPEG')
            temp_image_path = temp_file.name
        
        try:
            config = ProcessingConfig()
            
            # Mock the API key to avoid requiring real credentials for testing
            original_getenv = os.getenv
            os.getenv = Mock(return_value="test_api_key")
            
            client = QwenMultiModalClient(config)
            
            # Test message content creation
            content = client.create_message_content(
                text="Test prompt",
                image_paths=[temp_image_path]
            )
            
            # Validate message structure
            assert len(content) == 2  # text + image
            assert content[0]['type'] == 'text'
            assert content[0]['text'] == 'Test prompt'
            assert content[1]['type'] == 'image_url'
            assert 'image_url' in content[1]
            assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
            
            print("✅ Message formatting tests passed")
            return True
            
        finally:
            # Cleanup
            os.getenv = original_getenv
            os.unlink(temp_image_path)
            
    except Exception as e:
        print(f"❌ Message formatting test failed: {e}")
        return False

def test_batch_processing_structure():
    """Test batch processing structure without API calls"""
    print("Testing batch processing structure...")
    
    try:
        from multimodal_processor import ProcessingConfig, BatchProcessor
        from unittest.mock import Mock
        
        config = ProcessingConfig()
        
        # Mock the API key
        original_getenv = os.getenv
        os.getenv = Mock(return_value="test_api_key")
        
        try:
            processor = BatchProcessor(config)
            
            # Test image categorization
            test_paths = [
                Path("test_rgb_image.jpg"),
                Path("test_ir_thermal.jpg"),
                Path("unknown_image.jpg")
            ]
            
            categories = processor.categorize_images(test_paths)
            
            assert 'rgb' in categories
            assert 'ir' in categories
            assert 'unknown' in categories
            assert len(categories['rgb']) == 1
            assert len(categories['ir']) == 1
            assert len(categories['unknown']) == 1
            
            print("✅ Batch processing structure tests passed")
            return True
            
        finally:
            os.getenv = original_getenv
            
    except Exception as e:
        print(f"❌ Batch processing structure test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Multi-Modal Processor Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_path_handler,
        test_logger,
        test_processing_config,
        test_message_formatting,
        test_batch_processing_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)