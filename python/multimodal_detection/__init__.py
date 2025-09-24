"""
Multi-Modal Detection Package

A robust Python package for multi-modal object detection using the Qwen2.5-VL-72B-Instruct model.
Supports RGB and IR image processing with comprehensive error handling and batch processing.
"""

from .detector import MultiModalDetector, DetectionResult
from .config import get_config, validate_config, DEFAULT_MODEL_NAME
from .utils import (
    ensure_cross_platform_path,
    find_image_files,
    validate_image_file,
    get_image_info,
    setup_logging
)

__version__ = "1.0.0"
__author__ = "Riona AI Agent Team"
__email__ = "contact@riona-ai.com"
__description__ = "Multi-modal object detection using Qwen2.5-VL-72B-Instruct"

__all__ = [
    'MultiModalDetector',
    'DetectionResult',
    'get_config',
    'validate_config',
    'DEFAULT_MODEL_NAME',
    'ensure_cross_platform_path',
    'find_image_files',
    'validate_image_file',
    'get_image_info',
    'setup_logging'
]