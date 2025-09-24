"""
Configuration file for Multi-Modal Detection Script
"""

import os
from pathlib import Path
from typing import Dict, List

# API Configuration
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Default detection classes (can be expanded based on model capabilities)
DEFAULT_DETECTION_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Color palette for visualization
DETECTION_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#C0C0C0', '#808080', '#9999FF', '#993366', '#FFFFCC', '#CCFFFF',
    '#660066', '#FF8080', '#0066CC', '#CCCCFF', '#000080', '#FF00FF',
    '#FFFF00', '#00FFFF', '#800080', '#800000', '#008080', '#0000FF'
]

# Confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7

# Processing configuration
MAX_BATCH_SIZE = 100
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3

# Output configuration
DEFAULT_OUTPUT_FORMAT = 'json'
YOLO_CLASS_MAPPING = {cls: idx for idx, cls in enumerate(DEFAULT_DETECTION_CLASSES)}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
CONSOLE_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_config() -> Dict:
    """Get configuration dictionary with environment variable overrides"""
    return {
        'api_key': os.getenv('QWEN_API_KEY'),
        'api_url': os.getenv('QWEN_API_URL', DEFAULT_API_URL),
        'model_name': os.getenv('QWEN_MODEL_NAME', DEFAULT_MODEL_NAME),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', DEFAULT_CONFIDENCE_THRESHOLD)),
        'max_batch_size': int(os.getenv('MAX_BATCH_SIZE', MAX_BATCH_SIZE)),
        'timeout': int(os.getenv('API_TIMEOUT', DEFAULT_TIMEOUT)),
        'max_retries': int(os.getenv('MAX_RETRIES', MAX_RETRIES)),
    }

def validate_config(config: Dict) -> List[str]:
    """Validate configuration and return list of errors"""
    errors = []
    
    if not config.get('api_key'):
        errors.append("API key is required (QWEN_API_KEY environment variable)")
    
    if not config.get('api_url'):
        errors.append("API URL is required")
    
    if not config.get('model_name'):
        errors.append("Model name is required")
    
    if config.get('confidence_threshold', 0) < 0 or config.get('confidence_threshold', 0) > 1:
        errors.append("Confidence threshold must be between 0 and 1")
    
    if config.get('max_batch_size', 0) <= 0:
        errors.append("Max batch size must be greater than 0")
    
    if config.get('timeout', 0) <= 0:
        errors.append("Timeout must be greater than 0")
    
    return errors