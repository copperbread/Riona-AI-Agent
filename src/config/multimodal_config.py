"""
Configuration settings for multi-modal processing
"""

import os
from pathlib import Path
from typing import Dict, Any, List

class MultiModalConfig:
    """Configuration class for multi-modal processing"""
    
    # Model Configuration
    MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
    API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    
    # Processing Configuration
    MAX_RETRIES = 3
    TIMEOUT = 60
    MAX_BATCH_SIZE = 10
    SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    MAX_IMAGE_SIZE = (2048, 2048)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Image categorization keywords
    RGB_KEYWORDS = ['rgb', 'color', 'visible', 'normal']
    IR_KEYWORDS = ['ir', 'infrared', 'thermal', 'heat']
    
    # Default API parameters
    DEFAULT_API_PARAMS = {
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": False
    }
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_directories(cls) -> Dict[str, Path]:
        """Get all directory paths"""
        return {
            "base": cls.BASE_DIR,
            "logs": cls.LOGS_DIR,
            "output": cls.OUTPUT_DIR,
            "temp": cls.TEMP_DIR
        }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for dir_path in cls.get_directories().values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        if not cls.API_KEY:
            errors.append("API key not found. Set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable.")
        
        if not cls.MODEL_NAME:
            errors.append("Model name not specified.")
        
        if cls.MAX_BATCH_SIZE <= 0:
            errors.append("MAX_BATCH_SIZE must be greater than 0.")
        
        if cls.TIMEOUT <= 0:
            errors.append("TIMEOUT must be greater than 0.")
        
        return errors

# Example prompt templates
PROMPT_TEMPLATES = {
    "image_analysis": "Analyze this {image_type} image and describe what you see in detail.",
    "object_detection": "Identify and list all objects visible in this {image_type} image.",
    "scene_description": "Provide a comprehensive description of the scene in this {image_type} image, including lighting, composition, and any notable features.",
    "comparison": "Compare and contrast the visual information between RGB and IR images if both are provided.",
    "technical_analysis": "Perform a technical analysis of this {image_type} image, focusing on quality, resolution, and any technical aspects."
}

# Example batch processing configurations
BATCH_CONFIGS = {
    "surveillance": {
        "prompt_template": PROMPT_TEMPLATES["object_detection"],
        "api_params": {
            "temperature": 0.3,
            "max_tokens": 500
        },
        "batch_size": 5
    },
    "research": {
        "prompt_template": PROMPT_TEMPLATES["technical_analysis"],
        "api_params": {
            "temperature": 0.1,
            "max_tokens": 1500
        },
        "batch_size": 3
    },
    "general": {
        "prompt_template": PROMPT_TEMPLATES["image_analysis"],
        "api_params": MultiModalConfig.DEFAULT_API_PARAMS,
        "batch_size": MultiModalConfig.MAX_BATCH_SIZE
    }
}