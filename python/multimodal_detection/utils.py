"""
Utility functions for multi-modal detection
"""

import os
import json
import platform
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import logging
import hashlib
import mimetypes

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available in utils module.")
    np = None

try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow (PIL) not available in utils module.")
    Image = None

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available in utils module.")
    cv2 = None


def ensure_cross_platform_path(path: Union[str, Path]) -> Path:
    """
    Ensure path works across different platforms (Windows, Linux, macOS)
    
    Args:
        path: Input path as string or Path object
        
    Returns:
        Normalized Path object
    """
    path = Path(path)
    
    # Resolve any symlinks and make absolute
    try:
        path = path.resolve()
    except (OSError, RuntimeError):
        # If resolve fails (e.g., broken symlink), use as-is
        pass
    
    return path


def validate_directory_structure(base_dir: Union[str, Path], 
                                create_missing: bool = True) -> Dict[str, Path]:
    """
    Validate and optionally create directory structure
    
    Args:
        base_dir: Base directory path
        create_missing: Whether to create missing directories
        
    Returns:
        Dictionary of directory paths
    """
    base_dir = ensure_cross_platform_path(base_dir)
    
    directories = {
        'base': base_dir,
        'logs': base_dir / 'logs',
        'results': base_dir / 'results',
        'visualizations': base_dir / 'results' / 'visualizations',
        'yolo_annotations': base_dir / 'results' / 'yolo_annotations',
        'intermediate': base_dir / 'results' / 'intermediate'
    }
    
    if create_missing:
        for name, dir_path in directories.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logging.warning(f"Permission denied creating directory: {dir_path}")
            except OSError as e:
                logging.error(f"Error creating directory {dir_path}: {e}")
    
    return directories


def find_image_files(directory: Union[str, Path], 
                    supported_formats: set = None,
                    recursive: bool = True) -> List[Path]:
    """
    Find all image files in a directory
    
    Args:
        directory: Directory to search
        supported_formats: Set of supported file extensions
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    if supported_formats is None:
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    directory = ensure_cross_platform_path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    image_files = []
    
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for ext in supported_formats:
        # Search for both lowercase and uppercase extensions
        image_files.extend(directory.glob(f'{pattern}{ext}'))
        image_files.extend(directory.glob(f'{pattern}{ext.upper()}'))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files


def validate_image_file(image_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        image_path = ensure_cross_platform_path(image_path)
        
        if not image_path.exists():
            return False, f"File does not exist: {image_path}"
        
        if not image_path.is_file():
            return False, f"Path is not a file: {image_path}"
        
        # Try to open with PIL to validate
        with Image.open(image_path) as img:
            # Verify image can be loaded
            img.verify()
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about an image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    image_path = ensure_cross_platform_path(image_path)
    
    info = {
        'path': str(image_path),
        'filename': image_path.name,
        'size_bytes': 0,
        'width': 0,
        'height': 0,
        'channels': 0,
        'format': None,
        'mode': None,
        'mime_type': None
    }
    
    try:
        # File size
        info['size_bytes'] = image_path.stat().st_size
        
        # MIME type
        info['mime_type'] = mimetypes.guess_type(str(image_path))[0]
        
        # Image properties
        with Image.open(image_path) as img:
            info['width'], info['height'] = img.size
            info['format'] = img.format
            info['mode'] = img.mode
            
            # Estimate channels
            if img.mode == 'L':
                info['channels'] = 1
            elif img.mode in ['RGB', 'HSV']:
                info['channels'] = 3
            elif img.mode in ['RGBA', 'CMYK']:
                info['channels'] = 4
            else:
                info['channels'] = len(img.getbands()) if hasattr(img, 'getbands') else 0
                
    except Exception as e:
        info['error'] = str(e)
    
    return info


def create_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Create hash of a file for duplicate detection
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    file_path = ensure_cross_platform_path(file_path)
    
    hash_func = getattr(hashlib, algorithm)()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
    except Exception as e:
        return f"error_{str(e)}"
    
    return hash_func.hexdigest()


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range
    
    Args:
        bbox: Bounding box as [x_center, y_center, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Normalized bounding box coordinates
    """
    return [
        bbox[0] / img_width,   # x_center
        bbox[1] / img_height,  # y_center
        bbox[2] / img_width,   # width
        bbox[3] / img_height   # height
    ]


def denormalize_bbox(normalized_bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Convert normalized bounding box to pixel coordinates
    
    Args:
        normalized_bbox: Normalized bbox as [x_center, y_center, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Pixel coordinates as [x1, y1, x2, y2]
    """
    x_center = normalized_bbox[0] * img_width
    y_center = normalized_bbox[1] * img_height
    width = normalized_bbox[2] * img_width
    height = normalized_bbox[3] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return [x1, y1, x2, y2]


def filter_detections_by_confidence(detections: List[Dict[str, Any]], 
                                  threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Filter detections by confidence threshold
    
    Args:
        detections: List of detection dictionaries
        threshold: Minimum confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [det for det in detections if det.get('confidence', 0.0) >= threshold]


def save_json_safely(data: Any, file_path: Union[str, Path], 
                    backup: bool = True) -> bool:
    """
    Safely save data to JSON file with backup option
    
    Args:
        data: Data to save
        file_path: Output file path
        backup: Whether to create backup if file exists
        
    Returns:
        True if successful, False otherwise
    """
    file_path = ensure_cross_platform_path(file_path)
    
    try:
        # Create backup if requested and file exists
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
            file_path.rename(backup_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def load_json_safely(file_path: Union[str, Path]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Safely load JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (data, error_message)
    """
    file_path = ensure_cross_platform_path(file_path)
    
    try:
        if not file_path.exists():
            return None, f"File does not exist: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data, None
        
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format: {e}"
    except Exception as e:
        return None, f"Error loading file: {e}"


def get_system_info() -> Dict[str, str]:
    """Get system information for debugging"""
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cwd': str(Path.cwd())
    }


def setup_logging(log_file: Union[str, Path] = None, 
                 level: int = logging.INFO,
                 console: bool = True) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('multimodal_detection')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = ensure_cross_platform_path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1: First bounding box as [x1, y1, x2, y2]
        box2: Second bounding box as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union