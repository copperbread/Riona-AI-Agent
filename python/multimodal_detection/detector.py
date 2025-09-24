#!/usr/bin/env python3
"""
Multi-Modal Detection Script using Qwen2.5-VL-72B-Instruct
Supports RGB and IR image processing with YOLO-format outputs
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import base64
from io import BytesIO

try:
    import requests
except ImportError:
    print("Warning: requests not available. API functionality will be limited.")
    requests = None

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available. Some numerical operations may not work.")
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Warning: Pillow (PIL) not available. Image processing will be limited.")
    Image = ImageDraw = ImageFont = None

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available. Advanced image processing features disabled.")
    cv2 = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Plotting features disabled.")
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars disabled.")
    # Fallback tqdm implementation
    def tqdm(iterable, desc="", **kwargs):
        print(f"{desc}...")
        for i, item in enumerate(iterable):
            if i % max(1, len(list(iterable)) // 10) == 0:
                print(f"Progress: {i}/{len(list(iterable))}")
            yield item


@dataclass
class DetectionResult:
    """Data class for detection results"""
    image_path: str
    detections: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float
    error: Optional[str] = None


class MultiModalDetector:
    """
    Multi-modal detection class using Qwen2.5-VL-72B-Instruct API
    Handles RGB and IR images with robust error handling and batch processing
    """
    
    def __init__(self, api_key: str = None, api_url: str = None, 
                 model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        """
        Initialize the detector with API credentials and configuration
        
        Args:
            api_key: API key for the Qwen model service
            api_url: Base URL for the API endpoint
            model_name: Model name to use for detection
        """
        self.api_key = api_key or os.getenv('QWEN_API_KEY')
        self.api_url = api_url or os.getenv('QWEN_API_URL', 'https://api.openai.com/v1/chat/completions')
        self.model_name = model_name
        self.session = requests.Session()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Validate initialization
        self._validate_initialization()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging"""
        logger = logging.getLogger('multimodal_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / 'multimodal_detection.log')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _validate_initialization(self):
        """Validate API credentials and configuration"""
        if not self.api_key:
            raise ValueError(
                "API key is required. Set QWEN_API_KEY environment variable or pass api_key parameter."
            )
        
        if not self.api_url:
            raise ValueError(
                "API URL is required. Set QWEN_API_URL environment variable or pass api_url parameter."
            )
        
        self.logger.info(f"Initialized MultiModalDetector with model: {self.model_name}")
    
    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string for API transmission
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Get MIME type
            mime_type = self._get_mime_type(image_path)
            return f"data:{mime_type};base64,{base64_string}"
            
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type for image file"""
        extension = image_path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def _validate_image_file(self, image_path: Union[str, Path]) -> Path:
        """
        Validate image file exists and has supported format
        
        Args:
            image_path: Path to image file
            
        Returns:
            Validated Path object
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        return image_path
    
    def _create_detection_prompt(self, image_type: str = "RGB") -> str:
        """Create detection prompt for the model"""
        return f"""
        Analyze this {image_type} image and detect all objects, people, vehicles, and other significant items.
        
        For each detection, provide:
        1. Object class/category
        2. Confidence score (0-1)
        3. Bounding box coordinates in normalized format (x_center, y_center, width, height) where values are between 0 and 1
        4. Additional attributes if relevant (color, size, orientation, etc.)
        
        Return the results in JSON format with the following structure:
        {{
            "detections": [
                {{
                    "class": "object_class",
                    "confidence": 0.95,
                    "bbox": [x_center, y_center, width, height],
                    "attributes": {{"color": "red", "size": "large"}}
                }}
            ],
            "image_info": {{
                "width": image_width,
                "height": image_height,
                "type": "{image_type}"
            }}
        }}
        """
    
    def _make_api_request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make API request to Qwen model with robust error handling
        
        Args:
            messages: List of message objects for the API
            
        Returns:
            API response as dictionary
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        try:
            self.logger.debug(f"Making API request to {self.api_url}")
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Handle different types of API errors
            if response.status_code == 401:
                raise requests.exceptions.HTTPError(
                    f"Authentication failed. Please check your API key. Status: {response.status_code}"
                )
            elif response.status_code == 429:
                raise requests.exceptions.HTTPError(
                    f"Rate limit exceeded. Please wait before making more requests. Status: {response.status_code}"
                )
            elif response.status_code == 500:
                raise requests.exceptions.HTTPError(
                    f"Internal server error. The API service may be temporarily unavailable. Status: {response.status_code}"
                )
            elif not response.ok:
                raise requests.exceptions.HTTPError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
            
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error("API request timed out after 60 seconds")
            raise
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response as JSON: {str(e)}")
            raise
    
    def detect_objects(self, image_path: Union[str, Path], 
                      image_type: str = "RGB") -> DetectionResult:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to the image file
            image_type: Type of image ("RGB" or "IR")
            
        Returns:
            DetectionResult object with detection information
        """
        import time
        start_time = time.time()
        
        try:
            # Validate image file
            validated_path = self._validate_image_file(image_path)
            self.logger.info(f"Processing {image_type} image: {validated_path}")
            
            # Encode image
            base64_image = self._encode_image_to_base64(validated_path)
            
            # Create prompt
            prompt = self._create_detection_prompt(image_type)
            
            # Prepare messages for API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ]
            
            # Make API request
            response = self._make_api_request(messages)
            
            # Parse response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            try:
                # Try to parse JSON response
                detection_data = json.loads(content)
                detections = detection_data.get('detections', [])
                confidence_scores = [det.get('confidence', 0.0) for det in detections]
                
                processing_time = time.time() - start_time
                
                self.logger.info(
                    f"Successfully processed {validated_path}: "
                    f"{len(detections)} detections in {processing_time:.2f}s"
                )
                
                return DetectionResult(
                    image_path=str(validated_path),
                    detections=detections,
                    confidence_scores=confidence_scores,
                    processing_time=processing_time
                )
                
            except json.JSONDecodeError:
                # If response is not JSON, treat as error
                self.logger.warning(f"Non-JSON response from API: {content}")
                return DetectionResult(
                    image_path=str(validated_path),
                    detections=[],
                    confidence_scores=[],
                    processing_time=time.time() - start_time,
                    error=f"Invalid response format: {content}"
                )
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {image_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
            return DetectionResult(
                image_path=str(image_path),
                detections=[],
                confidence_scores=[],
                processing_time=processing_time,
                error=error_msg
            )
    
    def batch_detect(self, image_paths: List[Union[str, Path]], 
                    output_dir: Union[str, Path] = None,
                    save_intermediate: bool = True) -> List[DetectionResult]:
        """
        Process multiple images in batch with progress tracking
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save intermediate results
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of DetectionResult objects
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Determine image type based on filename or directory
                image_type = self._determine_image_type(image_path)
                
                # Process image
                result = self.detect_objects(image_path, image_type)
                results.append(result)
                
                # Save intermediate results if requested
                if save_intermediate and output_dir:
                    intermediate_file = output_dir / f"result_{i:04d}.json"
                    self._save_result_json(result, intermediate_file)
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append(DetectionResult(
                    image_path=str(image_path),
                    detections=[],
                    confidence_scores=[],
                    processing_time=0.0,
                    error=str(e)
                ))
        
        # Save final batch results
        if output_dir:
            batch_results_file = output_dir / "batch_results.json"
            self._save_batch_results(results, batch_results_file)
        
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        self.logger.info(
            f"Batch processing completed: {len(successful_results)} successful, "
            f"{len(failed_results)} failed"
        )
        
        return results
    
    def _determine_image_type(self, image_path: Union[str, Path]) -> str:
        """Determine if image is RGB or IR based on filename or directory"""
        path_str = str(image_path).lower()
        if 'ir' in path_str or 'thermal' in path_str or 'infrared' in path_str:
            return "IR"
        return "RGB"
    
    def _save_result_json(self, result: DetectionResult, output_file: Path):
        """Save detection result to JSON file"""
        try:
            result_dict = {
                'image_path': result.image_path,
                'detections': result.detections,
                'confidence_scores': result.confidence_scores,
                'processing_time': result.processing_time,
                'error': result.error
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save result to {output_file}: {str(e)}")
    
    def _save_batch_results(self, results: List[DetectionResult], output_file: Path):
        """Save batch results to JSON file"""
        try:
            batch_dict = {
                'total_images': len(results),
                'successful': len([r for r in results if r.error is None]),
                'failed': len([r for r in results if r.error is not None]),
                'results': [
                    {
                        'image_path': r.image_path,
                        'detections': r.detections,
                        'confidence_scores': r.confidence_scores,
                        'processing_time': r.processing_time,
                        'error': r.error
                    } for r in results
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Batch results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch results to {output_file}: {str(e)}")
    
    def save_yolo_annotations(self, result: DetectionResult, output_dir: Union[str, Path]):
        """
        Save detections in YOLO format
        
        Args:
            result: DetectionResult object
            output_dir: Directory to save YOLO annotation files
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create annotation filename
            image_path = Path(result.image_path)
            annotation_file = output_dir / f"{image_path.stem}.txt"
            
            with open(annotation_file, 'w') as f:
                for detection in result.detections:
                    # YOLO format: class_id x_center y_center width height
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    class_name = detection.get('class', 'unknown')
                    
                    # For now, use a simple class mapping - in production, you'd have a proper class map
                    class_id = hash(class_name) % 80  # Simple hash to class ID
                    
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            self.logger.info(f"YOLO annotations saved to {annotation_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save YOLO annotations: {str(e)}")
    
    def visualize_detections(self, result: DetectionResult, 
                           output_dir: Union[str, Path] = None,
                           show_confidence: bool = True) -> Optional[Path]:
        """
        Create visualization of detections overlaid on the original image
        
        Args:
            result: DetectionResult object
            output_dir: Directory to save visualization
            show_confidence: Whether to show confidence scores
            
        Returns:
            Path to saved visualization file
        """
        try:
            # Load original image
            image = Image.open(result.image_path)
            draw = ImageDraw.Draw(image)
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Define colors for different classes
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown']
            class_colors = {}
            
            # Draw detections
            for i, detection in enumerate(result.detections):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Convert normalized coordinates to pixel coordinates
                x_center = bbox[0] * img_width
                y_center = bbox[1] * img_height
                width = bbox[2] * img_width
                height = bbox[3] * img_height
                
                # Calculate corner coordinates
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Get color for this class
                if class_name not in class_colors:
                    class_colors[class_name] = colors[len(class_colors) % len(colors)]
                color = class_colors[class_name]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{class_name}"
                if show_confidence:
                    label += f" ({confidence:.2f})"
                
                # Try to load a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # Get text size and draw background
                bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1, y1 - 20), label, fill='white', font=font)
            
            # Save visualization
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = Path(result.image_path)
                output_file = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
                image.save(output_file)
                
                self.logger.info(f"Visualization saved to {output_file}")
                return output_file
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {str(e)}")
            return None


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Multi-Modal Object Detection using Qwen2.5-VL-72B-Instruct')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--api-key', help='API key for Qwen service')
    parser.add_argument('--api-url', help='API URL for Qwen service')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-72B-Instruct', help='Model name to use')
    parser.add_argument('--batch', action='store_true', help='Process directory of images in batch')
    parser.add_argument('--visualize', action='store_true', help='Create detection visualizations')
    parser.add_argument('--yolo', action='store_true', help='Save annotations in YOLO format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger('multimodal_detector').setLevel(logging.DEBUG)
    
    try:
        # Initialize detector
        detector = MultiModalDetector(
            api_key=args.api_key,
            api_url=args.api_url,
            model_name=args.model
        )
        
        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else Path.cwd() / 'detection_results'
        
        if args.batch or input_path.is_dir():
            # Batch processing
            if not input_path.is_dir():
                raise ValueError("Input must be a directory for batch processing")
            
            # Find all image files
            image_files = []
            for ext in detector.supported_formats:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                raise ValueError(f"No supported image files found in {input_path}")
            
            print(f"Found {len(image_files)} images for batch processing")
            
            # Process batch
            results = detector.batch_detect(image_files, output_dir, save_intermediate=True)
            
            # Generate visualizations and YOLO annotations if requested
            for result in results:
                if result.error is None:
                    if args.visualize:
                        viz_dir = output_dir / 'visualizations'
                        detector.visualize_detections(result, viz_dir)
                    
                    if args.yolo:
                        yolo_dir = output_dir / 'yolo_annotations'
                        detector.save_yolo_annotations(result, yolo_dir)
        
        else:
            # Single image processing
            if not input_path.is_file():
                raise ValueError("Input must be a valid image file")
            
            result = detector.detect_objects(input_path)
            
            if result.error:
                print(f"Error processing image: {result.error}")
                return 1
            
            # Save results
            output_dir.mkdir(parents=True, exist_ok=True)
            detector._save_result_json(result, output_dir / 'detection_result.json')
            
            # Generate visualization and YOLO annotations if requested
            if args.visualize:
                detector.visualize_detections(result, output_dir)
            
            if args.yolo:
                detector.save_yolo_annotations(result, output_dir)
            
            print(f"Detection completed: {len(result.detections)} objects detected")
            print(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())