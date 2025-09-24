#!/usr/bin/env python3
"""
Multi-modal input processor using Qwen/Qwen2.5-VL-72B-Instruct model.
Handles both text and image inputs (RGB and IR) with batch processing and progress tracking.
"""

import os
import sys
import json
import logging
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import requests
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ProcessingConfig:
    """Configuration for multi-modal processing"""
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    api_base_url: str = "https://api.openai.com/v1"
    max_retries: int = 3
    timeout: int = 30
    max_batch_size: int = 10
    supported_image_formats: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    max_image_size: tuple = (2048, 2048)

class MultiModalLogger:
    """Enhanced logging system for multi-modal processing"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("multimodal_processor")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "multimodal_processor.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

class PathHandler:
    """Cross-platform file path handling utility"""
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """Normalize path for cross-platform compatibility"""
        return Path(path).resolve()
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary"""
        path = PathHandler.normalize_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_extension(path: Union[str, Path]) -> str:
        """Get file extension in lowercase"""
        return Path(path).suffix.lower()
    
    @staticmethod
    def validate_image_file(path: Union[str, Path]) -> bool:
        """Validate if file is a supported image format"""
        config = ProcessingConfig()
        return PathHandler.get_file_extension(path) in config.supported_image_formats

class ImageProcessor:
    """Image processing utilities for RGB and IR images"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = MultiModalLogger()
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        if image.size[0] <= self.config.max_image_size[0] and image.size[1] <= self.config.max_image_size[1]:
            return image
        
        image.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
        return image
    
    def encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string"""
        try:
            image_path = PathHandler.normalize_path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not PathHandler.validate_image_file(image_path):
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode not in ['RGB', 'RGBA']:
                    image = image.convert('RGB')
                
                # Resize if necessary
                image = self.resize_image(image)
                
                # Save to base64
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                
                return base64.b64encode(image_bytes).decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Get image metadata"""
        try:
            image_path = PathHandler.normalize_path(image_path)
            
            with Image.open(image_path) as image:
                return {
                    'path': str(image_path),
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format,
                    'file_size': image_path.stat().st_size
                }
        except Exception as e:
            self.logger.error(f"Error getting image info {image_path}: {str(e)}")
            return {}

class QwenMultiModalClient:
    """Client for Qwen multi-modal API with proper message formatting"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = MultiModalLogger()
        self.image_processor = ImageProcessor(config)
        
        # Get API key from environment
        self.api_key = os.getenv('QWEN_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Set QWEN_API_KEY or OPENAI_API_KEY environment variable.")
    
    def create_message_content(self, text: str, image_paths: List[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Create properly formatted message content for API request"""
        content = []
        
        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # Add image content
        if image_paths:
            for image_path in image_paths:
                try:
                    base64_image = self.image_processor.encode_image_to_base64(image_path)
                    image_info = self.image_processor.get_image_info(image_path)
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                    
                    self.logger.debug(f"Added image to content: {image_path} ({image_info.get('size', 'unknown size')})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process image {image_path}: {str(e)}")
                    continue
        
        return content
    
    def make_api_request(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Make API request with proper error handling and retries"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 1000),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0)
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Making API request (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    f"{self.config.api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.debug("API request successful")
                    return result
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    
                    if attempt == self.config.max_retries - 1:
                        raise requests.RequestException(error_msg)
            
            except requests.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt == self.config.max_retries - 1:
                    raise
            
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def process_single_input(self, text: str, image_paths: List[Union[str, Path]] = None, **kwargs) -> Dict[str, Any]:
        """Process a single multi-modal input"""
        try:
            content = self.create_message_content(text, image_paths)
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            response = self.make_api_request(messages, **kwargs)
            
            return {
                'success': True,
                'response': response,
                'input_text': text,
                'input_images': [str(p) for p in (image_paths or [])],
                'timestamp': self._get_timestamp()
            }
        
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'input_text': text,
                'input_images': [str(p) for p in (image_paths or [])],
                'timestamp': self._get_timestamp()
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

class BatchProcessor:
    """Batch processing with progress tracking"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = MultiModalLogger()
        self.client = QwenMultiModalClient(config)
        self.path_handler = PathHandler()
    
    def find_images_in_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[Path]:
        """Find all supported image files in directory"""
        directory = self.path_handler.normalize_path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        
        pattern = "**/*" if recursive else "*"
        all_files = directory.glob(pattern)
        
        image_files = [
            f for f in all_files 
            if f.is_file() and self.path_handler.validate_image_file(f)
        ]
        
        self.logger.info(f"Found {len(image_files)} image files in {directory}")
        return image_files
    
    def categorize_images(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Categorize images into RGB and IR based on filename patterns"""
        categories = {"rgb": [], "ir": [], "unknown": []}
        
        for path in image_paths:
            filename = path.stem.lower()
            if any(keyword in filename for keyword in ['rgb', 'color', 'visible']):
                categories["rgb"].append(path)
            elif any(keyword in filename for keyword in ['ir', 'infrared', 'thermal']):
                categories["ir"].append(path)
            else:
                categories["unknown"].append(path)
        
        self.logger.info(f"Categorized images: RGB({len(categories['rgb'])}), IR({len(categories['ir'])}), Unknown({len(categories['unknown'])})")
        return categories
    
    def process_batch(self, 
                     input_data: List[Dict[str, Any]], 
                     output_dir: Union[str, Path],
                     batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Process batch of multi-modal inputs"""
        
        batch_size = batch_size or self.config.max_batch_size
        output_dir = self.path_handler.ensure_directory(output_dir)
        
        results = []
        errors = []
        
        self.logger.info(f"Starting batch processing of {len(input_data)} items...")
        
        # Process in batches with progress bar
        with tqdm(total=len(input_data), desc="Processing") as pbar:
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size]
                
                for item in batch:
                    try:
                        result = self.client.process_single_input(
                            text=item.get('text', ''),
                            image_paths=item.get('image_paths', []),
                            **item.get('api_params', {})
                        )
                        
                        if result['success']:
                            results.append(result)
                        else:
                            errors.append(result)
                    
                    except Exception as e:
                        error_item = {
                            'success': False,
                            'error': str(e),
                            'input_data': item,
                            'timestamp': self.client._get_timestamp()
                        }
                        errors.append(error_item)
                    
                    pbar.update(1)
        
        # Save results
        summary = {
            'total_processed': len(input_data),
            'successful': len(results),
            'failed': len(errors),
            'success_rate': len(results) / len(input_data) if input_data else 0,
            'results': results,
            'errors': errors,
            'timestamp': self.client._get_timestamp()
        }
        
        # Save to output directory
        results_file = output_dir / f"batch_results_{self.client._get_timestamp().replace(':', '-')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch processing complete. Results saved to: {results_file}")
        self.logger.info(f"Success rate: {summary['success_rate']:.2%} ({summary['successful']}/{summary['total_processed']})")
        
        return summary
    
    def process_image_directory(self,
                               image_dir: Union[str, Path],
                               prompt_template: str,
                               output_dir: Union[str, Path],
                               **kwargs) -> Dict[str, Any]:
        """Process all images in a directory with a prompt template"""
        
        image_files = self.find_images_in_directory(image_dir)
        categorized = self.categorize_images(image_files)
        
        # Create input data for batch processing
        input_data = []
        
        for category, paths in categorized.items():
            for path in paths:
                input_data.append({
                    'text': prompt_template.format(
                        image_type=category,
                        filename=path.name,
                        filepath=str(path)
                    ),
                    'image_paths': [path],
                    'api_params': kwargs.get('api_params', {})
                })
        
        return self.process_batch(input_data, output_dir, kwargs.get('batch_size'))

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-modal processor using Qwen model")
    parser.add_argument("--text", type=str, help="Input text prompt")
    parser.add_argument("--images", nargs="+", help="Image file paths")
    parser.add_argument("--image-dir", type=str, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch processing size")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize components
    config = ProcessingConfig()
    logger = MultiModalLogger(args.log_level)
    processor = BatchProcessor(config)
    
    try:
        if args.image_dir:
            # Process directory of images
            prompt_template = args.text or "Analyze this {image_type} image: {filename}"
            
            results = processor.process_image_directory(
                image_dir=args.image_dir,
                prompt_template=prompt_template,
                output_dir=args.output_dir,
                batch_size=args.batch_size
            )
            
            print(f"\nProcessing completed!")
            print(f"Success rate: {results['success_rate']:.2%}")
            print(f"Results saved to: {args.output_dir}")
        
        elif args.text and args.images:
            # Process single input
            client = QwenMultiModalClient(config)
            result = client.process_single_input(args.text, args.images)
            
            if result['success']:
                response = result['response']
                if 'choices' in response and response['choices']:
                    print(f"\nResponse: {response['choices'][0]['message']['content']}")
                else:
                    print(f"\nFull response: {json.dumps(response, indent=2)}")
            else:
                print(f"Error: {result['error']}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()