# Multi-Modal Detection Script

A robust Python script for multi-modal object detection using the Qwen2.5-VL-72B-Instruct model. This script supports both RGB and IR (thermal) image processing with comprehensive error handling, batch processing, and YOLO-format output generation.

## Features

- **Multi-Modal Support**: Handles both RGB and IR (thermal) images
- **Robust API Integration**: Compatible with Qwen2.5-VL-72B-Instruct model
- **Batch Processing**: Efficient processing of multiple images with progress tracking
- **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS
- **Comprehensive Error Handling**: Detailed error messages and logging
- **Multiple Output Formats**: JSON results, YOLO annotations, and visualizations
- **Intermediate Result Saving**: Progress preservation for large batches
- **Configurable Detection Thresholds**: Adjustable confidence levels

## Installation

1. **Install Python Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   ```bash
   export QWEN_API_KEY="your_api_key_here"
   export QWEN_API_URL="https://api.openai.com/v1/chat/completions"  # Optional
   ```

3. **Verify Installation**:
   ```bash
   python example_usage.py
   ```

## Usage

### Command Line Interface

#### Single Image Detection
```bash
python detector.py path/to/image.jpg --output results/
```

#### Batch Processing
```bash
python detector.py path/to/images/ --batch --output results/ --visualize --yolo
```

#### Complete Example
```bash
python detector.py images/ \
    --batch \
    --output detection_results/ \
    --visualize \
    --yolo \
    --api-key YOUR_API_KEY \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --verbose
```

### Python API

#### Basic Usage
```python
from detector import MultiModalDetector

# Initialize detector
detector = MultiModalDetector(
    api_key="your_api_key",
    model_name="Qwen/Qwen2.5-VL-72B-Instruct"
)

# Process single image
result = detector.detect_objects("image.jpg", image_type="RGB")

if result.error:
    print(f"Error: {result.error}")
else:
    print(f"Found {len(result.detections)} objects")
    for detection in result.detections:
        print(f"- {detection['class']}: {detection['confidence']:.2f}")
```

#### Batch Processing
```python
from pathlib import Path
from utils import find_image_files

# Find all images in directory
image_files = find_image_files("images/", recursive=True)

# Process batch
results = detector.batch_detect(
    image_files,
    output_dir="results/",
    save_intermediate=True
)

# Generate visualizations
for result in results:
    if result.error is None:
        detector.visualize_detections(result, "visualizations/")
        detector.save_yolo_annotations(result, "annotations/")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_API_KEY` | API key for Qwen service | Required |
| `QWEN_API_URL` | API endpoint URL | `https://api.openai.com/v1/chat/completions` |
| `QWEN_MODEL_NAME` | Model name to use | `Qwen/Qwen2.5-VL-72B-Instruct` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections | `0.3` |
| `MAX_BATCH_SIZE` | Maximum images per batch | `100` |
| `API_TIMEOUT` | API request timeout (seconds) | `60` |

### Configuration File

Create a `detection_config.json` file:
```json
{
  "api_key": "your_api_key_here",
  "api_url": "https://api.openai.com/v1/chat/completions",
  "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
  "confidence_threshold": 0.3,
  "max_batch_size": 50,
  "timeout": 60
}
```

## Output Formats

### JSON Results
```json
{
  "image_path": "path/to/image.jpg",
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [0.5, 0.3, 0.2, 0.4],
      "attributes": {"color": "blue", "size": "medium"}
    }
  ],
  "confidence_scores": [0.95],
  "processing_time": 2.5,
  "error": null
}
```

### YOLO Annotations
Text files with format: `class_id x_center y_center width height`
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.1 0.2
```

### Visualizations
Images with bounding boxes and labels overlaid on the original image.

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)
- WebP (`.webp`)

## Error Handling

The script provides comprehensive error handling for:

- **API Errors**: Authentication, rate limiting, service unavailability
- **File Errors**: Missing files, permission issues, invalid formats
- **Network Errors**: Connection timeouts, DNS resolution failures
- **Processing Errors**: Invalid responses, memory issues

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Authentication failed` | Invalid API key | Check `QWEN_API_KEY` environment variable |
| `Rate limit exceeded` | Too many requests | Wait and retry, or use batch processing |
| `File does not exist` | Invalid file path | Verify file path and permissions |
| `Unsupported image format` | Invalid image type | Use supported formats (JPEG, PNG, etc.) |
| `Connection timeout` | Network issues | Check internet connection and API URL |

## Performance Optimization

### Batch Processing Tips
- Use batch processing for multiple images (faster than individual requests)
- Enable intermediate result saving for large batches
- Adjust batch size based on available memory and API limits

### Memory Management
- Images are processed one at a time to minimize memory usage
- Base64 encoding is done just before API call
- Temporary files are cleaned up automatically

### API Optimization
- Requests are made with connection pooling
- Automatic retry with exponential backoff
- Proper error handling to avoid unnecessary retries

## Directory Structure

```
python/multimodal_detection/
├── detector.py              # Main detection script
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── example_usage.py        # Usage examples
├── README.md              # This file
└── tests/                 # Test files (if implemented)
```

## Logging

The script provides detailed logging:

- **Console Output**: Progress and summary information
- **File Logging**: Detailed debug information saved to `logs/multimodal_detection.log`
- **Error Tracking**: Full stack traces for debugging

### Log Levels
- `INFO`: General progress and status
- `DEBUG`: Detailed processing information
- `WARNING`: Non-fatal issues
- `ERROR`: Processing failures

## Troubleshooting

### API Issues
1. **Verify API Key**: Ensure `QWEN_API_KEY` is set correctly
2. **Check API URL**: Confirm the endpoint URL is correct
3. **Test Connection**: Use a simple HTTP client to test API connectivity

### File Processing Issues
1. **Check Permissions**: Ensure read access to input images
2. **Verify Paths**: Use absolute paths to avoid confusion
3. **Test with Single Image**: Start with one image before batch processing

### Performance Issues
1. **Reduce Batch Size**: Lower `MAX_BATCH_SIZE` for memory-constrained systems
2. **Increase Timeout**: Set higher `API_TIMEOUT` for slow connections
3. **Monitor Resources**: Check CPU, memory, and network usage

## Examples

See `example_usage.py` for comprehensive examples including:
- Single image processing
- Batch processing
- Visualization generation
- IR image handling
- Error handling patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen team for the VL-72B-Instruct model
- PIL and OpenCV communities for image processing libraries
- Contributors to the requests and other Python libraries used