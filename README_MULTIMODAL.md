# Multi-Modal Processing with Qwen Model

This directory contains a Python-based multi-modal processor that complements the existing TypeScript AI agent. It handles both text and image inputs (RGB and IR) using the Qwen/Qwen2.5-VL-72B-Instruct model.

## Features

- ✅ **Correct Model Name**: Uses `Qwen/Qwen2.5-VL-72B-Instruct` as specified
- ✅ **API Compliance**: Proper message format according to API documentation
- ✅ **Enhanced Error Handling**: Comprehensive error handling with retries and logging
- ✅ **Cross-Platform Path Handling**: Compatible across different operating systems
- ✅ **Batch Processing**: Handles multiple images with progress tracking
- ✅ **RGB/IR Image Support**: Automatic categorization and processing of different image types
- ✅ **Output Management**: Saves results to specified output directories

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   # For Qwen API
   export QWEN_API_KEY="your_qwen_api_key_here"
   
   # Alternative: For DashScope API
   export DASHSCOPE_API_KEY="your_dashscope_api_key_here"
   ```

## Project Structure

```
src/
├── multimodal_processor.py      # Main multi-modal processing script
├── config/
│   └── multimodal_config.py    # Configuration settings
└── logs/                       # Log files (auto-created)

examples/
├── multimodal_example.py       # Usage examples
├── sample_images/              # Sample images directory
└── README.md                   # Example documentation

output/                         # Processing results (auto-created)
requirements.txt               # Python dependencies
```

## Usage

### Command Line Interface

```bash
# Process single image with text
python src/multimodal_processor.py \
    --text "Analyze this image and describe what you see" \
    --images path/to/image1.jpg path/to/image2.jpg \
    --output-dir ./results

# Process directory of images
python src/multimodal_processor.py \
    --image-dir ./sample_images \
    --text "Analyze this {image_type} image: {filename}" \
    --output-dir ./results \
    --batch-size 5
```

### Python API

```python
from multimodal_processor import QwenMultiModalClient, ProcessingConfig

# Initialize
config = ProcessingConfig()
client = QwenMultiModalClient(config)

# Process single input
result = client.process_single_input(
    text="Describe this image",
    image_paths=["image1.jpg", "image2.jpg"],
    temperature=0.7,
    max_tokens=800
)

# Batch processing
from multimodal_processor import BatchProcessor

processor = BatchProcessor(config)
results = processor.process_image_directory(
    image_dir="./images",
    prompt_template="Analyze this {image_type} image",
    output_dir="./results"
)
```

## API Message Format

The processor formats messages according to the API specification:

```python
{
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Your text prompt here"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{base64_encoded_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.7
}
```

## Image Processing Features

### Supported Formats
- JPEG, PNG, BMP, TIFF, WebP
- Automatic format detection
- Cross-platform path handling

### Image Categorization
Images are automatically categorized based on filename keywords:
- **RGB**: 'rgb', 'color', 'visible', 'normal'
- **IR**: 'ir', 'infrared', 'thermal', 'heat'

### Preprocessing
- Automatic resizing (max 2048x2048)  
- Format conversion to JPEG
- Base64 encoding for API transmission

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Automatic retries with exponential backoff
- **Rate Limiting**: Intelligent handling of 429 errors
- **File Errors**: Validation and informative error messages
- **Network Issues**: Timeout handling and connection retries
- **Logging**: Detailed logs saved to `src/logs/`

## Configuration

Customize processing through `src/config/multimodal_config.py`:

```python
class MultiModalConfig:
    MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
    MAX_RETRIES = 3
    TIMEOUT = 60
    MAX_BATCH_SIZE = 10
    MAX_IMAGE_SIZE = (2048, 2048)
    # ... more settings
```

## Examples

Run the example script to see all features:

```bash
python examples/multimodal_example.py
```

This will demonstrate:
- Single image processing
- Batch processing
- Directory processing
- Different configuration options

## Logging

Logs are automatically saved to `src/logs/multimodal_processor.log` with:
- Timestamp and log level
- Detailed error information
- Processing statistics
- API request/response details

## Integration with Existing System

This Python multi-modal processor complements the existing TypeScript AI agent:

- **TypeScript Agent**: Handles Instagram/Twitter automation with Gemini
- **Python Processor**: Handles advanced multi-modal analysis with Qwen
- **Shared Configuration**: Both use similar logging and error handling patterns

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   export QWEN_API_KEY="your_key_here"
   ```

2. **Image Format Not Supported**
   - Check supported formats: JPEG, PNG, BMP, TIFF, WebP
   - Verify file is not corrupted

3. **Rate Limiting**
   - Reduce batch size
   - Increase delay between requests

4. **Large Images**
   - Images are automatically resized to 2048x2048
   - Adjust `MAX_IMAGE_SIZE` in config if needed

### Debug Mode

Enable debug logging:
```bash
python src/multimodal_processor.py --log-level DEBUG
```

## Performance

- **Batch Processing**: Up to 10 images per batch (configurable)
- **Progress Tracking**: Real-time progress bars with `tqdm`
- **Memory Efficient**: Images processed one at a time
- **Error Recovery**: Continues processing after individual failures

## Contributing

When contributing to the multi-modal processor:

1. Follow existing code patterns
2. Add comprehensive error handling
3. Include logging for debugging
4. Test with both RGB and IR images
5. Ensure cross-platform compatibility

## License

This project is licensed under the MIT License - see the main LICENSE file for details.