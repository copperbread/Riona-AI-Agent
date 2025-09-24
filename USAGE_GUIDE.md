# Multi-Modal Processing Usage Guide

## Quick Start

1. **Setup Environment**:
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Set API key
   export QWEN_API_KEY="your_api_key_here"
   ```

2. **Run Demo**:
   ```bash
   python demo_multimodal.py
   ```

3. **Process Images**:
   ```bash
   # Single image
   python src/multimodal_processor.py \
     --text "Analyze this image" \
     --images path/to/image.jpg \
     --output-dir ./results
   
   # Directory of images
   python src/multimodal_processor.py \
     --image-dir ./sample_images \
     --output-dir ./results
   ```

## Key Features Implemented

### ✅ 1. Correct Model Name
- **Requirement**: Use `Qwen/Qwen2.5-VL-72B-Instruct`
- **Implementation**: Configured in `ProcessingConfig.model_name`
- **Verification**: See demo output showing correct model name

### ✅ 2. API Compliance
- **Requirement**: Proper message format for text + images
- **Implementation**: 
  ```python
  {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Your prompt"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }
  ```
- **Verification**: See `demo_message_formatting()` in demo

### ✅ 3. Enhanced Error Handling
- **Requirement**: Better debugging and issue tracking
- **Implementation**:
  - Automatic retries with exponential backoff
  - Rate limiting handling (429 errors)
  - Network timeout management
  - Comprehensive logging to files
  - Graceful error recovery
- **Verification**: See `demo_error_handling()` in demo

### ✅ 4. Cross-Platform File Paths
- **Requirement**: Compatible across different operating systems
- **Implementation**:
  - `pathlib.Path` for all path operations
  - `PathHandler.normalize_path()` for cross-platform compatibility
  - Automatic directory creation with `mkdir(parents=True, exist_ok=True)`
- **Verification**: See path handling demo with Windows/Linux/Mac paths

### ✅ 5. Batch Processing & Progress Tracking
- **Requirement**: Handle multiple images with progress tracking
- **Implementation**:
  - `BatchProcessor` class for handling multiple inputs
  - Progress bars with `tqdm` (or `SimpleProgressBar` fallback)
  - Configurable batch sizes
  - Results saved to JSON files
- **Verification**: See `demo_batch_processing_structure()` in demo

### ✅ 6. RGB/IR Image Support
- **Requirement**: Handle both RGB and IR images
- **Implementation**:
  - Automatic image categorization based on filename keywords
  - Support for RGB keywords: 'rgb', 'color', 'visible', 'normal'
  - Support for IR keywords: 'ir', 'infrared', 'thermal', 'heat'
  - Batch processing preserves image categories
- **Verification**: See `demo_image_categorization()` in demo

## File Structure

```
📁 Multi-Modal Processing Files
├── requirements.txt                 # Python dependencies
├── README_MULTIMODAL.md            # Detailed documentation
├── demo_multimodal.py              # Comprehensive demo
├── USAGE_GUIDE.md                  # This guide
├── src/
│   ├── multimodal_processor.py     # Main implementation
│   ├── multimodal_processor_lite.py # Testing version
│   └── config/
│       └── multimodal_config.py    # Configuration
├── examples/
│   ├── multimodal_example.py       # Usage examples
│   └── sample_images/              # Sample images directory
└── output/                         # Results saved here
```

## Example Usage

### Python API
```python
from multimodal_processor import QwenMultiModalClient, ProcessingConfig

# Initialize
config = ProcessingConfig()
client = QwenMultiModalClient(config)

# Process single input
result = client.process_single_input(
    text="Analyze this RGB and IR image pair",
    image_paths=["rgb_image.jpg", "ir_image.jpg"],
    temperature=0.7
)

print(result['response']['choices'][0]['message']['content'])
```

### Command Line
```bash
# Process multiple images with custom prompt
python src/multimodal_processor.py \
  --text "Compare RGB and IR: {image_type} - {filename}" \
  --image-dir ./thermal_survey_images \
  --output-dir ./analysis_results \
  --batch-size 5 \
  --log-level DEBUG
```

## Integration with Existing System

This Python multi-modal processor complements the existing TypeScript AI agent:

- **TypeScript Agent** (`src/Agent/index.ts`): Social media automation with Gemini
- **Python Processor** (`src/multimodal_processor.py`): Advanced vision analysis with Qwen
- **Shared Patterns**: Both use similar logging, error handling, and configuration patterns

## Testing

Run the comprehensive test suite:
```bash
# Basic functionality test
python src/multimodal_processor_lite.py

# Full feature demo
python demo_multimodal.py

# With real API (requires API key)
python examples/multimodal_example.py
```

## Troubleshooting

1. **Missing Dependencies**: `pip install -r requirements.txt`
2. **API Key Issues**: Set `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
3. **Path Issues**: Use absolute paths or relative to project root
4. **Image Format**: Ensure images are JPEG, PNG, BMP, TIFF, or WebP
5. **Rate Limits**: Reduce batch size or add delays

## Next Steps

1. Set up API credentials
2. Add sample RGB/IR images to `examples/sample_images/`
3. Run examples to test functionality
4. Customize configuration in `src/config/multimodal_config.py`
5. Integrate with existing TypeScript workflows