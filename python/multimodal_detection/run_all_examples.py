#!/usr/bin/env python3
"""
Complete demonstration of Multi-Modal Detection System
Run all examples and tests to showcase the functionality
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✓ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("✗ FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False

def main():
    """Run all example scripts"""
    print("Multi-Modal Detection System - Complete Examples")
    print("This script runs all available examples and tests")
    
    scripts = [
        ("test_detector.py", "Unit Tests - Validate core functionality"),
        ("demo.py", "Feature Demo - Show all capabilities without API calls"),
        ("example_usage.py", "Usage Examples - Real-world usage patterns (requires API key)")
    ]
    
    results = []
    
    for script, description in scripts:
        success = run_script(script, description)
        results.append((script, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print('='*60)
    
    for script, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{script:<25} {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {successful}/{total} scripts ran successfully")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Set QWEN_API_KEY environment variable for real API usage")
        print("2. Install dependencies: pip install -r ../../requirements.txt")
        print("3. Try detector.py with real images")
    else:
        print(f"\n⚠️  {total - successful} scripts had issues")
        print("Check the output above for details")
    
    return successful == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)