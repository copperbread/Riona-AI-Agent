import json
import os
import time
import pandas as pd
from openai import OpenAI
import base64
from datetime import datetime
from tqdm import tqdm
from rich.console import Console
import cv2
import numpy as np
from PIL import Image
import io
import glob
from pathlib import Path


class MultiModalDetector:
    def __init__(self, api_key, api_url="http://localhost:8000/v1", model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        初始化多模态检测器

        参数:
            api_key (str): OpenAI API密钥
            api_url (str): API接口地址
            model_name (str): 要使用的模型名称
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.console = Console()
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

    def encode_image_to_base64(self, image_path):
        """
        将图像文件编码为base64格式

        参数:
            image_path (str): 图像文件路径

        返回:
            str: base64编码的图像数据
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.console.print(f"[bold red]编码图像失败 {image_path}: {str(e)}[/]")
            return None

    def detect_objects(self, rgb_image_path, ir_image_path):
        """
        使用Qwen2.5-VL-7B-Instruct检测可见光和红外图像中的目标

        参数:
            rgb_image_path (str): 可见光图像路径
            ir_image_path (str): 红外图像路径

        返回:
            dict: 包含检测结果的字典
        """
        try:
            # 编码图像
            rgb_base64 = self.encode_image_to_base64(rgb_image_path)
            ir_base64 = self.encode_image_to_base64(ir_image_path)

            if not rgb_base64 or not ir_base64:
                return {"error": "图像编码失败"}

            base64_rgb_qwen = f"data:image;base64,{rgb_base64}"
            base64_ir_qwen = f"data:image;base64,{ir_base64}"

            prompt = """
请分析这两张图像（第一张是可见光图像，第二张是红外图像），检测其中的person（人）和car（汽车）目标。

要求：
1. 仔细观察两张图像中的所有person和car目标
2. 结合可见光和红外图像的信息进行更准确的检测
3. 返回YOLO格式的标注结果
4. YOLO格式：class_id x_center y_center width height（归一化坐标，范围0-1）
5. class_id: person=0, car=1

请只返回JSON格式的检测结果，格式如下：
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person", 
      "x_center": 0.5,
      "y_center": 0.5,
      "width": 0.1,
      "height": 0.2,
      "confidence": 0.95
    }
  ]
}
"""

            # 构建消息内容，支持多图像输入
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_rgb_qwen
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_ir_qwen
                            }
                        }
                    ]
                }
            ]

            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            result = json.loads(response['choices'][0]['message']['content'])
            return result

        except Exception as e:
            self.console.print(f"[bold red]检测过程中出错: {str(e)}[/]")
            return {"error": str(e)}

    def detect_with_retry(self, rgb_image_path, ir_image_path, max_retries=3, retry_delay=5):
        """
        带重试机制的检测函数

        参数:
            rgb_image_path (str): 可见光图像路径
            ir_image_path (str): 红外图像路径
            max_retries (int): 最大重试次数
            retry_delay (int): 初始重试间隔（秒）

        返回:
            dict: 包含检测结果的字典
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.detect_objects(rgb_image_path, ir_image_path)
                if "error" not in result and attempt > 0:
                    self.console.print(f"[green]在第{attempt}次重试后成功完成检测[/]")
                return result
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)
                    self.console.print(
                        f"[yellow]检测失败 ({error_msg})，{wait_time:.1f}秒后重试... (尝试 {attempt + 1}/{max_retries})[/]"
                    )
                    time.sleep(wait_time)
                else:
                    self.console.print(f"[bold red]达到最大重试次数，检测失败: {error_msg}[/]")
                    return {"error": f"检测失败 (重试{max_retries}次后): {error_msg}"}

# 示例用法
if __name__ == "__main__":
    # 配置参数
    API_KEY = "sk-your-api-key"  # 替换为您的OpenAI API密钥
    API_URL = "http://localhost:8000/v1"
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    # 输入和输出目录
    RGB_DIR = "/path/to/rgb/images"
    IR_DIR = "/path/to/ir/images"
    OUTPUT_DIR = "/path/to/output"

    # 创建检测器实例
    console = Console()
    console.print("[bold green]初始化多模态目标检测器...[/]")

    detector = MultiModalDetector(
        api_key=API_KEY,
        api_url=API_URL,
        model_name=MODEL_NAME
    )

    # 示例检测
    rgb_image = "/path/to/rgb/image.jpg"
    ir_image = "/path/to/ir/image.jpg"

    result = detector.detect_objects(rgb_image, ir_image)
    print("Detection Results:", result)