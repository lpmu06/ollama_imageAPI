from ollama import chat
from typing import Literal, Optional
from pydantic import BaseModel
import time
from PIL import Image
import json
from datetime import datetime

class ImageDescription(BaseModel):
    image_context: str
    has_weapon: bool
    has_people: bool
    confidence: float
    scene_type: Literal['Indoor', 'Outdoor', 'Vehicle', 'Other']
    potential_threats: list[str]
    detected_objects: list[str]
    timestamp: Optional[str]

class ImageOptimizer:
    def __init__(self, max_size=640, quality=80, use_grayscale=True):
        self.max_size = max_size
        self.quality = quality
        self.use_grayscale = use_grayscale

    def optimize_image(self, image_path):
        img = Image.open(image_path)
        
        if self.use_grayscale and img.mode in ('RGB', 'RGBA'):
            img = img.convert('L')

        if max(img.size) > self.max_size:
            ratio = self.max_size / max(img.size)
            new_size = tuple([int(x * ratio) for x in img.size])
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        optimized_path = 'optimized_' + image_path
        img.save(optimized_path, 
                format='JPEG', 
                quality=self.quality, 
                optimize=True,
                progressive=True)

        return optimized_path

def get_timestamp():
    """Returns a formatted timestamp string for the analysis"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_image(image_path, model="llava:7b"):
    timestamp = get_timestamp()
    optimizer = ImageOptimizer(max_size=640, quality=80, use_grayscale=True)
    optimized_image = optimizer.optimize_image(image_path)
    
    print(f"Starting analysis at: {timestamp}")
    
    SYSTEM_PROMPT = """You are a security-focused image analysis system. Your task is to carefully analyze images 
    for potential security threats, weapons, suspicious activities, and provide detailed scene information."""

    USER_PROMPT = """Analyze this image from a security perspective and provide a detailed assessment. 
    Look for weapons, people, potential threats, and important objects in the scene."""

    start_time = time.time()
    
    response = chat(
        model=model,
        format=ImageDescription.model_json_schema(),  # Pass the schema for structured output
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': USER_PROMPT,
                'images': [optimized_image]
            }
        ],
        options={
            "num_ctx": 512,
            "num_thread": 6,
            "num_gpu": 1,
            "temperature": 0.1,
            "top_k": 10,
            "top_p": 0.95
        }
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    try:
        result = ImageDescription.model_validate_json(response.message.content)
    except Exception as e:
        result = ImageDescription(
            image_context="Error parsing image",
            has_weapon=False,
            has_people=False,
            confidence=0,
            scene_type="Other",
            potential_threats=[],
            detected_objects=[],
            timestamp=None
        )
    
    return result, execution_time

def print_analysis_results(result: ImageDescription, exec_time: float):
    timestamp = get_timestamp()
    print("\nSECURITY ANALYSIS RESULTS:")
    print("-" * 50)
    print(f"Analysis Timestamp: {timestamp}")
    print(f"\nContext: {result.image_context}")
    print(f"Scene Type: {result.scene_type}")
    print(f"Timestamp: {result.timestamp or 'Not recorded'}")
    print(f"\nSecurity Findings:")
    print(f"- Weapons Detected: {'Yes' if result.has_weapon else 'No'}")
    print(f"- People Present: {'Yes' if result.has_people else 'No'}")
    print(f"- Confidence Level: {result.confidence:.2f}%")
    
    print("\nPotential Threats:")
    for threat in result.potential_threats:
        print(f"  - {threat}")
    
    print("\nDetected Objects:")
    for obj in result.detected_objects:
        print(f"  - {obj}")
    
    print(f"\nAnalysis Time: {exec_time:.2f} seconds")

def get_json_results(result: ImageDescription, exec_time: float) -> str:
    """
    Converts the analysis results to a formatted JSON string
    """
    analysis_data = {
        "analysis_results": {
            "timestamp": result.timestamp,
            "image_context": result.image_context,
            "scene_type": result.scene_type,
            "security_findings": {
                "has_weapon": result.has_weapon,
                "has_people": result.has_people,
                "confidence": result.confidence
            },
            "potential_threats": result.potential_threats,
            "detected_objects": result.detected_objects,
            "execution_time": round(exec_time, 2)
        }
    }
    
    return json.dumps(analysis_data, indent=2)

if __name__ == "__main__":
    image_path = 'assalto.jpg'
    print("Analyzing image for security threats...")
    result, exec_time = process_image(image_path)
    print_analysis_results(result, exec_time)
    
    # Get JSON format
    json_output = get_json_results(result, exec_time)
    print("\nJSON Output:")
    print(json_output)