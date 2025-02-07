from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
import shutil
from pathlib import Path
import json
import base64
from py_models import ImageDescription, ImageOptimizer


app = FastAPI()

# Directory to save uploaded images
UPLOAD_DIRECTORY = Path("uploaded_images")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

@app.post("/analyze-image/", response_model=ImageDescription)
async def analyze_image(file: UploadFile = File(...)):
    # Validate the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save the file temporarily
    file_path = UPLOAD_DIRECTORY / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Enhanced prompt for better accuracy
        prompt = """Please analyze this image carefully and provide a detailed security assessment. 
        Focus specifically on:
        1. The presence of any humans/people in the image (even partial body parts)
        2. The presence of any weapons (firearms, knives, or other dangerous objects)
        3. Overall context and potential security threats

        Respond ONLY with a JSON object in this exact format:
        {
          "image_context": "detailed description of the scene",
          "has_weapon": true/false,
          "has_people": true/false (mark true if ANY human presence is detected),
          "confidence": number between 0-100
        }

        Be especially thorough in detecting human presence - even if it's just parts of a person visible."""

        # Optimize the image - reduced grayscale for better detail retention
        optimizer = ImageOptimizer(max_size=800, quality=90, use_grayscale=False)
        optimized_image = optimizer.optimize_image(str(file_path))
        
        # Read and encode the optimized image in base64
        with open(optimized_image, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llava:7b",
                "messages": [{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]  # Send base64 encoded image
                }],
                "stream": False,
                "options": {
                    "num_ctx": 1024,        
                    "num_thread": 8,        
                    "num_gpu": 1,
                    "temperature": 0.2,     
                    "top_k": 40,           
                    "top_p": 0.9,          
                }
            }
        )

        # Parse the response - Updated version
        try:
            response_json = response.json()
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Error from Ollama API: {response_json}")
                
            response_text = response_json.get('message', {}).get('content', '')
            
            # Debug logging
            print("Raw response:", response_text)
            
            json_str = response_text.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]  # Remove ```json and ``` if present
            
            result = json.loads(json_str)
            return ImageDescription(**result)
            
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", response_text)  # Debug logging
            raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
    finally:
        # Cleanup: Remove temporary files
        if file_path.exists():
            file_path.unlink()
        optimized_path = Path(f"optimized_{file_path}")  # Fixed path for optimized image
        if optimized_path.exists():
            optimized_path.unlink()
    







