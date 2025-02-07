from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
import shutil
from pathlib import Path
import json
import base64
from py_utils import ImageDescription, ImageOptimizer


app = FastAPI()

# Directory to save uploaded images
UPLOAD_DIRECTORY = Path("uploaded_images")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

OPTIMIZED_DIRECTORY = Path("optimized_uploaded_images")
OPTIMIZED_DIRECTORY.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

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
        prompt = """Analise esta imagem cuidadosamente e forneça uma avaliação detalhada de segurança.
        Concentre-se especificamente em:
        1. A presença de pessoas na imagem (mesmo que sejam apenas partes do corpo)
        2. A presença de armas (armas de fogo, facas ou outros objetos perigosos)
        3. Contexto geral e potenciais ameaças à segurança

        Responda APENAS com um objeto JSON no seguinte formato:
        {
          "image_context": "descrição detalhada da cena em português",
          "has_weapon": true/false,
          "has_people": true/false (marque true se QUALQUER presença humana for detectada),
        }

        Seja especialmente minucioso na detecção de presença humana - mesmo que sejam apenas partes visíveis de uma pessoa.
        IMPORTANTE: A descrição da cena (image_context) deve estar em português do Brasil."""

        # Optimize the image - reduced grayscale for better detail retention
        optimizer = ImageOptimizer(max_size=800, quality=90, use_grayscale=False)
        optimized_image = optimizer.optimize_image(str(file_path))
        
        # Read and encode the optimized image in base64
        with open(optimized_image, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        response = requests.post(
            "http://ollama:11434/api/chat",
            json={
                "model": "llava:7b",
                "messages": [{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                }],
                "stream": False,
                "options": {
                    "num_ctx": 1024,        # Tamanho do contexto em tokens - define quanto texto/informação o modelo pode processar de uma vez
                    "num_thread": 8,        # Número de threads CPU para processamento paralelo - mais threads podem acelerar a inferência
                    "num_gpu": 1,           # Número de GPUs a serem utilizadas - aumentar pode melhorar performance em hardware adequado
                    "temperature": 0.2,     # Controla aleatoriedade das respostas (0-1) - valores menores = respostas mais determinísticas
                    "top_k": 40,            # Limita a seleção aos k tokens mais prováveis - ajuda a manter respostas mais focadas
                    "top_p": 0.9,           # Amostragem nucleus - seleciona tokens cuja prob. acumulada atinge p% - balanceia criatividade/coerência
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
    







