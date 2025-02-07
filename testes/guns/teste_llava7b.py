from ollama import chat
import time
from PIL import Image
import json
from pydantic import BaseModel

class ImageDescription(BaseModel):
    image_context: str
    has_weapon: bool
    has_people: bool
    confidence: int

class ImageOptimizer:
    def __init__(self, max_size=640, quality=80, use_grayscale=True):
        self.max_size = max_size
        self.quality = quality
        self.use_grayscale = use_grayscale

    def optimize_image(self, image_path):
        # Abre e otimiza a imagem
        img = Image.open(image_path)

        # Converte para escala de cinza se configurado
        if self.use_grayscale and img.mode in ('RGB', 'RGBA'):
            img = img.convert('L')

        # Redimensiona se necessário
        if max(img.size) > self.max_size:
            ratio = self.max_size / max(img.size)
            new_size = tuple([int(x * ratio) for x in img.size])
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Salva a imagem otimizada
        optimized_path = 'optimized_' + image_path
        img.save(optimized_path, 
                format='JPEG', 
                quality=self.quality, 
                optimize=True,
                progressive=True)

        return optimized_path

def process_image(image_path, model="llava:7b"):
    optimizer = ImageOptimizer(max_size=640, quality=80, use_grayscale=True)
    optimized_image = optimizer.optimize_image(image_path)
    
    prompt = """Analyze this image searching for a weapon, people, potencial danger and respond ONLY with a JSON object in the following format:
{
  "image_context": "brief description of the scene",
  "has_weapon": true/false,
  "has_people": true/false,
  "confidence": number between 0-100
}"""

    start_time = time.time()
    
    response = chat(
        model=model,
        format=ImageDescription.model_json_schema(),
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [optimized_image]
        }],
        options={
            "num_ctx": 512,         # Contexto ainda mais reduzido
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
        # Extrai apenas o JSON da resposta
        response_text = response['message']['content']
        json_str = response_text.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:-3]  # Remove ```json e ``` se presentes
        result = json.loads(json_str)
    except json.JSONDecodeError:
        result = {
            "image_context": "Error parsing image",
            "has_weapon": False,
            "has_people": False,
            "confidence": 0
        }
    
    return result, execution_time

if __name__ == "__main__":
    image_path = 'image.png'

    print("Analisando imagem...")
    result, exec_time = process_image(image_path)
    print("\nRESULTADO:")
    print(json.dumps(result, indent=2))
    print(f"\nTempo de execução: {exec_time:.2f} segundos")