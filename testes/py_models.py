from pydantic import BaseModel
from PIL import Image

class ImageDescription(BaseModel):
    image_context: str
    has_weapon: bool
    has_people: bool
    confidence: int

class ImageOptimizer:
    def __init__(self, max_size=800, quality=90, use_grayscale=False):
        self.max_size = max_size
        self.quality = quality
        self.use_grayscale = use_grayscale

    def optimize_image(self, image_path):
        # Abre e otimiza a imagem
        img = Image.open(image_path)

        # Converte para escala de cinza se configurado
        if self.use_grayscale and img.mode in ('RGB', 'RGBA'):
            img = img.convert('L')
        elif img.mode != 'RGB':  # Garante que a imagem está em RGB se não for grayscale
            img = img.convert('RGB')

        # Redimensiona se necessário, mantendo a proporção
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
