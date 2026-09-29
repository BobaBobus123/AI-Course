import io
import os
import base64
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()
MAX_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1600"))

def prepare_image(image_bytes: bytes) -> tuple[str, dict]:
    """
    Відкриває зображення, вирівнює по EXIF, зменшує та кодує в Data URL.
    """
    try:
        # Відкриваємо зображення з пам'яті
        img = Image.open(io.BytesIO(image_bytes))
        
        # Перевірка та конвертація формату
        if img.format not in ['JPEG', 'PNG', 'WEBP']:
            img = img.convert('RGB')
            
        orig_size = img.size
        
        # Враховуємо орієнтацію EXIF (якщо фото з телефону)
        img = ImageOps.exif_transpose(img)
        
        # Зменшуємо зображення для економії токенів та часу
        if max(img.size) > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
            
        new_size = img.size
        
        # Кодуємо в base64 JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        data_url = f"data:image/jpeg;base64,{img_b64}"
        
        return data_url, {"original_size": orig_size, "new_size": new_size}
        
    except Exception as e:
        raise ValueError(f"Файл не є дійсним зображенням або пошкоджений: {str(e)}")