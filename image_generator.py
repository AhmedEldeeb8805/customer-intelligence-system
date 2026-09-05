from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN
)

IMAGES_FOLDER = "images/products"

# التأكد من وجود المجلد، وإنشاؤه لو مش موجود
os.makedirs(IMAGES_FOLDER, exist_ok=True)


def get_or_generate_image(stock_code, description):
    image_path = os.path.join(IMAGES_FOLDER, f"{stock_code}.png")
    
    if os.path.exists(image_path):
        return image_path
    
    try:
        image = client.text_to_image(description, model="black-forest-labs/FLUX.1-dev")
        image.save(image_path)
        return image_path
    except Exception:
        # لو التوليد فشل (زي نفاذ الرصيد)، رجّع None بدل ما توقف كل حاجة
        return None