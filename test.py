from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN
)

def generate_image(description):
    image = client.text_to_image(
        description,
        model="black-forest-labs/FLUX.1-dev"
    )
    return image

# اختبار
image = generate_image("WHITE HANGING HEART T-LIGHT HOLDER")
image.save("test_image.png")

print("تم الحفظ! افتح test_image.png وشوف النتيجة")
from image_generator import get_or_generate_image

# اختبار مباشر لدالة الصور بس، بمنتج حقيقي من الداتا
path = get_or_generate_image("21769", "VINTAGE POST OFFICE CABINET")
print("تم الحفظ في:", path)