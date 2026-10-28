from google import genai
from PIL import Image
from app.config import GOOGLE_API_KEY, LLM_MODEL, FALLBACK_LLM_MODELS
from app.gemini_utils import generate_content_with_retry


class VisionAnalyzer:

    def __init__(self, api_key=None):
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please configure GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self.client = genai.Client(
            api_key=key
        )

    def analyze_image(
        self,
        image_path
    ):

        image = Image.open(image_path)

        prompt = """
Identify the product.

Return only:
- Product Type
- Brand
- Category
- Key Features
- Search Keywords

Maximum 50 words.
"""

        response = generate_content_with_retry(
            client=self.client,
            model=LLM_MODEL,
            contents=[prompt, image],
            fallback_models=FALLBACK_LLM_MODELS,
            max_retries=3,
            initial_delay=2.0
        )

        return response.text