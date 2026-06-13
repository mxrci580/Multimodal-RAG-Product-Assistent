from google import genai
from PIL import Image
from app.config import GOOGLE_API_KEY

class VisionAnalyzer:

    def __init__(self):

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

    def analyze_image(
        self,
        image_path
    ):

        image = Image.open(image_path)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                """
Identify the product.

Return only:
- Product Type
- Brand
- Category
- Key Features
- Search Keywords

Maximum 50 words.
""",
                image
            ]
        )

        return response.text