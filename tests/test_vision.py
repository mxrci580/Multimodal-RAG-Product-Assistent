from app.vision import VisionAnalyzer

vision = VisionAnalyzer()

description = vision.analyze_image(
    "data/test_images/image.png"
)

print(description)