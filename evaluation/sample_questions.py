evaluation_data =[
    {
    "question": "best earbuds under 1500",
    "ground_truth": "Wecool Moonwalk M1"
    },
    {
        "question": "wireless earbuds",
        "ground_truth": "boAt Airdopes 141"
    },
    {
        "question": "bluetooth earbuds",
        "ground_truth": "Noise Buds VS104"
    }
]


def get_golden_dataset():
    """Load the comprehensive 25-question golden dataset."""
    import os
    import json
    path = os.path.join(os.path.dirname(__file__), "golden_dataset", "questions.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return evaluation_data

