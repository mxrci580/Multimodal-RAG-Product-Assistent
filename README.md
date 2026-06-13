# 🛍️ Multimodal Product Recommendation System

An AI-powered product recommendation system that combines:

- Gemini Vision
- ChromaDB Vector Search
- Retrieval-Augmented Generation (RAG)
- Streamlit UI

Users can upload a product image and receive similar product recommendations using multimodal AI.

---

## Features

✅ Product image upload

✅ Gemini Vision image understanding

✅ Semantic vector search using ChromaDB

✅ Retrieval-Augmented Generation (RAG)

✅ AI-generated product recommendations

✅ Streamlit web interface

✅ Docker support

---

## Architecture

User Uploads Image
↓
Gemini Vision
↓
Image Description
↓
Embedding Generation
↓
ChromaDB Retrieval
↓
Top Similar Products
↓
Gemini Recommendation Generation
↓
Streamlit UI

---

## Tech Stack

- Python
- Gemini API
- ChromaDB
- Streamlit
- Docker
- Vector Embeddings

---

## Project Structure

```text
.
├── app
│   ├── vision.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── multimodal_agent.py
│
├── data
│
├── tests
│
├── vector_db
│
├── streamlit_app.py
│
├── requirements.txt
│
└── README.md
```

## Installation

### Clone Repository

```bash
git clone <repo-url>
cd <repo-name>
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Running the Application

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Example Workflow

1. Upload a product image.
2. Gemini Vision analyzes the image.
3. Product description is generated.
4. ChromaDB retrieves similar products.
5. Gemini generates recommendations.
6. Results are displayed in Streamlit.

---

## Future Improvements

- AWS Deployment
- Product Re-ranking
- Hybrid Search
- Agentic Workflow
- Advanced Evaluation Metrics

---

## Author

Akash Kanwar