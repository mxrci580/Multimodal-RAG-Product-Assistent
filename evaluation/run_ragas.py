from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from app.retrieval import ProductRetriever
from app.generator import ProductGenerator
from evaluation.sample_questions import evaluation_data


retriever = ProductRetriever()
generator = ProductGenerator()

questions = []
answers = []
contexts = []
ground_truths = []

for item in evaluation_data:

    query = item["question"]

    results = retriever.search(
        query=query,
        top_k=3
    )

    docs = results["documents"][0]

    answer = generator.generate_answer(
        query,
        docs
    )

    questions.append(query)
    answers.append(answer)
    contexts.append(docs)
    ground_truths.append(
        item["ground_truth"]
    )

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ]
)

print(result)