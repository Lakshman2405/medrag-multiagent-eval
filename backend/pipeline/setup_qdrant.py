from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "medical_kb"

embedder = SentenceTransformer(MODEL_NAME)

import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 🔥 Create collection
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 🔥 Sample medical data (replace later if needed)
documents = [
    "Myocardial infarction presents with chest pain and sweating",
    "Asthma causes wheezing and shortness of breath",
    "Diabetes leads to high blood sugar levels",
    "Tuberculosis affects lungs and causes chronic cough",
    "Stroke leads to sudden neurological deficit"
]

points = []

for i, doc in enumerate(documents):
    vector = embedder.encode(doc).tolist()

    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={"text": doc}
        )
    )

qdrant.upsert(
    collection_name=COLLECTION,
    points=points
)

print("✅ Qdrant setup complete")