from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from pypdf import PdfReader
import os

# --- CHUNKING ---
def chunking(metin, chunk_size=300, overlap=50):
    kelimeler = metin.split()
    parcalar = []

    for i in range(0, len(kelimeler), chunk_size - overlap):
        chunk = " ".join(kelimeler[i:i+chunk_size])
        if len(chunk) > 50:
            parcalar.append(chunk)

    return parcalar


# --- MODEL + QDRANT ---
print("Model yükleniyor...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

client = QdrantClient(url="http://localhost:6333")
koleksiyon_adi = "test_koleksiyonu"


# --- COLLECTION CHECK ---
if koleksiyon_adi not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=koleksiyon_adi,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )
    print("Collection oluşturuldu.")


# --- PDF DOSYASI ---
pdf_path = "ornek.pdf"   # buraya kendi pdf'ini koy

if not os.path.exists(pdf_path):
    raise FileNotFoundError("PDF dosyası bulunamadı!")

reader = PdfReader(pdf_path)

full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"


# --- CHUNKING ---
parcalar = chunking(full_text)

print(f"{len(parcalar)} chunk oluşturuldu.")


# --- EMBEDDING ---
print("Embedding oluşturuluyor...")
vektorler = model.encode(
    parcalar,
    show_progress_bar=True,
    normalize_embeddings=True
)


# --- QDRANT INSERT ---
points = []

for text, vector in zip(parcalar, vektorler):
    points.append(
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "orijinal_metin": text,
                "kaynak": pdf_path
            }
        )
    )

client.upsert(
    collection_name=koleksiyon_adi,
    points=points
)

print("PDF başarıyla Qdrant'a yüklendi!")