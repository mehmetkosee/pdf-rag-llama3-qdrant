from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Modeli ve Veritabanı İstemcisini Yükle
print("Model yükleniyor...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient(url="http://localhost:6333")

koleksiyon_adi = "test_koleksiyonu"

# 2. Kullanıcının Sorusu
kullanici_sorusu = "Banka kartım çalışmıyor ne yapmalıyım?"

# 3. Soruyu Vektöre Çevir
soru_vektoru = model.encode(kullanici_sorusu).tolist()

# 4. Qdrant İçinde Arama Yap (Güncel Metod: query_points)
arama_sonuclari = client.query_points(
    collection_name=koleksiyon_adi,
    query=soru_vektoru,
    limit=2
).points

# 5. Sonuçları Ekrana Bas
print(f"\nSoru: '{kullanici_sorusu}'\n")
print("Bulunan En Yakın Sonuçlar:")
for sonuc in arama_sonuclari:
    benzerlik_skoru = sonuc.score
    orijinal_metin = sonuc.payload['orijinal_metin']
    print(f"- Skor: {benzerlik_skoru:.4f} | Metin: {orijinal_metin}")