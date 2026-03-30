from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Embedding Modelini Yükle (Türkçe uyumlu 384 boyutlu modelimiz)
print("Model yükleniyor...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Qdrant Veritabanına Bağlan
# Docker'da 6333 portunu açtığımız için doğrudan localhost'a bağlanıyoruz.
client = QdrantClient(url="http://localhost:6333")

koleksiyon_adi = "test_koleksiyonu"

# 3. Koleksiyon (Tablo) Oluşturma
# Eğer bu isimde bir koleksiyon yoksa yeni baştan oluşturuyoruz.
if not client.collection_exists(collection_name=koleksiyon_adi):
    client.create_collection(
        collection_name=koleksiyon_adi,
        vectors_config=models.VectorParams(
            size=384,  # Modelimizin ürettiği vektör boyutu
            distance=models.Distance.COSINE  # Benzerlik ölçüm matematiğimiz
        )
    )
    print(f"'{koleksiyon_adi}' başarıyla oluşturuldu.")

# 4. Kaydedilecek Verilerimiz (Çözüm odaklı şirket dokümanları)
metinler = [
    "Kredi kartı reddedilen veya ödeme işleminde hata alan müşteriler, destek için 0850 123 45 67 numaralı hattı aramalı veya kimlikleriyle şubeye başvurmalıdır.",
    "Şifresini unutan kullanıcılar mobil uygulama üzerinden 'Şifremi Unuttum' sekmesine tıklayarak SMS onayı ile yeni şifre alabilirler.",
    "Bugun hava gercekten cok gunesli."
]

# Metinleri vektörlere (sayılara) çevir
vektorler = model.encode(metinler)

# 5. Qdrant'a Kayıt Formatını Hazırlama (Points)
noktalar = []
for i, (metin, vektor) in enumerate(zip(metinler, vektorler)):
    noktalar.append(
        models.PointStruct(
            id=i + 1,  # Her veriye benzersiz bir ID veriyoruz (1, 2, 3...)
            vector=vektor.tolist(),  # NumPy dizisini standart Python listesine çeviriyoruz
            payload={"orijinal_metin": metin, "kaynak": "kullanici_testi"}  # İnsan okuyabilsin diye metni buraya koyuyoruz
        )
    )

# 6. Veritabanına Yazma (Upsert)
client.upsert(
    collection_name=koleksiyon_adi,
    points=noktalar
)

print("Veriler Qdrant veritabanına başarıyla kaydedildi!")