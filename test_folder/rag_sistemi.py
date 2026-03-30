# Arayüz (Streamlit) yazmadan önce terminalde sohbet ettiğimiz eski ana dosyamızdır. Burada API istemcileri, Qdrant sorguları, LLM çağrıları ve PostgreSQL işlemleri tek bir dosyada toplanmıştı. Şimdi bu dosyayı sadece arayüzün çalıştıracağı kodları içerecek şekilde düzenledik.
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from veritabani import SessionLocal, SohbetGecmisi  # PostgreSQL bağlantılarımız

# 1. API Anahtarları ve İstemciler
GROQ_API_KEY = "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"

print("Sistem başlatılıyor...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
qdrant_client = QdrantClient(url="http://localhost:6333")
llm_client = Groq(api_key=GROQ_API_KEY)

# Veritabanı oturumunu açıyoruz
db = SessionLocal()

koleksiyon_adi = "test_koleksiyonu"
aktif_kullanici = "kullanici_1"  # Sistemi kullanan kişinin ID'si

# 2. Kullanıcı Sorusu
# Önceki sorumuzun devamı niteliğinde bir soru soruyoruz
kullanici_sorusu = "Peki oraya giderken yanıma ne almalıyım?"

# 3. Qdrant'tan Şirket Verisi Çekme (Retrieval)
soru_vektoru = embedding_model.encode(kullanici_sorusu).tolist()
arama_sonuclari = qdrant_client.query_points(
    collection_name=koleksiyon_adi,
    query=soru_vektoru,
    limit=2
).points

baglam_metni = ""
for sonuc in arama_sonuclari:
    baglam_metni += f"- {sonuc.payload['orijinal_metin']}\n"

# 4. PostgreSQL'den Sohbet Geçmişini Çekme (Hafıza)
# Bu kullanıcının veritabanındaki son 3 konuşmasını tarihe göre sondan başa doğru çekiyoruz.
gecmis_kayitlar = db.query(SohbetGecmisi)\
    .filter(SohbetGecmisi.kullanici_id == aktif_kullanici)\
    .order_by(SohbetGecmisi.id.desc())\
    .limit(3).all()

# Doğru kronolojik sıra için listeyi tersine çeviriyoruz (Eskiden yeniye)
gecmis_kayitlar.reverse()

gecmis_metni = ""
for kayit in gecmis_kayitlar:
    gecmis_metni += f"Kullanıcı: {kayit.soru}\nAsistan: {kayit.cevap}\n"

# 5. Generation (Üretim) Aşaması - LLM'e Gönder
sistem_istemi = f"""
Sen profesyonel bir müşteri destek asistanısın.
Aşağıda sana şirket veritabanından çekilmiş 'Bağlam' bilgileri ve kullanıcının seninle yaptığı 'Sohbet Geçmişi' verilmiştir.

1. Kullanıcının eksik veya zamir (o, oraya, bunu) içeren sorularını 'Sohbet Geçmişi'ne bakarak tamamla.
2. Cevabını SADECE 'Bağlam'daki bilgilere dayanarak ver. 

Bağlam:
{baglam_metni}

Sohbet Geçmişi:
{gecmis_metni if gecmis_metni else "Bu kullanıcıyla ilk konuşman."}
"""

print("LLM cevabı hesaplanıyor...\n")
llm_yaniti = llm_client.chat.completions.create(
    messages=[
        {"role": "system", "content": sistem_istemi},
        {"role": "user", "content": kullanici_sorusu}
    ],
    model="llama-3.1-8b-instant",
    temperature=0.0
)

asistan_cevabi = llm_yaniti.choices[0].message.content
print("--- Yapay Zeka Asistanının Yanıtı ---")
print(asistan_cevabi)

# 6. PostgreSQL'e Yeni Konuşmayı Kaydetme (INSERT İşlemi)
# SQLAlchemy kullanarak yeni bir satır oluşturuyoruz
yeni_kayit = SohbetGecmisi(
    kullanici_id=aktif_kullanici,
    soru=kullanici_sorusu,
    cevap=asistan_cevabi
)

db.add(yeni_kayit)  # Veriyi oturuma ekle
db.commit()         # Veritabanına kalıcı olarak yaz (Commit)
print("\n[BİLGİ] Bu konuşma PostgreSQL veritabanına kaydedildi.")