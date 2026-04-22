import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from groq import Groq
from veritabani import SessionLocal, SohbetGecmisi
import PyPDF2
import uuid
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# --- CHUNKING ---
def chunking(metin, chunk_size=300, overlap=50):
    kelimeler = metin.split()
    parcalar = []
    
    for i in range(0, len(kelimeler), chunk_size - overlap):
        chunk = " ".join(kelimeler[i:i+chunk_size])
        if len(chunk) > 50:
            parcalar.append(chunk)
    
    return parcalar

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Akıllı Doküman Analiz Asistanı", page_icon="📄")
st.title("📄 Yapay Zeka Destekli Doküman Analiz Asistanı")
st.markdown("Yüklediğiniz PDF dosyalarını okur, analiz eder ve doküman içindeki spesifik sorularınızı yanıtlar.")

# --- 2. SİSTEM YÜKLEMESİ ---
@st.cache_resource
def sistem_yukle():
    print("API KEY:", os.getenv("GROQ_API_KEY"))
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    qdrant_client = QdrantClient(url="http://localhost:6333")
    
    koleksiyon_adi = "test_koleksiyonu"

    # ✅ COLLECTION KONTROL (FIXED)
    if koleksiyon_adi not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=koleksiyon_adi,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    api_anahtari = os.getenv("GROQ_API_KEY")
    llm_client = Groq(api_key=api_anahtari)
    
    return embedding_model, qdrant_client, llm_client

embedding_model, qdrant_client, llm_client = sistem_yukle()
koleksiyon_adi = "test_koleksiyonu"
aktif_kullanici = "kullanici_1"

# --- DB ---
db = SessionLocal()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📂 Doküman Yükle")
    st.markdown("Analiz edilmesini istediğiniz PDF dosyasını yükleyin.")
    
    yuklenen_dosya = st.file_uploader("Bir PDF dosyası yükleyin", type="pdf")
    
    if st.button("Dokümanı Analiz Et") and yuklenen_dosya is not None:
        with st.spinner("PDF işleniyor..."):
            
            pdf_okuyucu = PyPDF2.PdfReader(yuklenen_dosya)
            tum_metin = ""
            
            for sayfa in pdf_okuyucu.pages:
                metin = sayfa.extract_text()
                if metin:
                    tum_metin += metin + "\n"
            
            parcalar = chunking(tum_metin)

            if parcalar:
                vektorler = embedding_model.encode(
                    parcalar,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )

                noktalar = []
                for metin_parcasi, vektor in zip(parcalar, vektorler):
                    noktalar.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vektor.tolist(),
                            payload={
                                "orijinal_metin": metin_parcasi,
                                "kaynak": yuklenen_dosya.name
                            }
                        )
                    )

                qdrant_client.upsert(
                    collection_name=koleksiyon_adi,
                    points=noktalar
                )

                st.success(f"{yuklenen_dosya.name} başarıyla yüklendi!")

    st.markdown("---")

    if st.button("🗑️ Sohbet Geçmişini Temizle"):
        db.query(SohbetGecmisi).filter(
            SohbetGecmisi.kullanici_id == aktif_kullanici
        ).delete()
        db.commit()
        st.rerun()

# --- 4. CHAT ---
gecmis_kayitlar = db.query(SohbetGecmisi)\
    .filter(SohbetGecmisi.kullanici_id == aktif_kullanici)\
    .order_by(SohbetGecmisi.id.asc())\
    .all()

for kayit in gecmis_kayitlar:
    with st.chat_message("user"):
        st.write(kayit.soru)
    with st.chat_message("assistant"):
        st.write(kayit.cevap)

kullanici_sorusu = st.chat_input("Doküman hakkında soru sorun...")

if kullanici_sorusu:

    with st.chat_message("user"):
        st.write(kullanici_sorusu)

    with st.chat_message("assistant"):
        with st.spinner("Doküman taranıyor..."):

            # --- RETRIEVAL ---
            soru_vektoru = embedding_model.encode(
                kullanici_sorusu,
                normalize_embeddings=True
            ).tolist()

            arama_sonuclari = qdrant_client.query_points(
                collection_name=koleksiyon_adi,
                query=soru_vektoru,
                limit=8
            ).points

            # ✅ GÜVENLİ PAYLOAD
            baglam_listesi = [
                sonuc.payload.get("orijinal_metin", "")
                for sonuc in arama_sonuclari
                if sonuc.payload
            ]

            # ✅ BOŞ KONTROL
            if not baglam_listesi:
                baglam_metni = "Bağlam bulunamadı."
            else:
                baglam_listesi = baglam_listesi[:4]
                baglam_metni = "\n".join([f"- {m}" for m in baglam_listesi])

            # --- MEMORY ---
            son_kayitlar = gecmis_kayitlar[-3:] if len(gecmis_kayitlar) >= 3 else gecmis_kayitlar
            gecmis_metni = ""

            for kayit in son_kayitlar:
                gecmis_metni += f"Kullanıcı: {kayit.soru}\nAsistan: {kayit.cevap}\n"

            # --- PROMPT ---
            sistem_istemi = f"""
Sen profesyonel bir doküman analiz asistanısın.

KURALLAR:
- SADECE verilen bağlamı kullan
- Uydurma, tahmin yapma
- Cevap yoksa: "Bu bilgi yüklenen dokümanda bulunmamaktadır" de
- Kısa ve net cevap ver

Bağlam:
{baglam_metni}

Sohbet geçmişi:
{gecmis_metni if gecmis_metni else "İlk konuşma."}
"""

            # --- LLM ---
            llm_yaniti = llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sistem_istemi},
                    {"role": "user", "content": kullanici_sorusu}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0
            )

            asistan_cevabi = llm_yaniti.choices[0].message.content

            st.write(asistan_cevabi)

            # --- SAVE ---
            yeni_kayit = SohbetGecmisi(
                kullanici_id=aktif_kullanici,
                soru=kullanici_sorusu,
                cevap=asistan_cevabi
            )
            db.add(yeni_kayit)
            db.commit()