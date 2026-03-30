import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from groq import Groq
from veritabani import SessionLocal, SohbetGecmisi
import PyPDF2
import uuid

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Akıllı Doküman Analiz Asistanı", page_icon="📄")
st.title("📄 Akıllı Doküman Analiz Asistanı")
st.markdown("Yüklediğiniz PDF dosyalarını okur, analiz eder ve doküman içindeki spesifik sorularınızı yanıtlar.")

# --- 2. SİSTEM YÜKLEMESİ ---
@st.cache_resource
def sistem_yukle():
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    qdrant_client = QdrantClient(url="http://localhost:6333")
    llm_client = Groq(api_key="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")
    return embedding_model, qdrant_client, llm_client

embedding_model, qdrant_client, llm_client = sistem_yukle()
koleksiyon_adi = "test_koleksiyonu"
aktif_kullanici = "kullanici_1"

# Veritabanı bağlantısını başlat
db = SessionLocal()

# --- 3. YAN MENÜ (SİDEBAR) VE DOSYA YÜKLEME ---
with st.sidebar:
    st.header("📂 Doküman Yükle")
    st.markdown("Analiz edilmesini istediğiniz PDF dosyasını yükleyin.")
    
    yuklenen_dosya = st.file_uploader("Bir PDF dosyası yükleyin", type="pdf")
    
    # PDF İşleme Butonu
    if st.button("Dokümanı Analiz Et") and yuklenen_dosya is not None:
        with st.spinner("PDF okunuyor ve uzaysal vektörlere dönüştürülüyor..."):
            
            pdf_okuyucu = PyPDF2.PdfReader(yuklenen_dosya)
            tum_metin = ""
            for sayfa in pdf_okuyucu.pages:
                metin = sayfa.extract_text()
                if metin:
                    tum_metin += metin + "\n"
            
            chunk_boyutu = 400
            parcalar = [tum_metin[i:i+chunk_boyutu] for i in range(0, len(tum_metin), chunk_boyutu)]
            
            if parcalar:
                vektorler = embedding_model.encode(parcalar)
                noktalar = []
                for metin_parcasi, vektor in zip(parcalar, vektorler):
                    noktalar.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vektor.tolist(),
                            payload={"orijinal_metin": metin_parcasi, "kaynak": yuklenen_dosya.name}
                        )
                    )
                qdrant_client.upsert(collection_name=koleksiyon_adi, points=noktalar)
                st.success(f"{yuklenen_dosya.name} başarıyla belleğe alındı! Artık sorular sorabilirsiniz.")

    # Sohbet Geçmişini Temizleme Butonu
    st.markdown("---")
    if st.button("🗑️ Sohbet Geçmişini Temizle"):
        db.query(SohbetGecmisi).filter(SohbetGecmisi.kullanici_id == aktif_kullanici).delete()
        db.commit()
        st.rerun()

# --- 4. ANA SOHBET EKRANI (RAG MİMARİSİ) ---

# Eski mesajları ekrana bas
gecmis_kayitlar = db.query(SohbetGecmisi).filter(SohbetGecmisi.kullanici_id == aktif_kullanici).order_by(SohbetGecmisi.id.asc()).all()

for kayit in gecmis_kayitlar:
    with st.chat_message("user"):
        st.write(kayit.soru)
    with st.chat_message("assistant"):
        st.write(kayit.cevap)

# Kullanıcıdan soru al
kullanici_sorusu = st.chat_input("Doküman hakkında spesifik bir soru sorun (Örn: Yetenekleri nelerdir?)...")

if kullanici_sorusu:
    # Soruyu ekrana yazdır
    with st.chat_message("user"):
        st.write(kullanici_sorusu)

    with st.chat_message("assistant"):
        with st.spinner("Doküman taranıyor..."):
            
            # Soruya en yakın 4 metin parçasını Qdrant'tan çek
            soru_vektoru = embedding_model.encode(kullanici_sorusu).tolist()
            arama_sonuclari = qdrant_client.query_points(
                collection_name=koleksiyon_adi,
                query=soru_vektoru,
                limit=4
            ).points
            
            baglam_metni = ""
            for sonuc in arama_sonuclari:
                baglam_metni += f"- {sonuc.payload['orijinal_metin']}\n"

            # Son 3 sohbet geçmişini PostgreSQL'den çek
            son_kayitlar = gecmis_kayitlar[-3:] if len(gecmis_kayitlar) >= 3 else gecmis_kayitlar
            gecmis_metni = ""
            for kayit in son_kayitlar:
                gecmis_metni += f"Kullanıcı: {kayit.soru}\nAsistan: {kayit.cevap}\n"

            # Modele gönderilecek kesin talimat (Prompt)
            sistem_istemi = f"""
            Sen profesyonel bir doküman analiz asistanısın. 
            Aşağıda, kullanıcının sisteme yüklediği PDF dosyasından çekilen 'Bağlam' metinleri bulunmaktadır.
            Kullanıcının sorusunu SADECE bu 'Bağlam' metinlerine dayanarak yanıtla.
            Eğer sorunun cevabı bağlamda yoksa, uydurma ve 'Bu bilgi yüklenen dokümanda bulunmamaktadır' de.
            
            Bağlam (Doküman İçeriği):
            {baglam_metni}

            Sohbet Geçmişi:
            {gecmis_metni if gecmis_metni else "İlk konuşma."}
            """

            # Groq API ile yanıt üret
            llm_yaniti = llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sistem_istemi},
                    {"role": "user", "content": kullanici_sorusu}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0
            )
            
            asistan_cevabi = llm_yaniti.choices[0].message.content
            
            # Cevabı ekrana yazdır
            st.write(asistan_cevabi)

            # Konuşmayı PostgreSQL veritabanına kaydet
            yeni_kayit = SohbetGecmisi(
                kullanici_id=aktif_kullanici,
                soru=kullanici_sorusu,
                cevap=asistan_cevabi
            )
            db.add(yeni_kayit)
            db.commit()