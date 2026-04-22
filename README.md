# 📄 Akıllı Doküman Analiz Asistanı (RAG Architecture)

Bu proje, yapısal olmayan dokümanları (PDF) analiz ederek kullanıcı sorularına bağlama dayalı (context-aware) yanıtlar üreten, uçtan uca tasarlanmış bir **Retrieval-Augmented Generation (RAG)** asistanıdır. Yapay zeka modelinin halüsinasyon görmesini (bilgi uydurmasını) engellemek amacıyla, yanıtlar sadece vektör veritabanından çekilen doküman parçalarına dayandırılır ve diyalog bütünlüğü ilişkisel bir veritabanı ile korunur.

## 📸 Arayüz Görseli

![Uygulama Demo](/app_demo.png)

##  Proje Mimarisi ve Teknoloji Yığını (Tech Stack)

Sistem, modern yapay zeka ve arka uç (backend) mühendisliği standartlarına uygun olarak modüler bir yapıda geliştirilmiştir:

* **Frontend / UI:** [Streamlit](https://streamlit.io/) (Etkileşimli web arayüzü ve oturum yönetimi)
* **LLM (Büyük Dil Modeli):** Llama-3.1-8b-instant (Groq API üzerinden düşük gecikmeli çıkarım)
* **Vektör Veritabanı (Semantic Search):** [Qdrant](https://qdrant.tech/) (Docker üzerinden yerel hosting)
* **Gömme Modeli (Embedding):** `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers)
* **İlişkisel Veritabanı (Memory/State):** PostgreSQL & SQLAlchemy ORM (Kullanıcı sohbet geçmişi)
* **Doküman İşleme (Data Ingestion):** PyPDF2 (Metin çıkarma ve 400 karakterlik parçalara/chunk'lara bölme)

##  Temel Özellikler

1.  **Dinamik Veri Sindirimi:** Sisteme yüklenen PDF dosyalarındaki metinler anında okunur, vektörleştirilir (UUID atanarak) ve Qdrant'a kaydedilir.
2.  **Bağlamsal Doğruluk:** RAG mimarisi sayesinde, model sadece kendi eğitim verisine değil, kullanıcının yüklediği özel dokümanlara sadık kalarak cevap üretir.
3.  **Kalıcı Hafıza:** PostgreSQL entegrasyonu ile sohbet geçmişi kaydedilir. Asistan, diyalog bağlamını unutmaz.
4.  **Geçmişi Yönetme:** Kullanıcı dilediği zaman sohbet geçmişini tek tuşla veritabanından tamamen silebilir (CRUD - Delete operasyonu).

##  Kurulum ve Çalıştırma (Local Development)

Projeyi kendi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADIN/Akilli-Dokuman-Asistani.git](https://github.com/KULLANICI_ADIN/Akilli-Dokuman-Asistani.git)
cd Akilli-Dokuman-Asistani
```

### 2. Sanal Ortam ve Bağımlılıkları Kurun

```bash
python -m venv venv
# Windows için: .\venv\Scripts\activate
# Mac/Linux için: source venv/bin/activate

pip install -r requirements.txt
```
### 3. Çevre Değişkenlerini Ayarlayın

Proje dizininde .env adında bir dosya oluşturun ve Groq API anahtarınızı ekleyin:
```bash
GROQ_API_KEY=sizin_api_anahtariniz_buraya
```

### 4. Veritabanlarını Ayaklandırın (Docker)
Sistemin çalışması için Qdrant (Port: 6333) ve PostgreSQL (Port: 5432) konteynerlerinin aktif olması gerekmektedir:

```bash
# Qdrant'ı başlatmak için
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant

# PostgreSQL'i başlatmak için
docker run --name rag_postgres -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=1234 -e POSTGRES_DB=rag_hafiza -p 5432:5432 -d postgres
```

### 5. Uygulamayı Başlatın
```bash
streamlit run arayuz.py
```