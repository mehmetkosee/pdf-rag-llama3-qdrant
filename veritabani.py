# Bu dosya, PostgreSQL veritabanı bağlantısı kurar, sohbet geçmişi tablosunu tanımlar ve oluşturur.
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# 1. Bağlantı Adresi (Connection String)
# Format: veritabani_turu://kullanici_adi:sifre@sunucu_adresi:port/veritabani_adi
DATABASE_URL = "postgresql://admin:1234@localhost:5432/rag_hafiza"

# 2. Veritabanı Motoru (Engine)
# Python ile PostgreSQL arasındaki fiziksel bağlantıyı kuran ana merkezdir.
# echo=True parametresi, SQLAlchemy'nin arka planda ürettiği ham SQL komutlarını terminalde görmemizi sağlar.
engine = create_engine(DATABASE_URL, echo=True)

# 3. Temel Sınıf (Base Class)
# ORM yapısında tüm tablolarımız bu Base sınıfından miras alarak oluşturulur.
Base = declarative_base()

# 4. Tablo Tasarımı (Model)
class SohbetGecmisi(Base):
    __tablename__ = 'sohbet_gecmisi'  # PostgreSQL içinde görünecek gerçek tablo adı

    # Sütun tanımlamaları
    id = Column(Integer, primary_key=True, autoincrement=True) # Her satırın benzersiz numarası
    kullanici_id = Column(String(50), nullable=False, default="varsayilan_kullanici") # Sistemi kimin kullandığı
    soru = Column(Text, nullable=False) # Kullanıcının LLM'e sorduğu soru
    cevap = Column(Text, nullable=False) # LLM'in ürettiği yanıt
    tarih = Column(DateTime, default=datetime.utcnow) # İşlemin yapıldığı anın zaman damgası

# 5. Tabloları Veritabanına İşleme (Migration)
print("Veritabanına bağlanılıyor ve tablolar oluşturuluyor...")
# Bu komut, Base sınıfına bağlı tüm modelleri tarar ve veritabanında yoksa SQL 'CREATE TABLE' komutu çalıştırarak oluşturur.
Base.metadata.create_all(engine)
print("Veritabanı tabloları başarıyla oluşturuldu!")

# 6. Oturum (Session) Fabrikası
# Veri ekleme ve okuma işlemleri için veritabanı ile açacağımız güvenli iletişim kanalıdır.
SessionLocal = sessionmaker(bind=engine)