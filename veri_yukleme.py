# Bu dosya, kurumsal bilgi bankasından ham veriyi alır, metni parçalara böler (chunking), her parçayı vektörleştirir (embedding) ve Qdrant veritabanına kaydeder.
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Modelleri ve Veritabanı Bağlantısını Yükle
print("Model yükleniyor...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient(url="http://localhost:6333")
koleksiyon_adi = "test_koleksiyonu"

# 2. Kurumsal Bilgi Bankası (Ham Veri)
# Gerçek bir projede bu veri bir PDF okuyucu (PyPDF2) veya veritabanından gelir.
sirket_verisi = """
Bankaya giderken kimlik belgesiyle şubeye gidilmelidir.
Kredi kartı limit artırım işlemleri için müşterilerimizin en az 6 aydır bankamızla çalışıyor olması gerekmektedir. Limit artırımı mobil uygulama üzerinden 'Kartlarım > Ayarlar > Limit Belirle' menüsünden yapılabilir.
Kredi kartının kaybolması veya çalınması durumunda anında 0850 123 45 67 numaralı çağrı merkezimiz aranmalı ve kart kullanıma kapatılmalıdır. Aksi takdirde doğacak zararlardan bankamız sorumlu değildir.
Yurtdışı harcamalarında kredi kartı kullanımı için müşterinin mobil uygulamadan 'Yurtdışı Kullanımına Aç' seçeneğini aktif etmesi şarttır.
İhtiyaç kredisi faiz oranlarımız 24 ay vadeye kadar yüzde 3.15, 24-36 ay arası vadelerde ise yüzde 3.45 olarak güncellenmiştir.
EFT işlemleri hafta içi saat 09:00 ile 16:45 arasında yapılabilir. Havale işlemleri ise 7 gün 24 saat kesintisiz olarak gerçekleştirilmektedir.
"""

# 3. Chunking (Metin Parçalama) İşlemi
# Metni satır sonu karakterlerine (\n) göre bölerek her bir kuralı bağımsız bir "Chunk" haline getiriyoruz.
ham_cumleler = sirket_verisi.strip().split('\n')
parcalar = [cumle.strip() for cumle in ham_cumleler if len(cumle.strip()) > 10]

print(f"Toplam {len(parcalar)} adet bağımsız bilgi parçası (chunk) oluşturuldu.")

# 4. Vektörleştirme (Embedding)
print("Metinler vektörlere dönüştürülüyor...")
vektorler = model.encode(parcalar)

# 5. Qdrant'a Kayıt İçin Paketleme (Payload Hazırlığı)
noktalar = []
# Qdrant'ta eski verilerimizin (ID 1, 2, 3) üzerine yazmamak için yeni ID'leri 100'den başlatıyoruz.
baslangic_id = 100 

for i, (metin, vektor) in enumerate(zip(parcalar, vektorler)):
    noktalar.append(
        models.PointStruct(
            id=baslangic_id + i, 
            vector=vektor.tolist(),
            payload={"orijinal_metin": metin, "kaynak": "kurumsal_bilgi_bankasi_v1"}
        )
    )

# 6. Veritabanına İletme (Upsert)
client.upsert(
    collection_name=koleksiyon_adi,
    points=noktalar
)

print("Yeni kurumsal veriler Qdrant veritabanına başarıyla eklendi!")