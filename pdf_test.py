import PyPDF2

# 1. Dosya Yolu ve İkili Okuma (Binary Read)
pdf_dosyasi = "ornek.pdf"

print(f"{pdf_dosyasi} okunuyor...\n")

try:
    with open(pdf_dosyasi, "rb") as dosya: # 'rb' modunda açarak dosyayı ikili (binary) formatta okuyoruz, bu PDF gibi özel formatlı dosyalar için gereklidir.
        
        # 2. PyPDF2 Okuyucu Nesnesini (Parser) Başlatma
        pdf_okuyucu = PyPDF2.PdfReader(dosya)
        
        # Dosyanın içindeki toplam sayfa sayısını bulma
        toplam_sayfa = len(pdf_okuyucu.pages)
        print(f"Toplam Sayfa Sayısı: {toplam_sayfa}\n")
        
        # 3. Metin Çıkarma (Text Extraction)
        # Sadece ilk sayfayı (indeks 0) okuyup içindeki görünmez metin katmanını çekiyoruz
        ilk_sayfa = pdf_okuyucu.pages[0]
        cikarilan_metin = ilk_sayfa.extract_text()
        
        print("--- İlk Sayfadan Çıkarılan Ham Metin ---")
        print(cikarilan_metin)

except FileNotFoundError:
    print(f"[HATA] Klasörde '{pdf_dosyasi}' bulunamadı. Lütfen PDF dosyasını eklediğinden emin ol.")
except Exception as e:
    print(f"[HATA] Dosya okunurken beklenmeyen bir sorun oluştu: {e}")