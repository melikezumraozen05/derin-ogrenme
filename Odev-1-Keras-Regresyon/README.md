# Ödev 1 — Keras Regresyon

Melike Zümra Özen • 363426821

Bu, önceki `ikili_siniflandirici.py` ödevinden ayrı bir regresyon çalışmasıdır. Eski ödev korunmuştur.

- `melike_regresyon_odev1.py`: 25 deneyi yapan Keras/PyTorch Python kodu.
- `sonuclar/melike_odev1_sonuclar.csv`: Gerçek 25 deneyin sonuçları.
- `rapor_melike_metni.md`: Son Word belgesinin tabloları ve yorumlarının metin önizlemesi.
- `requirements.txt`: Python gereksinimleri.

**Önemli:** Son gönderilen `ödev1_rapor_melike(1).docx` dosyasının aslı henüz buraya yüklenmemiştir. Word dosyasındaki grafiklerin ve sayfa düzeninin korunması için özgün DOCX ayrıca yüklenmelidir; bu metin önizlemesi Word tesliminin yerine geçmez.

## Çalıştırma

```powershell
pip install -r requirements.txt
python melike_regresyon_odev1.py
```

21 seed, 300 epoch, 201 örnek, batch size=201. Her tabloda en düşük MSE veren ayar sonraki tabloya aktarılır. Çıktılar `sonuclar/` ve `grafikler/` altına üretilir.

**Not:** Öğretmenin yönergesindeki `y=1+sin(pi*y/4)` formülü, giriş `p` olduğundan programda `y=1+sin(pi*p/4)` olarak yorumlanmıştır. Hocadan doğrulanmalıdır. MSE ve R² ayrı bir test kümesinde değil, eğitim verisi üzerinde ölçülmüştür.
