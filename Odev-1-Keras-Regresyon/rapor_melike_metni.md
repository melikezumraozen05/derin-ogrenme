# Ödev 1 — Regresyon Probleminde Ağ Metaparametreleri

**Hazırlayan:** Melike Zümra Özen • **Okul no:** 363426821 • **Tarih:** 02.11.2026

Bu, kullanıcının son Word raporundaki tabloların ve kişisel yorumlarının **metin önizlemesidir**. Word belgesindeki grafikler bu dosyada bulunmaz.

**Deney:** Keras / PyTorch, 300 epoch, seed = 21, 201 giriş örneği, batch size = 201. `y=1+sin(π*p/4)` yorumuna göre çalışılmıştır. Özgün belgedeki formülde y/p belirsizliği vardır.

## Tablo 1. Aktivasyon fonksiyonları

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| Lineer | 0.007430 | 0.086195 | 0.98521 | 2.41 |
| Sigmoid | 0.000261 | 0.016147 | 0.99948 | 2.54 |
| **Hiperbolik Tanjant** | **0.000232** | **0.015223** | **0.99954** | **2.58** |
| ReLU | 0.001281 | 0.035787 | 0.99745 | 2.46 |
| LeakyReLU | 0.002338 | 0.048356 | 0.99535 | 2.51 |
| ELU | 0.003241 | 0.056931 | 0.99355 | 2.56 |
| SiLU | 0.005833 | 0.076373 | 0.98839 | 2.54 |

**Aktivasyon fonksiyonları değişimine ait yorumlar:** Aktivasyon fonksiyonlarını karşılaştırdığımda en düşük hatayı hiperbolik tanjant ile elde ettim (MSE:0,000232). Sigmoid de 0,000261 ile buna çok yakın bir sonuç verdi. Lineer fonksiyonda hata daha yüksek kaldığı için bu deneyde tanh ve sigmoid’in sinüs eğrisini daha iyi öğrendiğini düşünüyorum.


## Tablo 2. Öğrenme oranı

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| **0.1** | **0.000232** | **0.015223** | **0.99954** | **2.43** |
| 0.01 | 0.005775 | 0.075996 | 0.98851 | 2.53 |
| 0.001 | 0.795210 | 0.891746 | -0.58255 | 2.47 |

**Öğrenme oranı değişimine ait yorumlar:** Öğrenme oranı 0,1 olduğunda en düşük hatayı (MSE: 0,000232) aldım. Oranı 0,01 yaptığımda hata arttı, 0,001 yaptığımda ise 0,795210’a yükseldiğini gördüm. Yani bu deneyde öğrenme oranını çok küçültmek, 300 epoch sonunda iyi sonuç almamı sağlamadı.


## Tablo 3. Öğrenme oranı ayar yöntemi

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| SGD | 0.000232 | 0.015223 | 0.99954 | 2.55 |
| RMSProp | 0.010236 | 0.101173 | 0.97963 | 2.70 |
| **ADAM** | **0.000094** | **0.009720** | **0.99981** | **4.27** |

**Öğrenme oranı ayar yöntemi değişimine ait yorumlar:** Üç optimizasyon yöntemini denediğimde en düşük MSE değerini ADAM ile (0,000094) gördüm. SGD de iyi sonuç verdi ama RMSProp’un hatası daha yüksekti. ADAM biraz daha uzun sürede çalıştı ama bu deneyde tahmin başarısı açısından daha iyi çıktığını görüyorum.


## Tablo 4. Başlangıç değerleri dağılımı

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| **normal** | **0.000094** | **0.009720** | **0.99981** | **2.71** |
| uniform | 0.000097 | 0.009851 | 0.99981 | 2.92 |

**Başlangıç değerleri dağılımı değişimine ait yorumlar:** Normal dağılımda MSE 0,000094, uniform dağılımda 0,000097 çıktı. Değerler birbirine çok yakın olduğu için başlangıç dağılımının bu deneyde büyük bir fark oluşturmadığını düşünüyorum. Yine de normal dağılım çok az farkla daha iyi sonuç verdi.


## Tablo 5. Gizli katman sayısı

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| **1** | **0.000094** | **0.009720** | **0.99981** | **2.69** |
| 2 | 0.000109 | 0.010449 | 0.99978 | 3.17 |
| 5 | 0.001048 | 0.032376 | 0.99791 | 4.76 |
| 10 | 0.502488 | 0.708864 | 0.00000 | 7.33 |
| 20 | 0.502488 | 0.708864 | 0.00000 | 14.52 |

**Gizli katman sayısı değişimine ait yorumlar:** Bir gizli katmanla en düşük hatayı elde ettim (MSE: 0,000094). Katman sayısı arttığında sonuçlar iyileşmedi, özellikle 10 ve 20 katmanda hata yaklaşık 0,502488 oldu. Bu yüzden bu problemde daha fazla katman kullanmanın her zaman daha iyi olmadığını gördüm.


## Tablo 6. Gizli nöron sayısı

| Ayar | MSE | RMSE | R² | Süre (sn) |
|---|---:|---:|---:|---:|
| 2 | 0.000094 | 0.009720 | 0.99981 | 2.72 |
| **5** | **0.000090** | **0.009474** | **0.99982** | **3.12** |
| 10 | 0.000136 | 0.011656 | 0.99973 | 3.30 |
| 20 | 0.000103 | 0.010150 | 0.99979 | 2.88 |
| 50 | 0.002053 | 0.045310 | 0.99591 | 3.23 |

**Gizli nöron sayısı değişimine ait yorumlar:** Nöron sayısını değiştirince en düşük hata 5 nöronda çıktı (MSE: 0,000090). 2 nöron da buna çok yakın sonuç verdi, ama 50 nöronda hata belirgin şekilde arttı. Bence bu küçük regresyon problemi için çok fazla nöron kullanmak gerekli değil.

