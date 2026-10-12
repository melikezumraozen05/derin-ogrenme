# Derin Öğrenme — İkili Sınıflandırıcı

Bu ödev, hiçbir kütüphane kullanmadan iki girdiyi 0 veya 1 olarak sınıflandıran tek nöronlu bir perceptron örneğidir. Çok katmanlı bir derin öğrenme modeli değildir; sinir ağlarına giriş için hazırlanmıştır.

## Çalıştırma

Python 3 yeterlidir. Ek paket kurmaya gerek yoktur.

Windows'ta dosyanın bulunduğu klasörde terminal açıp çalıştırın:

```powershell
py ikili_siniflandirici.py
```

Diğer sistemlerde:

```bash
python3 ikili_siniflandirici.py
```

## Ne öğreniyor?

Örnek veri olarak AND işlemi seçilmiştir. Her satır `(x1, x2, hedef)` biçimindedir. `hedef`, modelin öğrenmesi gereken doğru cevaptır.

| x1 | x2 | Hedef |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Yalnızca iki girdi de 1 olduğunda cevap 1 olur.

## Denklem ve tahmin

```text
z = w1 * x1 + w2 * x2 + b
```

- `w1` ve `w2`: Girdilerin ağırlıkları. Başlangıçta elle seçilen `0.3` ve `-0.2` değerleri kullanılır; rastgele sayı üreten bir kütüphane yoktur.
- `b`: Bias. Karar sınırının yerini değiştiren, eğitim sırasında güncellenen ek değerdir. Başlangıç değeri `0.1` olur.
- `z >= 0` ise tahmin 1; aksi durumda tahmin 0 olur.
- `ogrenme_orani = 0.1`: Her güncellemenin büyüklüğünü belirler.

## Eğitim döngüsü nasıl çalışıyor?

1. Model, mevcut ağırlıklarla dört örneğin de cevabını tahmin eder.
2. Her örnek için `fark = hedef - tahmin` hesaplanır. Doğru tahminde fark 0 olur.
3. `dw1`, `dw2` ve `db` değişkenleri, dört örneğin önerdiği düzeltmeleri toplar.
4. Dört örnek bitince bu düzeltmelerin ortalaması, öğrenme oranıyla çarpılarak ağırlıklara ve bias'a eklenir. Böylece güncelleme bütün batch için birlikte yapılır.
5. Güncel ağırlıklarla veriler tekrar değerlendirilir. Mutlak farklar toplanarak toplam hata bulunur.
6. Toplam hata 0 değilse yeni bir epoch başlar; 0 olduğunda eğitim durur.

Mutlak değer kullanıldığı için +1 ve -1 hataları birbirini götürmez. Bu veride toplam hata, yanlış sınıflandırılan örnek sayısıdır. Hata her epoch'ta azalmak zorunda değildir.

## Batch size ve epoch size varsayımı

```python
batch_size = epoch_size = len(veriler)
```

Bu ödevde `batch_size` birlikte işlenen örnek sayısı, `epoch_size` ise bir epoch'ta işlenen örnek sayısı olarak tanımlanmıştır. İkisi de 4'tür. Her epoch, dört örneğin tamamını kapsar ve bir batch içerir.

Toplam epoch sayısı ayrı bir kavramdır: `epoch` sayacı, hata sıfıra ulaşana kadar artar. Hocanın “epoch size” ile sabit toplam epoch sayısını kastetmesi durumunda bu yorumun teyit edilmesi gerekir. Sabit dört epoch'ta durmak ile hata sıfırlanana kadar devam etmek, her başlangıç değeri için aynı sonucu vermez.

Sıfırlanan değer `z` denkleminin çıktısı değildir; toplam sınıflandırma hatasıdır. Hocanın verdiği özel bir denklem veya veri seti varsa örnek ona göre uyarlanmalıdır.

## Doğrulanan sonuç

Verilen başlangıç değerleriyle kod 19. epoch'ta toplam hata 0'a ulaşmıştır:

```text
Epoch 19: toplam hata = 0
0, 0 -> 0 (hedef: 0)
0, 1 -> 0 (hedef: 0)
1, 0 -> 0 (hedef: 0)
1, 1 -> 1 (hedef: 1)
```

Başlangıç ağırlıkları değiştirilirse gereken epoch sayısı değişebilir. Bu örnek AND verisi içindir. Tek bir doğrusal karar sınırıyla ayrılamayan XOR gibi verilerde bu model sıfır hataya ulaşamaz; mevcut durma koşulu böyle bir veri için sonsuza kadar devam edebilir.

## Kısa sözlü anlatım

“Modelim iki girdiyi alıp ağırlıklarıyla çarpıyor ve bias ekliyor. Çıkan değerin işaretine göre 0 veya 1 tahmini yapıyor. Dört örneğin hatalarını biriktirip ağırlıkları birlikte güncelliyor. Bütün örnekler doğru sınıflandırılana kadar eğitim devam ediyor. Hiçbir kütüphane kullanmadım; başlangıç ağırlıklarını elle belirledim.”
