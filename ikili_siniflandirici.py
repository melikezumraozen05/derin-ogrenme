# AND: sadece iki girdi de 1 ise hedef 1'dir.
veriler = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
w1, w2, b = 0.3, -0.2, 0.1  # Elle secilen baslangic agirliklari ve bias
ogrenme_orani = 0.1
batch_size = epoch_size = len(veriler)  # epoch_size: bir turdaki ornek sayisi
epoch, hata = 0, 1

while hata != 0:
    dw1 = dw2 = db = 0
    for x1, x2, hedef in veriler[:batch_size]:
        tahmin = int(w1 * x1 + w2 * x2 + b >= 0)
        fark = hedef - tahmin
        dw1 += fark * x1
        dw2 += fark * x2
        db += fark
    w1 += ogrenme_orani * dw1 / batch_size
    w2 += ogrenme_orani * dw2 / batch_size
    b += ogrenme_orani * db / batch_size
    epoch += 1
    hata = sum(abs(y - int(w1 * x1 + w2 * x2 + b >= 0)) for x1, x2, y in veriler)
    print(f"Epoch {epoch}: toplam hata = {hata}")

for x1, x2, hedef in veriler:
    print(f"{x1}, {x2} -> {int(w1 * x1 + w2 * x2 + b >= 0)} (hedef: {hedef})")
