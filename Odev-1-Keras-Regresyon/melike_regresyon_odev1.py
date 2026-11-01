"""Derin Öğrenme - Ödev 1 (Melike Zümra Özen, öğrenci no: 363426821).

Keras regresyon deneyi: altı parametre grubu sırayla karşılaştırılır. Her grupta
MSE'yi en fazla düşüren değer bir sonraki gruba taşınır.

Not: Hocanın dosyasında 'y = 1 + sin(pi*y/4)' yazılmıştır. Giriş değişkeni
p olarak tanımlandığı için deneyde y=1+sin(pi*p/4) varsayılmıştır.
Bu yorum hocaya teyit ettirilmelidir.
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras import layers

# Okul numarası 363426821 --> son iki hane 21. Arkadaşın 16 kullanmıştı.
SEED = 21
EPOCHS = 300
DATA_POINTS = 201  # [-2, 2] arasında eşit aralıklı örnekler
BATCH_SIZE = DATA_POINTS  # Tam-batch: her epoch'ta bir ağırlık güncellemesi
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sonuclar"
GRAFIK = ROOT / "grafikler"

ACTIVATIONS = [
    "Lineer", "Sigmoid", "Hiperbolik Tanjant", "ReLU",
    "LeakyReLU", "ELU", "SiLU"
]
TESTS = [
    ("aktivasyon", "Aktivasyon fonksiyonlarının değişimi", ACTIVATIONS),
    ("ogrenme_orani", "Öğrenme oranının değişimi", [0.1, 0.01, 0.001]),
    ("optimizer", "Öğrenme oranı ayar yönteminin değişimi", ["SGD", "RMSProp", "ADAM"]),
    ("dagilim", "Başlangıç dağılımının değişimi", ["normal", "uniform"]),
    ("katman", "Gizli katman sayısının değişimi", [1, 2, 5, 10, 20]),
    ("noron", "Gizli nöron sayısının değişimi", [2, 5, 10, 20, 50]),
]


@dataclass(frozen=True)
class ModelSettings:
    aktivasyon: str = "ReLU"
    ogrenme_orani: float = 0.1
    optimizer: str = "SGD"
    dagilim: str = "normal"
    katman: int = 1
    noron: int = 2


def fixed_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    if keras.backend.backend() == "torch":
        import torch
        torch.set_num_threads(2)


def get_data():
    x = np.linspace(-2, 2, DATA_POINTS, dtype=np.float32).reshape(-1, 1)
    target = (1 + np.sin(np.pi * x / 4)).astype(np.float32)
    return x, target


def new_initializer(name, seq):
    # Ayrı katmanların aynı başlangıç matrisini tekrar kullanmasını önler.
    # Tüm sıra okul numarasından üretilen sabit tohumla deterministiktir.
    seed = SEED + seq
    if name == "normal":
        return keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    return keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)


def make_activation(name):
    if name in ("Lineer", "Sigmoid", "Hiperbolik Tanjant", "ReLU", "SiLU"):
        return layers.Activation({
            "Lineer": "linear", "Sigmoid": "sigmoid", "Hiperbolik Tanjant": "tanh",
            "ReLU": "relu", "SiLU": "silu"
        }[name])
    if name == "LeakyReLU":
        return layers.LeakyReLU(negative_slope=0.3)
    if name == "ELU":
        return layers.ELU(alpha=1.0)
    raise ValueError(f"Bilinmeyen aktivasyon: {name}")


def create_network(settings):
    fixed_seed()
    network = keras.Sequential(name="melike_regresyon")
    network.add(layers.Input(shape=(1,)))
    for number in range(settings.katman):
        network.add(layers.Dense(
            settings.noron, kernel_initializer=new_initializer(settings.dagilim, number)))
        network.add(make_activation(settings.aktivasyon))
    network.add(layers.Dense(
        1, activation="linear",
        kernel_initializer=new_initializer(settings.dagilim, settings.katman)))
    optimizers = {"SGD": keras.optimizers.SGD,
                  "RMSProp": keras.optimizers.RMSprop,
                  "ADAM": keras.optimizers.Adam}
    network.compile(
        optimizer=optimizers[settings.optimizer](learning_rate=settings.ogrenme_orani),
        loss="mse")
    return network


class SaveHistory(keras.callbacks.Callback):
    """Her epoch sonunda eğitim noktalarındaki gerçek tahmin hatasını ölçer."""
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y.ravel()
        self.sst = float(np.sum((self.y - np.mean(self.y)) ** 2))
        self.mse = []
        self.r2 = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = np.asarray(keras.ops.convert_to_numpy(self.model(self.x, training=False))).reshape(-1)
        mse = float(np.mean((self.y - predictions) ** 2))
        r2 = float(1 - np.sum((self.y - predictions) ** 2) / self.sst)
        self.mse.append(mse if np.isfinite(mse) else None)
        self.r2.append(r2 if np.isfinite(r2) else None)


def train_one(settings, x, y):
    keras.backend.clear_session()
    network = create_network(settings)
    track = SaveHistory(x, y)
    started = time.perf_counter()
    network.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                shuffle=False, callbacks=[track], verbose=0)
    elapsed = time.perf_counter() - started
    prediction = np.asarray(keras.ops.convert_to_numpy(network(x, training=False))).reshape(-1)
    error = y.ravel() - prediction
    mse = float(np.mean(error ** 2))
    r2 = float(1 - np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))
    if not (np.isfinite(mse) and np.isfinite(r2)):
        mse, r2 = None, None
    return {
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)) if mse is not None else None,
        "R2": r2,
        "Sure_sn": round(elapsed, 3),
        "epoch_MSE": track.mse,
        "epoch_R2": track.r2,
        "tahmin": prediction.tolist(),
    }


def visualize_best(group_num, title, selected, x, y, result):
    epochs = np.arange(1, EPOCHS + 1)
    color = "#2c5480"
    accent = "#cb663f"
    fig, axs = plt.subplots(1, 3, figsize=(14.3, 3.9))
    axs[0].semilogy(epochs, np.maximum(
        np.asarray([v if v is not None else np.nan for v in result["epoch_MSE"]]),
        1e-12), color=color, lw=1.7)
    axs[0].set(xlabel="Epoch", ylabel="MSE (log)", title="Epoch / MSE")
    axs[1].plot(epochs, result["epoch_R2"], color=color, lw=1.7)
    axs[1].set(xlabel="Epoch", ylabel="R²", title="Epoch / R²")
    axs[2].plot(x.ravel(), y.ravel(), color=color, label="Hedef", lw=1.7)
    axs[2].plot(x.ravel(), result["tahmin"], color=accent,
                linestyle="--", label="Çıkış", lw=1.7)
    axs[2].set(xlabel="Giriş (p)", ylabel="y", title="Hedef / Çıkış")
    axs[2].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.20)
    fig.suptitle(f"Tablo {group_num}: {title} | Seçilen: {selected}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .92])
    image_path = GRAFIK / f"tablo_{group_num}_en_iyi.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    # Aynı üç grafiğin Word içinde daha büyük okunabilen 2-satırlı yerleşimi.
    # Kaynak görüntüdeki üç paneli korur, verileri değiştirmez.
    from PIL import Image
    with Image.open(image_path) as source:
        im = source.convert("RGB")
        total_width, height = im.size
        panel_width = total_width // 3
        top, bottom = 75, height - 4
        panels = [im.crop((panel_width * i, top, panel_width * (i + 1), bottom))
                  for i in range(3)]
        width, panel_height = panel_width * 2, bottom - top
        lower_width = round(panel_width * 1.35)
        lower_height = round(panel_height * 1.35)
        combined = Image.new("RGB", (width, panel_height + lower_height + 32), "white")
        combined.paste(panels[0], (0, 0))
        combined.paste(panels[1], (panel_width, 0))
        lower_panel = panels[2].resize((lower_width, lower_height), Image.Resampling.LANCZOS)
        combined.paste(lower_panel, ((width - lower_width) // 2, panel_height + 25))
        combined.save(GRAFIK / f"tablo_{group_num}_okunakli.png", optimize=True)
    return image_path.name


def write_csv(rows):
    columns = ["Tablo", "Parametre", "Deger", "MSE", "RMSE", "R2",
               "Sure_sn", "En_iyi", "Seed", "Epoch", "Backend"]
    with (OUT / "melike_odev1_sonuclar.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows([{k: row[k] for k in columns} for row in rows])


def run_experiments():
    OUT.mkdir(exist_ok=True, parents=True)
    GRAFIK.mkdir(exist_ok=True, parents=True)
    x, y = get_data()
    current = ModelSettings()
    all_rows = []
    groups = []
    for group_num, (field, title, alternatives) in enumerate(TESTS, 1):
        initial = asdict(current)
        candidates = []
        for value in alternatives:
            cfg = replace(current, **{field: value})
            result = train_one(cfg, x, y)
            candidates.append((value, result, cfg))
            print(f"Tablo {group_num} | {field}={value} | "
                  f"MSE={result['MSE']} | R²={result['R2']} | "
                  f"Süre={result['Sure_sn']}sn", flush=True)
        valid = [item for item in candidates if item[1]["MSE"] is not None]
        if not valid:
            raise RuntimeError(f"Tablo {group_num}: Hiçbir model geçerli sonuç üretmedi")
        selected, winner, final_cfg = min(valid, key=lambda i: i[1]["MSE"])
        plot_name = visualize_best(group_num, title, selected, x, y, winner)
        current = final_cfg
        groups.append({
            "numara": group_num,
            "baslik": title,
            "degisken": field,
            "sabit_ayarlar": initial,
            "secili_deger": selected,
            "secili_ayarlar": asdict(current),
            "grafik": plot_name,
            "satirlar": [{"deger": v, "MSE": r["MSE"], "RMSE": r["RMSE"],
                         "R2": r["R2"], "Sure_sn": r["Sure_sn"]}
                        for v, r, cfg in candidates]
        })
        for value, result, config in candidates:
            all_rows.append({
                "Tablo": group_num, "Parametre": title, "Deger": value,
                "MSE": result["MSE"], "RMSE": result["RMSE"],
                "R2": result["R2"], "Sure_sn": result["Sure_sn"],
                "En_iyi": value == selected, "Seed": SEED, "Epoch": EPOCHS,
                "Backend": keras.backend.backend()
            })
        write_csv(all_rows)
        (OUT / "melike_odev1_ozet.json").write_text(
            json.dumps({"ogrenci":"Melike Zümra Özen", "okul_no":"363426821",
                        "seed":SEED,"epoch":EPOCHS,"num_points":DATA_POINTS,
                        "batch_size":BATCH_SIZE,"backend":keras.backend.backend(),
                        "hedef":"1 + sin(pi * p / 4) (belgedeki değişken doğrulanmalı)",
                        "gruplar":groups}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    print("\nTamamlandı. Son ayarlar:", asdict(current))
    print("Çıktı klasörü:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ödev 1: Keras regresyon karşılaştırması")
    parser.add_argument("--calistir", action="store_true", help="25 deneyin tamamını çalıştır")
    args = parser.parse_args()
    run_experiments()