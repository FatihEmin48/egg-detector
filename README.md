# Yumurta Tespiti ve Sayımı

Holluktaki yumurtaları tarayıcıda tespit edip sayan bir araç. Model, [YumurtaTespiti](https://github.com/FatihEmin48/YumurtaTespiti) mobil uygulaması için eğitilmiş gerçek YOLO11m-seg modelinin ONNX'e aktarılmış hâlidir; sunucu gerekmez, tüm çıkarım istemci tarafında çalışır.

🔗 **Canlı:** https://fatihemin48.github.io/egg-detector/

## Nasıl çalışır

Orijinal model bir YOLO11 **segmentasyon** modelidir. Bu demo, modelin kutu çıktısını kullanır; maske katsayıları (segmentasyon) şimdilik kullanılmaz — amaç basit ve hızlı bir tespit/sayım deneyimi sunmak.

Model 640×640 girişle çalışır ve önceki nesne tespiti demosundan (YOLOv8n, 320×320) daha büyüktür; ilk yüklemede indirme biraz daha uzun sürer, kamera modu daha yavaş çalışabilir.

## Dosyalar

- `index.html` / `style.css` — arayüz
- `app.js` — ön işleme, model çalıştırma, kutu çizimi ve sayım mantığı
- `model/egg_yolo.onnx` — dışa aktarılmış model (640×640, opset 12)

## Yerelde çalıştırma

```bash
python3 -m http.server 8000
```

ardından tarayıcıdan `http://localhost:8000` adresine gidilir.
