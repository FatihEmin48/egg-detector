# Yumurta Tespiti ve Sayımı

Holluktaki yumurtalar YOLO ile tespit edilip sayılıyor. Tamamen tarayıcıda, istemci tarafında çalışır — sunucu gerekmez.

🔗 **Canlı:** https://fatihemin48.github.io/egg-detector/

## Nasıl çalışır

Bir fotoğraf yükleyebilir ya da kamerayı açıp "Fotoğraf Çek" butonuyla o anki kareyi yakalayabilirsiniz; yumurtalar tespit edilip sayılır. "Sıfırla" butonuyla kameraya dönüp yeni bir fotoğraf çekebilirsiniz.

Model, kutu (box) çıktısını kullanır; segmentasyon maskeleri şimdilik kullanılmıyor — amaç basit ve hızlı bir tespit/sayım deneyimi sunmak.

## Dosyalar

- `index.html` / `style.css` — arayüz
- `app.js` — ön işleme, model çalıştırma, kutu çizimi ve sayım mantığı
- `model/egg_yolo.onnx` — dışa aktarılmış model (640×640, opset 12)

## Yerelde çalıştırma

```bash
python3 -m http.server 8000
```

ardından tarayıcıdan `http://localhost:8000` adresine gidilir.
