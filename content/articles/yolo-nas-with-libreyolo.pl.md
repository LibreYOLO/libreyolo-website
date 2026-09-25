---
title: "YOLO-NAS nadal wspierany: trenowanie i eksport w LibreYOLO"
description: "Uruchamiaj, trenuj, waliduj i eksportuj YOLO-NAS w LibreYOLO. Ostatnie wydanie SuperGradients ukazało się w kwietniu 2024 roku. Planujemy wytrenować nowe wagi od zera, aby nie podlegały warunkom licencji Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "Czy YOLO-NAS jest porzucony?"
    a: "Ostatnie wydanie SuperGradients, 3.7.1, ukazało się w kwietniu 2024 roku. LibreYOLO obsługuje predykcję, trenowanie, walidację i eksport YOLO-NAS w aktualnej wersji PyTorch."
  - q: "Czy Ultralytics utrzymuje YOLO-NAS?"
    a: "Ultralytics obsługuje inferencję, walidację i eksport. Nie obsługuje trenowania."
  - q: "Czy można komercyjnie używać wstępnie wytrenowanych wag YOLO-NAS?"
    a: "Wstępnie wytrenowane wagi Deci zachowują pierwotne warunki niekomercyjnego użytku niezależnie od biblioteki, która je wczytuje. Trenowanie od losowej inicjalizacji pozwala uniknąć tych checkpointów. Planujemy też opublikować wagi COCO wytrenowane od zera."
---

YOLO-NAS nadal jest użytecznym detektorem. Jego pierwotne środowisko, [SuperGradients](https://github.com/Deci-AI/super-gradients), wydało ostatnią wersję, 3.7.1, w kwietniu 2024 roku. LibreYOLO nadal obsługuje predykcję, trenowanie, walidację i eksport tego modelu.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO obsługuje warianty S, M i L do detekcji, a także estymację pozy. Checkpointy są pobierane z publicznego CDN Deci pod nazwami takimi jak `LibreYOLONASs.pt`. LibreYOLO nie hostuje żadnego z nich, a niekomercyjne warunki Deci nadal obowiązują. SuperGradients wymaga wersji PyTorch od 1.9 do 1.13, natomiast LibreYOLO wymaga wersji 2.4 lub nowszej. Eksport detekcji obsługuje ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch i Core AI. CoreML nie jest obsługiwany.

Aby trenować bez checkpointu Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Planujemy też wytrenować YOLO-NAS od zera na COCO i opublikować te wagi.

Zainstaluj bibliotekę poleceniem `pip install libreyolo`. [Dokumentacja YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) zawiera szczegóły modelu. Dostępny jest też dłuższy artykuł o [alternatywie dla SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja](https://www.libreyolo.com/docs)
