---
title: "LibreYOLO na CVPR 2026: inferencja edge na Hailo-8L i Snapdragon"
description: "Podczas CVPR 2026 w Denver zespół Jabra i IT University of Copenhagen wykorzystał LibreYOLOXs jako model przykładowy w samouczku „Edge AI in Action”, uruchamiając go na Hailo-8L i Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Jak szybko działa LibreYOLOXs na sprzęcie brzegowym?"
    a: "W samouczku CVPR 2026 LibreYOLOXs działał z szybkością 19.0 ms na klatkę (52.6 FPS) na Raspberry Pi 5 z Hailo-8L oraz 6.69 ms na HTP/DSP układu Qualcomm QCS6490 po kwantyzacji INT8."
  - q: "Jak wdrożyć model LibreYOLO na akceleratorze brzegowym?"
    a: "Wyeksportuj model do ONNX za pomocą funkcji eksportu LibreYOLO, a następnie skompiluj go dla urządzenia docelowego: Hailo Dataflow Compiler tworzy plik HEF dla układów Hailo, a stos SNPE / QAIRT / AI Hub obsługuje Qualcomm Snapdragon. To dokładnie pełny pipeline pokazany w samouczku CVPR."
---

![CVPR 2026, Denver, Kolorado, 3-7 czerwca](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO pojawiło się na CVPR 2026.

CVPR jest powszechnie uznawana za najbardziej prestiżową konferencję poświęconą wizji komputerowej na świecie. W tym roku odbyła się w Denver, a zespół Jabra i IT University of Copenhagen poprowadził tam samouczek zatytułowany „Edge AI in Action: Mastering On-Device Inference”. Jako model przykładowy do przenoszenia detekcji obiektów na układy brzegowe wybrano LibreYOLOXs.

## Co zbudowano

Samouczek pokazał pełny pipeline brzegowy, od początku do końca. Najpierw model LibreYOLOXs, następnie eksport do ONNX, a potem kompilacja tego pliku ONNX dla dwóch zupełnie różnych platform sprzętowych: akceleratora Hailo-8L i Qualcomm Snapdragon.

## Czas rzeczywisty na Hailo-8L

Plik ONNX skompilowano do HEF za pomocą Hailo Dataflow Compiler, a następnie uruchomiono na Raspberry Pi 5 z Hailo-8L AI HAT.

![LibreYOLOXs działa na Raspberry Pi 5 z Hailo-8L, wykrywając ludzi i torebki na ruchliwej ulicy](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Model działał z szybkością 19.0 ms na klatkę, czyli 52.6 FPS, na Raspberry Pi.

## Czas rzeczywisty na Snapdragon

Po stronie Qualcomm model LibreYOLOXs poddano kwantyzacji INT8 i uruchomiono z użyciem stosu SNPE, QAIRT i AI Hub. Benchmark wykonano na QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs na żywo wykrywa telewizor, laptop, klawiaturę, mysz i telefon w aplikacji EdgeVision AI na telefonie Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Uzyskano wynik 6.69 ms na HTP/DSP.

## Podziękowania

Serdeczne podziękowania dla Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera i Shan Ahmed Shaffi za zaprezentowanie projektu.

- Samouczek i slajdy: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Modele LibreYOLOXs po kwantyzacji dla Qualcomm: [na Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO jest objęte licencją MIT, działa w systemach Linux, Mac i Windows oraz obsługuje GPU, Apple Silicon i zwykły CPU bez zmian w kodzie. Jedno API obejmuje YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentację, estymację pozy, estymację głębi i inne zadania.

Dodaj gwiazdkę na GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
