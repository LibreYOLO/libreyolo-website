---
title: "LibreYOLO al CVPR 2026: object detection edge su Hailo e Snapdragon"
description: "Al CVPR 2026 di Denver, un team di Jabra e IT University of Copenhagen ha usato LibreYOLOXs come modello di riferimento nel tutorial \"Edge AI in Action\", eseguendolo su Hailo-8L e Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Quanto è veloce LibreYOLOXs su hardware edge?"
    a: "Nel tutorial CVPR 2026, LibreYOLOXs ha raggiunto 19.0 ms per frame (52.6 FPS) su un Raspberry Pi 5 con Hailo-8L e 6.69 ms sull'HTP/DSP di un Qualcomm QCS6490 dopo la quantizzazione INT8."
  - q: "Come si distribuisce un modello LibreYOLO su un acceleratore edge?"
    a: "Esporta in ONNX con la funzione di esportazione di LibreYOLO, poi compila per il dispositivo di destinazione: Hailo Dataflow Compiler genera un HEF per i chip Hailo, mentre lo stack SNPE / QAIRT / AI Hub copre Qualcomm Snapdragon. È proprio la pipeline che il tutorial CVPR ha mostrato dall'inizio alla fine."
---

![CVPR 2026, Denver, Colorado, 3-7 giugno](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO è arrivato al CVPR 2026.

Il CVPR è ampiamente riconosciuto come la conferenza di computer vision più prestigiosa al mondo. Quest'anno si è tenuto a Denver e un team di Jabra e IT University of Copenhagen ha condotto un tutorial dal titolo "Edge AI in Action: Mastering On-Device Inference". Per mostrare come portare il rilevamento di oggetti sui chip edge, ha scelto LibreYOLOXs come modello di riferimento.

## Cosa hanno realizzato

Il tutorial ha illustrato l'intera pipeline edge, dall'inizio alla fine. Si parte da un modello LibreYOLOXs. Lo si esporta in ONNX. Poi si compila quel modello ONNX per due hardware molto diversi: un acceleratore Hailo-8L e Qualcomm Snapdragon.

## Tempo reale su Hailo-8L

Hanno compilato il modello ONNX in un HEF con Hailo Dataflow Compiler, poi lo hanno eseguito su un Raspberry Pi 5 con Hailo-8L AI HAT.

![LibreYOLOXs in esecuzione su un Raspberry Pi 5 con Hailo-8L, mentre rileva persone e borse su una strada trafficata](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Ha raggiunto 19.0 ms per frame, 52.6 FPS, su un Raspberry Pi.

## Tempo reale su Snapdragon

Per Qualcomm hanno quantizzato LibreYOLOXs in INT8 e lo hanno eseguito con lo stack SNPE, QAIRT e AI Hub, con benchmark su un QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs rileva in tempo reale una TV, un laptop, una tastiera, un mouse e un cellulare nell'app EdgeVision AI su un telefono Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Ha impiegato 6.69 ms sull'HTP/DSP.

## Ringraziamenti

Un grande ringraziamento a Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera e Shan Ahmed Shaffi per aver presentato il progetto.

- Tutorial e slide: [Edge AI in azione: padroneggiare l'inferenza on-device](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Modelli LibreYOLOXs quantizzati per Qualcomm: [su Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO è distribuito con licenza MIT, funziona su Linux, Mac e Windows e gira su GPU, Apple Silicon e CPU standard senza modifiche al codice. Una sola API copre YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentazione, stima della posa, profondità e altro.

Aggiungi una stella su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
