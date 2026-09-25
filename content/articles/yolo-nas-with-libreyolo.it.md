---
title: "YOLO-NAS è ancora mantenuto? Usalo con LibreYOLO"
description: "Esegui, addestra, valida ed esporta YOLO-NAS con LibreYOLO. L'ultima release di SuperGradients risale ad aprile 2024. Puntiamo ad addestrare nuovi pesi da zero per non dipendere dalla licenza di Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS è abbandonato?"
    a: "L'ultima release di SuperGradients, la 3.7.1, risale ad aprile 2024. LibreYOLO supporta la predizione, l'addestramento, la validazione e l'esportazione di YOLO-NAS con le versioni attuali di PyTorch."
  - q: "Ultralytics mantiene YOLO-NAS?"
    a: "Ultralytics supporta l'inferenza, la validazione e l'esportazione. Non supporta l'addestramento."
  - q: "È possibile usare commercialmente i pesi preaddestrati di YOLO-NAS?"
    a: "I pesi preaddestrati di Deci restano soggetti alle condizioni non commerciali della licenza upstream, qualunque libreria li carichi. Addestrare da un'inizializzazione casuale evita quei checkpoint. Puntiamo anche a pubblicare pesi COCO addestrati da zero."
---

YOLO-NAS è ancora un rilevatore utile. La sua piattaforma originale, [SuperGradients](https://github.com/Deci-AI/super-gradients), ha rilasciato la versione 3.7.1 ad aprile 2024. LibreYOLO mantiene il supporto per la predizione, l'addestramento, la validazione e l'esportazione.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supporta le taglie S, M e L per il rilevamento, oltre alla posa. I checkpoint si scaricano dal CDN pubblico di Deci con nomi come `LibreYOLONASs.pt`. LibreYOLO non ne ospita nessuno e continuano ad applicarsi le condizioni non commerciali di Deci. SuperGradients vincola PyTorch a versioni dalla 1.9 alla 1.13; LibreYOLO richiede la 2.4 o successive. L'esportazione per il rilevamento supporta ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch e Core AI. CoreML non è supportato.

Per addestrare senza un checkpoint di Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Puntiamo anche ad addestrare YOLO-NAS da zero su COCO e a pubblicare quei pesi.

Installa con `pip install libreyolo`. La [documentazione di YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) descrive il modello. Nell'articolo [Un'alternativa a SuperGradients](/articles/supergradients-alternative) trovi un approfondimento.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
