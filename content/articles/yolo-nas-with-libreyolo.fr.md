---
title: "YOLO-NAS est-il encore maintenu ? Utiliser et entraîner YOLO-NAS avec LibreYOLO"
description: "Exécutez, entraînez, validez et exportez YOLO-NAS avec LibreYOLO. SuperGradients a publié sa dernière version en avril 2024. Nous prévoyons d'entraîner de nouveaux poids à partir de zéro pour nous affranchir de la licence de Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS est-il abandonné ?"
    a: "SuperGradients a publié la version 3.7.1 en avril 2024. LibreYOLO prend en charge la prédiction, l'entraînement, la validation et l'export de YOLO-NAS avec les versions actuelles de PyTorch."
  - q: "Ultralytics maintient-il YOLO-NAS ?"
    a: "Ultralytics prend en charge l'inférence, la validation et l'export. L'entraînement n'est pas pris en charge."
  - q: "Peut-on utiliser les poids pré-entraînés de YOLO-NAS à des fins commerciales ?"
    a: "Les poids pré-entraînés de Deci restent soumis aux conditions non commerciales de leur éditeur, quelle que soit la bibliothèque qui les charge. Un entraînement à partir d'une initialisation aléatoire évite ces checkpoints. Nous prévoyons également de publier des poids COCO entraînés à partir de zéro."
---

YOLO-NAS reste un détecteur utile. Son projet d'origine, [SuperGradients](https://github.com/Deci-AI/super-gradients), a publié sa dernière version, la 3.7.1, en avril 2024. LibreYOLO assure la maintenance de la prédiction, de l'entraînement, de la validation et de l'export pour ce modèle.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO prend en charge les tailles S, M et L pour la détection, ainsi que l'estimation de pose. Les checkpoints sont téléchargés depuis le CDN public de Deci sous des noms tels que `LibreYOLONASs.pt`. LibreYOLO n'en héberge aucun, et les conditions non commerciales de Deci restent applicables. SuperGradients impose une version de PyTorch comprise entre 1.9 et 1.13 ; LibreYOLO requiert la version 2.4 ou ultérieure. L'export pour la détection prend en charge ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch et Core AI. CoreML n'est pas pris en charge.

Pour entraîner le modèle sans checkpoint de Deci :

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Nous prévoyons également d'entraîner YOLO-NAS à partir de zéro sur COCO et de publier ces poids.

Installez le package avec `pip install libreyolo`. La [documentation de YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) présente les détails du modèle. Consultez aussi notre article plus long sur les [alternatives à SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
