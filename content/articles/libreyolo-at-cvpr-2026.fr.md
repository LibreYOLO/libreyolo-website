---
title: "LibreYOLO à CVPR 2026 : détection d'objets sur appareils edge"
description: Lors de CVPR 2026 à Denver, une équipe de Jabra et de l'IT University of Copenhagen a utilisé LibreYOLOXs comme modèle d'exemple dans son tutoriel « Edge AI in Action », en le faisant tourner sur Hailo-8L et Snapdragon.
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "À quelle vitesse LibreYOLOXs fonctionne-t-il sur du matériel edge ?"
    a: "Dans le tutoriel de CVPR 2026, LibreYOLOXs a fonctionné à 19.0 ms par frame (52.6 FPS) sur un Raspberry Pi 5 avec un Hailo-8L, et à 6.69 ms sur le HTP/DSP d'un Qualcomm QCS6490 après quantification INT8."
  - q: "Comment déployer un modèle LibreYOLO sur un accélérateur edge ?"
    a: "Exportez le modèle en ONNX avec la fonction d'export de LibreYOLO, puis compilez-le pour la cible : le Hailo Dataflow Compiler produit un HEF pour les puces Hailo, et la pile SNPE / QAIRT / AI Hub couvre Qualcomm Snapdragon. C'est exactement le pipeline présenté de bout en bout dans le tutoriel de CVPR."
---

![CVPR 2026, Denver, Colorado, du 3 au 7 juin](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO était présent à CVPR 2026.

CVPR est largement reconnu comme la conférence de vision par ordinateur la plus prestigieuse au monde. Cette année, elle se tenait à Denver, où une équipe de Jabra et de l'IT University of Copenhagen a animé un tutoriel intitulé « Edge AI in Action: Mastering On-Device Inference ». Elle a choisi LibreYOLOXs comme modèle d'exemple pour déployer la détection d'objets sur des puces edge.

## Ce qu'ils ont construit

Le tutoriel a présenté le pipeline edge complet, de bout en bout. Il commence avec un modèle LibreYOLOXs. Exportez-le en ONNX. Compilez ensuite cet ONNX pour deux matériels très différents : un accélérateur Hailo-8L et Qualcomm Snapdragon.

## Temps réel sur Hailo-8L

Ils ont compilé l'ONNX en HEF avec le Hailo Dataflow Compiler, puis l'ont exécuté sur un Raspberry Pi 5 équipé du Hailo-8L AI HAT.

![LibreYOLOXs exécuté sur un Raspberry Pi 5 avec un Hailo-8L, détectant des personnes et des sacs à main dans une rue animée](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Il a fonctionné à 19.0 ms par frame, soit 52.6 FPS, sur un Raspberry Pi.

## Temps réel sur Snapdragon

Côté Qualcomm, ils ont quantifié LibreYOLOXs en INT8 et l'ont exécuté avec la pile SNPE, QAIRT et AI Hub. Le benchmark a été réalisé sur un QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs détectant en direct un téléviseur, un ordinateur portable, un clavier, une souris et un téléphone portable dans l'application EdgeVision AI sur un téléphone Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Le résultat était de 6.69 ms sur le HTP/DSP.

## Remerciements

Un grand merci à Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera et Shan Ahmed Shaffi d'avoir présenté le projet.

- Tutoriel et diapositives : [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Modèles LibreYOLOXs quantifiés pour Qualcomm : [sur Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Essayer

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO est sous licence MIT, fonctionne sous Linux, Mac et Windows, et prend en charge les GPU, Apple Silicon et les CPU classiques sans changement de code. Une seule API couvre YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, la segmentation, l'estimation de pose, la profondeur et bien plus encore.

Ajoutez une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
