---
title: "LiteRT vs TensorFlow Lite : simple changement de nom, pas de nouveau format"
description: "Google a renommé TensorFlow Lite en LiteRT en septembre 2024. Même runtime, mêmes fichiers .tflite, rien ne casse. Voici pourquoi ce changement a eu lieu, ce qui a réellement changé et comment exporter un modèle LibreYOLO vers ce format."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite est-il obsolète ?"
    a: "Le nom est progressivement abandonné, pas la technologie. Le runtime continue sous le nom LiteRT avec les mêmes API, et les modèles TFLite et le code existants continuent de fonctionner. Les nouveaux développements, comme l'API CompiledModel et l'accélération NPU, sont publiés sous le nom LiteRT."
  - q: "Dois-je réexporter mes modèles pour LiteRT ?"
    a: "Non. Un fichier .tflite exporté pour TensorFlow Lite est un modèle LiteRT. Le fichier n'a pas changé, il n'y a donc rien à réexporter ni à migrer."
---

**LiteRT est TensorFlow Lite sous un nouveau nom. Ce n'est ni un nouveau runtime ni un nouveau format de fichier. Les modèles sont toujours des fichiers `.tflite`, et tous les fichiers `.tflite` qui fonctionnaient avec TensorFlow Lite fonctionnent avec LiteRT sans modification.**

Si vous avez recherché « LiteRT vs TensorFlow Lite », « TensorFlow Lite est-il obsolète » ou « qu'est-ce que LiteRT », le paragraphe précédent répond à votre question. Voici pourquoi le nom a changé et comment exporter un modèle vers ce format avec LibreYOLO.

## Pourquoi Google l'a renommé

TensorFlow Lite a été lancé en 2017 pour exécuter des modèles TensorFlow sur des téléphones et des cartes embarquées. Au fil des ans, son usage s'est largement étendu : Google a créé des voies de conversion depuis PyTorch, JAX et Keras. En 2024, une grande partie des modèles exécutés avec cet outil n'avait donc jamais utilisé TensorFlow.

À ce stade, le nom était franchement trompeur. Les utilisateurs de PyTorch le délaissaient parce qu'il semblait réservé à TensorFlow. En septembre 2024, Google l'a donc [renommé LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), abréviation de « Lite Runtime ». Voilà toute l'histoire. Même équipe, même code, nom indépendant du framework.

## Ce qui a réellement changé

Très peu de choses, et aucune ne casse quoi que ce soit :

* Le code a déménagé vers un nouvel emplacement : [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* La documentation se trouve désormais à l'adresse [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Le package Python pour exécuter les modèles s'appelle désormais `ai-edge-litert`. L'ancien package `tflite-runtime` n'est plus à jour ; le nouveau fournit la même classe `Interpreter`.
* Depuis le changement de nom, les nouvelles fonctionnalités sont publiées sous le nom LiteRT : une API `CompiledModel` plus simple et l'accélération GPU/NPU que Google a déclarée [prêtes pour la production en janvier 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

L'API Interpreter classique continue aussi de fonctionner comme avant. Une seule extension est réellement nouvelle : `.litertlm`, un format de packaging réservé aux LLM exécutés sur l'appareil. Pour les modèles de vision par ordinateur, il n'existe pas, et cela ne vous concerne pas. Ainsi, quand un document parle de « TFLite » et un autre de « LiteRT », ils parlent du même fichier.

## Exporter un modèle LibreYOLO vers LiteRT

LibreYOLO a ajouté une voie d'export TFLite/LiteRT dans la version v1.3.0. Elle nécessite Python 3.12+ et une dépendance supplémentaire :

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Ou depuis la CLI :

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Les voies validées à ce jour sont la détection avec YOLO9, ainsi que la détection, la segmentation et l'estimation de pose avec RF-DETR. Elles sont toutes encore marquées comme expérimentales. La matrice complète des fonctionnalités prises en charge se trouve dans la [documentation](/docs/v1.3.0).

Pour exécuter le fichier exporté, utilisez le package Google `ai-edge-litert` sur l'appareil cible :

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Notez que l'entrée est au format NHWC (canaux en dernier) : la conversion depuis ONNX transpose la disposition, ce qui est normal pour ce format.

## Ce que nous avons changé dans LibreYOLO

Pas grand-chose, car peu de changements étaient nécessaires. La documentation indique désormais « TFLite (LiteRT) » pour que les deux noms puissent être recherchés, et l'API conserve `format="tflite"` comme nom de format.

## FAQ

**TensorFlow Lite est-il obsolète ?**
Le nom est progressivement abandonné, pas la technologie. Le runtime continue sous le nom LiteRT avec les mêmes API, et les modèles TFLite et le code existants continuent de fonctionner. Les nouveaux développements, comme l'API CompiledModel et l'accélération NPU, sont publiés sous le nom LiteRT.

**Dois-je réexporter mes modèles pour LiteRT ?**
Non. Un fichier `.tflite` exporté pour TensorFlow Lite est un modèle LiteRT. Le fichier n'a pas changé, il n'y a donc rien à réexporter ni à migrer.
