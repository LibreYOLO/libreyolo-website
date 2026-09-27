---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet dans LibreYOLO : prédire, entraîner et exporter'
description: >-
  Exécutez PicoDet dans LibreYOLO pour la détection d'objets sur mobile.
  Installez, prédisez, entraînez, validez et exportez sous Apache-2.0.
lead: >-
  PicoDet est un détecteur à une étape conçu pour les CPU mobiles et edge : un
  backbone ESNet, un neck CSP-PAN et une tête Generalized Focal Loss partagée.
  LibreYOLO le prend en charge pour la détection.
keywords:
  - PicoDet
  - PP-PicoDet
  - détection d'objets
  - détection d'objets mobile
  - détection edge
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # Il est utile de définir imgsz : la CLI utilise 640 par défaut, alors
        que

        # la résolution native du checkpoint s est 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibrePICODETs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Installer

PicoDet ne nécessite aucun extra en plus du paquet de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
détecteur ne demande donc de modifier qu'une ligne. `conf` définit le seuil de
confiance et `iou` le seuil NMS. Consultez la [prédiction](/docs/predict) pour
les sources, le streaming et le traitement des résultats.

## Variantes

Trois tailles sont proposées, chacune avec sa propre résolution d'entrée fixe :
`s` est la plus petite et `l` la plus grande. La résolution augmente avec la
taille. Les checkpoints plus grands sont donc aussi plus coûteux à exécuter par
image, en plus de contenir davantage de paramètres.

<benchmark-table task="detect" />

<va-embed />

## Entraîner

<code-tabs name="train" />

Les composants de la loss et l'assigneur suivent la recette amont : VFL, DFL,
GIoU et SimOTA, avec une pondération par qualité de classification et des cibles
VFL à IoU dynamique. L'inférence est équivalente bit à bit à la version amont
sur le même checkpoint.

Comme l'indique la propre docstring de `train()`, les éléments suivants n'ont pas
été vérifiés : la convergence sur un dataset complet, le comportement multi-GPU
et toute augmentation autre que le retournement horizontal. À sa résolution
native de 320, le checkpoint `s` n'a pas non plus dépassé de manière fiable le
seuil d'exactitude de LibreYOLO sur l'échantillon de 30 images et deux classes
avec lequel la bibliothèque teste les petits fine-tunings. Cette taille convient
mieux à l'échelle de COCO complet.

`train()` accepte également un argument `pretrained`, mais sa valeur n'est
jamais lue dans la méthode : l'entraînement reprend toujours à partir des poids
avec lesquels le modèle a été construit, et `pretrained=False` ne réinitialise
donc pas le réseau. Si vous ne définissez pas `imgsz` en Python, il prend la
résolution native du checkpoint chargé : 320 pour `s`, 416 pour `m` et 640 pour
`l`. La CLI envoie toujours un `imgsz`, qui vaut 640 par défaut ; définissez-le
donc dans ce cas afin qu'il corresponde au checkpoint.

Avec les autres réglages par défaut, l'entraîneur exécute 300 époques avec SGD à
`lr0=0.01`, une quantité de mouvement de 0.9, un weight decay de 4e-5 et un
warmup d'une époque selon un planning cosinus. Le retournement horizontal est la
seule augmentation appliquée.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset dans le format
utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`. L'exécution du graphe dans un runtime brut, sans
installation de LibreYOLO, est également prise en charge, mais vous devez alors
écrire vous-même le prétraitement et le post-traitement.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Le portage de LibreYOLO suit Bo396543018/Picodet_Pytorch, une réimplémentation
PyTorch du PP-PicoDet original de PaddleDetection. mmcv a été retiré et chaque
activation reproduite exactement, afin que les checkpoints PaddlePaddle
convertis par le pipeline de Bo se chargent sans dérive numérique. Les deux
sources sont soumises aux mêmes conditions Apache-2.0 que celles des auteurs de
l'article.

</provenance-box>

## Citation

<citation-block />
