---
title: YOLOv7
families:
  - yolo7
seo_title: "YOLOv7 dans LibreYOLO\_: prédire, entraîner et exporter sous licence MIT"
description: "Exécutez YOLOv7 dans LibreYOLO pour la détection d'objets\_: installez, prédisez, entraînez, validez et exportez avec du code et des poids sous licence MIT."
lead: >-
  YOLOv7 est un détecteur mono-étape avec ancres dont la tête ajoute des
  décalages appris de connaissances implicites avant la convolution finale.
  LibreYOLO prend en charge son unique taille publiée pour la détection.
keywords:
  - YOLOv7
  - détection d'objets
  - détecteur avec ancres
  - connaissances implicites
  - ImplicitA
  - détection d'objets temps réel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Démarrage à chaud depuis un nouveau modèle
      language: python
      code: >
        from libreyolo import LibreYOLO7


        # pretrained=True charge toujours le checkpoint LibreYOLO7b.pt publié,

        # quel que soit l'objet avec lequel cette instance a été construite.
        Construire

        # la classe directement plutôt que via LibreYOLO() démarre sans aucun

        # poids chargé.

        model = LibreYOLO7(None, size="b")

        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLO7b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Installer

YOLOv7 ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. `conf` définit le seuil de
confiance et `iou` le seuil de NMS appliqué après le décodage de la tête avec
ancres. Consultez la [prédiction](/docs/predict) pour les sources, le streaming
et la gestion des résultats.

## Variantes

LibreYOLO fournit une seule taille, `b`. L'upstream publie un unique modèle
YOLOv7, vous n'avez donc aucune taille à choisir.

## Entraîner

<code-tabs name="train" />

`pretrained` est pris en compte, contrairement au paramètre homonyme sans
effet de certaines autres familles présentées ici\u00a0: transmettez `True` pour
démarrer à chaud depuis le checkpoint `LibreYOLO7b.pt` publié (téléchargé
automatiquement), ou un chemin ou un nom pour tout autre fichier. Ce
checkpoint publié utilise les 80 classes COCO. Si vous le demandez sur un
modèle déjà reconstruit pour un nombre de classes différent, le modèle
repasse d'abord à 80 classes, charge le checkpoint, puis transfère tous les
tenseurs dont la forme correspond vers la tête cible une fois le nombre de
classes du dataset lu. `resume=True` ne peut pas être combiné à `pretrained`.
Avec la valeur par défaut `None`, l'entraînement continue depuis l'état ayant
servi à construire le modèle, ou depuis une initialisation aléatoire si aucun
poids n'a été chargé.

Sans autre réglage, le trainer exécute 300 époques avec `lr0=0.01`, un momentum
SGD de 0.937, un warmup de 3 époques, ainsi que la même affectation SimOTA et
la même phase finale sans augmentation de 15 époques que YOLOX, adaptées à la
tête avec ancres. Une différence subsiste\u00a0: YOLOX ajoute durant ces dernières
époques un raffinement L1 de la régression des bounding boxes, que v7 omet, car
sa loss SimOTA ne possède aucune branche L1 de décalage brut à affiner.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. Vous pouvez également exécuter le graphe dans
un runtime nu, sans installer LibreYOLO, mais vous devez alors écrire vous-même
le prétraitement et le post-traitement.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
