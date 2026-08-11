---
title: YOLOX
families:
  - yolox
seo_title: "YOLOX\_: prédire, entraîner et exporter sous licence Apache-2.0"
description: "Utilisez YOLOX dans LibreYOLO pour la détection d'objets\_: installez, prédisez, entraînez, validez et exportez sous licence Apache-2.0."
lead: >-
  YOLOX est un détecteur mono-étape sans ancres, doté d'une tête découplée de
  classification et de régression, entraîné avec l'affectation d'étiquettes
  SimOTA. LibreYOLO le prend en charge pour la détection.
keywords:
  - YOLOX
  - détection d'objets
  - détection sans ancres
  - tête découplée
  - SimOTA
  - détection d'objets temps réel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Sur COCO
      language: bash
      code: >
        # Le YAML COCO fourni contient un script de téléchargement intégré, il
        nécessite donc

        # une autorisation explicite sauf si le dataset est déjà présent
        localement.

        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Installer

YOLOX ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. `conf` définit le seuil de
confiance et `iou` le seuil de NMS appliqué aux trois échelles de prédiction
découplées. Consultez la [prédiction](/docs/predict) pour les sources, le
streaming et la gestion des résultats.

## Variantes

Six tailles partagent le même backbone CSP et le même neck PAFPN. Les deux
plus petites, `n` et `t`, s'exécutent à une résolution d'entrée fixe plus
faible que les quatre autres\u00a0; le tableau de benchmark ci-dessous donne la
valeur exacte de chacune.

<benchmark-table task="detect" />

<va-embed />

## Entraîner

<code-tabs name="train" />

Sans autre réglage, le trainer exécute 300 époques avec `lr0=0.01`, un momentum
SGD de 0.9, un warmup de 5 époques, et désactive les augmentations mosaic et
mixup pendant les 15 dernières époques. `train()` accepte également un
argument `pretrained`, mais sa valeur n'est jamais lue dans la méthode\u00a0:
l'entraînement continue toujours depuis les poids ayant servi à construire le
modèle. `pretrained=False` ne réinitialise donc pas le réseau.

La valeur par défaut de `imgsz` est fixe dans la configuration d'entraînement
de base, elle ne correspond pas à la résolution native du checkpoint chargé.
Cela concerne tout particulièrement les checkpoints `n` et `t`\u00a0: si vous
poursuivez l'entraînement de l'un ou de l'autre sans définir explicitement
`imgsz`, il passe à la plus grande valeur par défaut et n'utilise plus la
taille inférieure avec laquelle il a été publié.

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
le prétraitement et le post-traitement. Un export CoreML peut intégrer la NMS
au graphe avec `nms=True`\u00a0; YOLOX et YOLOv9 sont les deux seules familles pour
lesquelles ce paramètre est actuellement accepté.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />

