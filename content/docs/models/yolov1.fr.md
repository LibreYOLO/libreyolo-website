---
title: YOLOv1
families:
  - yolo1
seo_title: "YOLOv1 dans LibreYOLO\_: prédire, valider et exporter"
description: "Exécutez le détecteur YOLOv1 d'origine dans LibreYOLO\_: une famille historique figée, réservée à l'inférence. Prédisez, validez et exportez sous une licence du domaine public."
lead: "YOLOv1 est le détecteur original de 2016 qui a donné son nom à la famille YOLO\_: un seul réseau convolutionnel doté d'une tête entièrement connectée prédit toutes les bounding boxes et tous les scores de classes en une passe, sans ancres. LibreYOLO le conserve comme pièce historique figée, réservée à l'inférence."
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - détection d'objets
  - Pascal VOC
  - ancien modèle YOLO
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLO1b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Installer

YOLOv1 ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Cette famille est réservée à l'inférence\u00a0: `train()` lève
`NotImplementedError`, cette page ne comporte donc aucune section Entraîner.
La prédiction, la validation et l'export sont tous pris en charge. Les poids
sont téléchargés depuis Hugging Face à la première utilisation et mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. Deux aspects sont propres
à cette famille. Le checkpoint publié est entraîné sur Pascal VOC (2007+2012),
et non sur COCO. `box.cls` indexe donc les 20 catégories VOC (aeroplane,
bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog,
horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor) plutôt
que les 80 catégories COCO. De plus, la tête de détection entièrement
connectée accepte une seule image à la fois. Une liste de sources est ainsi
traitée en boucle et non comme un véritable batch. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur un dataset utilisant le même
espace d'étiquettes de type VOC que celui sur lequel le checkpoint a été
entraîné.

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

