---
title: DETR
families:
  - detr
seo_title: "DETR\_: prédire et exporter sous Apache-2.0"
description: >-
  Utilisez DETR, le transformer de détection original, dans LibreYOLO.
  Installez, prédisez, validez et exportez quatre tailles basées sur ResNet,
  toutes sous licence Apache-2.0.
lead: "DETR est le transformer de détection original\_: il prédit un ensemble fixe d'objets avec un décodeur transformer à appariement hongrois, au lieu d'ancres ou d'une grille dense. LibreYOLO fournit quatre tailles pour la détection, en inférence seule."
keywords:
  - DETR
  - detection transformer
  - détection d'objets python
  - appariement hongrois
  - décodeur transformer
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le même
        # objet Results.
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Installation

DETR ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
passer à un autre détecteur ne demande qu'une ligne. `conf` et `max_det`
filtrent la sélection des requêtes ; `iou` est accepté par parité d'API mais
n'a aucun effet, car le décodeur est un prédicteur d'ensembles sans étape de
NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

Dans LibreYOLO, DETR est en inférence seule. Le projet d'origine l'entraîne
pendant 500 époques avec un appariement hongrois ; cette recette n'est pas
implémentée ici, donc `train()` lève `NotImplementedError`.

## Variantes

Quatre checkpoints combinent deux profondeurs de backbone, ResNet-50 ou
ResNet-101, avec un étage C5 dilaté optionnel : les variantes DC5 gardent le
dernier étage du backbone à pleine résolution au lieu de le sous-échantillonner
davantage, si bien que le décodeur lit une carte de caractéristiques plus fine à
taille d'entrée égale. Les quatre partagent 100 requêtes d'objets apprises et un
encodeur-décodeur transformer à six couches, et toutes tournent à la même
résolution d'entrée.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset au format
avec lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. [L'export](/docs/export) liste les
arguments que chaque format accepte.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>
