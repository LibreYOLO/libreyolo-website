---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR : prédire et exporter sous licence Apache-2.0'
description: >-
  Utilisez LW-DETR dans LibreYOLO pour la détection d'objets en temps réel.
  Installez, prédisez, validez et exportez cinq tailles basées sur ViT, toutes
  sous licence Apache-2.0.
lead: >-
  Un transformer de détection à ViT simple que Baidu a positionné comme une
  alternative temps réel aux détecteurs YOLO. LibreYOLO fournit cinq tailles
  pour la détection, en inférence seule.
keywords:
  - LW-DETR
  - detection transformer
  - détection d'objets temps réel
  - plain ViT
  - DETR
  - détection d'objets python
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory route selon l'extension du fichier, donc un artefact
        exporté

        # se charge comme n'importe quel checkpoint et renvoie le même Results.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Installation

LW-DETR ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
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
filtrent la sélection des requêtes ; `iou` est accepté par parité d'API mais
n'a aucun effet, car le décodeur est un prédicteur d'ensembles sans étape de
NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

Dans LibreYOLO, LW-DETR est en inférence seule. Le projet d'origine l'entraîne
avec la supervision un-à-plusieurs de Group-DETR sur plusieurs groupes de
requêtes et une loss de classification tenant compte de l'IoU ; cette recette
n'est pas branchée ici, donc `train()` lève `NotImplementedError`.

## Variantes

Cinq tailles, qui partagent toutes l'encodeur ViT simple, le projecteur
multi-échelle et le décodeur de deformable DETR, et qui tournent toutes à la
même résolution d'entrée. Les deux plus petites partagent une largeur
d'encodeur et se séparent par la profondeur en blocs ; les deux suivantes
partagent un encodeur plus large et se séparent par le nombre de niveaux du
projecteur qui alimentent le décodeur ; la plus grande passe à l'encodeur le
plus large.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset au
format avec lequel vous avez entraîné.

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

## Citation

<citation-block />
