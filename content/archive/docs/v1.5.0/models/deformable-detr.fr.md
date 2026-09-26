---
title: Deformable DETR
families:
  - deformable_detr
seo_title: "Deformable DETR\_: prédire et exporter, Apache-2.0"
description: >-
  Utilisez Deformable DETR dans LibreYOLO pour la détection d'objets. Installez,
  prédisez, validez et exportez cinq tailles à attention éparse, toutes sous
  licence Apache-2.0.
lead: >-
  Deformable DETR remplace la cross-attention dense de DETR par un
  échantillonnage épars et multi-échelle autour de chaque point de référence, ce
  qui a rendu les détecteurs transformer réellement entraînables. LibreYOLO
  fournit cinq tailles pour la détection, en inférence seule.
keywords:
  - Deformable DETR
  - detection transformer
  - attention éparse
  - attention multi-échelle
  - détection d'objets python
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Installation

Deformable DETR ne demande aucun extra optionnel. Tout ce qu'il importe fait
partie de l'installation de base, avec un cœur d'attention déformable
multi-échelle en PyTorch pur.

```bash
pip install libreyolo
```

Installer `libreyolo[hub-kernels]` est optionnel. Une fois le paquet `kernels`
présent, LibreYOLO récupère au runtime un kernel compilé d'attention
déformable multi-échelle depuis le Hugging Face Hub et l'utilise à la place du
cœur en PyTorch pur ; `LIBREYOLO_HUB_KERNELS=0` le désactive à nouveau.

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

Dans LibreYOLO, Deformable DETR est en inférence seule. Le projet d'origine
l'entraîne avec un appariement hongrois et une focal loss de classification ;
cette recette n'est pas implémentée ici, donc `train()` lève
`NotImplementedError`.

## Variantes

Cinq checkpoints couvrent les configurations publiées, toutes à la même
résolution d'entrée. `r50ss` limite l'attention à une seule échelle de
caractéristiques ; `r50ssdc5` y ajoute un étage C5 dilaté dans le backbone.
`r50` est la configuration multi-échelle par défaut, qui échantillonne sur
quatre niveaux de cartes de caractéristiques. `r50refine` ajoute un raffinement
itératif des bounding box au fil des couches du décodeur, et `r50twostage`
génère ses propositions de régions initiales à partir de la sortie de
l'encodeur au lieu de requêtes apprises.

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
