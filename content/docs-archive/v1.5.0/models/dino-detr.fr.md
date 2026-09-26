---
title: DINO-DETR
families:
  - dinodetr
seo_title: "DINO-DETR\_: prédire et exporter sous Apache-2.0"
description: >-
  Utilisez DINO-DETR dans LibreYOLO pour la détection d'objets. Installez,
  prédisez, validez et exportez trois tailles à ancres de débruitage, toutes
  sous licence Apache-2.0.
lead: >-
  DINO-DETR, publié par IDEA Research sous le nom DINO, combine un entraînement
  par débruitage contrastif et une sélection mixte des requêtes par-dessus
  l'attention éparse de Deformable DETR. LibreYOLO fournit trois tailles pour la
  détection, en inférence seule.
keywords:
  - DINO-DETR
  - DINO
  - detection transformer
  - ancres de débruitage
  - mixed query selection
  - détection d'objets python
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Installation

DINO-DETR ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base, avec le même cœur d'attention déformable
multi-échelle en PyTorch pur que la famille Deformable DETR de LibreYOLO.

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

Dans LibreYOLO, DINO-DETR est en inférence seule. Le projet d'origine
l'entraîne avec un débruitage contrastif et un appariement hongrois ; cette
recette n'est pas implémentée ici, donc `train()` lève `NotImplementedError`.

## Variantes

Trois checkpoints, tous à la même résolution d'entrée. `r50` et `r50s5`
partagent un backbone ResNet-50 et diffèrent par le nombre d'échelles de
cartes de caractéristiques qui alimentent le décodeur, quatre contre cinq.
`swinl` remplace le backbone par Swin-L et échantillonne lui aussi cinq
échelles.

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

<provenance-box>

Les trois checkpoints officiels proviennent du dossier de publication Google
Drive des auteurs, et non d'une model card Hugging Face. Le dépôt d'origine
déclare Apache-2.0 au niveau du dépôt mais n'attache ni fichier de licence ni
métadonnées de licence aux checkpoints eux-mêmes ; la base de redistribution
est donc cette déclaration au niveau du dépôt plutôt qu'une concession propre
aux checkpoints. Chaque miroir LibreYOLO fournit le texte de licence
Apache-2.0 d'origine à l'identique, accompagné d'une note qui l'explique.

</provenance-box>

## Citation

<citation-block />
