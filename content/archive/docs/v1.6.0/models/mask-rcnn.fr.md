---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: "Mask R-CNN dans LibreYOLO\_: prédire, valider et exporter"
description: >-
  Exécutez Mask R-CNN dans LibreYOLO pour la détection d'objets et la
  segmentation d'instances. Installez, prédisez, validez et exportez le portage
  torchvision sous licence BSD-3-Clause.
lead: >-
  Mask R-CNN ajoute à Faster R-CNN une branche de masques par région, qui prédit
  un masque de segmentation à côté de chaque boîte qu'il détecte. LibreYOLO
  porte l'implémentation torchvision pour la détection et la segmentation
  d'instances.
keywords:
  - Mask R-CNN
  - segmentation d'instances
  - détection d'objets python
  - Faster R-CNN
  - Mask R-CNN pytorch
  - torchvision
  - détecteur à deux étages
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Boîtes uniquement
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" ignore la tête de masques et renvoie les boîtes du
        # même checkpoint, sans masques dans le résultat.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # masques
        print(metrics["metrics/mAP50-95(B)"])   # boîtes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Installation

Mask R-CNN ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
passer à un autre détecteur ne demande qu'une ligne. Charger le checkpoint sans
argument `task` renvoie des masques d'instances, puisque la segmentation est la
tâche par défaut de cette famille ; `result.masks` les porte alors à côté des
boîtes. Passer `task="detect"` charge les mêmes poids sans la tête de masques
et ne renvoie que des boîtes. `conf` et `iou` fixent les seuils de confiance et
de NMS ; Mask R-CNN conserve son étape de NMS d'origine, contrairement à un
détecteur à base de requêtes. Voir [la prédiction](/docs/predict) pour les
sources, le streaming et le traitement des résultats.

## Variantes

Un seul backbone : ResNet-50 avec une pyramide de caractéristiques, en
utilisant le builder Mask R-CNN v2 de torchvision. Le checkpoint publié est
sous licence BSD-3-Clause et sert les deux tâches de cette famille, il n'y a
donc aucune taille entre laquelle choisir.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/`. Avec la tâche de
segmentation par défaut de ce checkpoint, la clé `metrics/mAP50-95` seule
contient le score des masques, et la même exécution reporte les boîtes sous le
suffixe `(B)`, si bien que les deux sont disponibles en une seule passe.

<code-tabs name="val" />

## Exporter

<export-matrix />

Mask R-CNN n'exporte que vers ONNX, avec une taille de batch de 1. Le graphe
exporté conserve en son sein les étapes de redimensionnement et de collage des
masques d'origine, si bien que LibreYOLO force `dynamic=True` quelle que soit
la valeur passée, afin de garder le graphe valide pour des sources qui ne sont
pas carrées. Un fichier `.onnx` exporté se recharge via `LibreYOLO()` selon son
extension de fichier et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille. Le seul checkpoint
ci-dessous est listé sous detect, mais le même fichier se charge aussi pour la
segmentation : ne passez pas d'argument `task` et il renvoie des masques par
défaut.

<checkpoint-table />

## Licence

<provenance-box>

Mask R-CNN est construit comme une sous-classe du wrapper Faster R-CNN de
LibreYOLO : il partage la même source torchvision et la même licence
BSD-3-Clause, et ajoute le prédicteur de masques et la tête RoI de masques du
même commit porté.

</provenance-box>
