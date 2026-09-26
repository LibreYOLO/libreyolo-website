---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: "Faster R-CNN dans LibreYOLO\_: prédire, valider et exporter"
description: >-
  Exécutez Faster R-CNN dans LibreYOLO pour la détection d'objets sur quatre
  backbones. Installez, prédisez, validez et exportez le portage torchvision
  sous licence BSD-3-Clause.
lead: >-
  Faster R-CNN détecte les objets avec un réseau de propositions de régions qui
  alimente un classifieur à deux étages, l'architecture qui a fait entrer les
  propositions de régions dans le même réseau entraîné au lieu d'en faire une
  étape séparée. LibreYOLO porte l'implémentation torchvision pour la détection.
keywords:
  - Faster R-CNN
  - détection d'objets python
  - détecteur à deux étages
  - region proposal network
  - RPN
  - torchvision détection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Installation

Faster R-CNN ne demande aucun extra optionnel. Tout ce qu'il importe fait
partie de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
passer à un autre détecteur ne demande qu'une ligne. `conf` et `iou` fixent
les seuils de confiance et de NMS ; Faster R-CNN conserve l'étape de NMS
d'origine, contrairement à un détecteur à base de requêtes. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Quatre tailles, chacune une configuration torchvision différente plutôt qu'une
version redimensionnée de la même : `n` est MobileNetV3-Large avec une entrée
de 320 px, `s` est le même backbone à 800 px, `m` est ResNet-50 avec une
pyramide de caractéristiques, et `l` est la révision v2, avec une tête de
propositions de régions plus profonde et une tête de boîtes à quatre
convolutions à la place de celle de `m`. `n` et `s` échangent de l'exactitude
contre un backbone plus léger.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset au
format avec lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

Faster R-CNN n'exporte que vers ONNX, avec une taille de batch de 1. Le graphe
exporté conserve en son sein l'étape de redimensionnement d'origine, si bien
que LibreYOLO force `dynamic=True` quelle que soit la valeur passée, afin de
garder le graphe valide pour des sources qui ne sont pas carrées. Un fichier
`.onnx` exporté se recharge via `LibreYOLO()` selon son extension de fichier
et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
