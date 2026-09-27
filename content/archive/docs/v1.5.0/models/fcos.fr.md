---
title: FCOS
families:
  - fcos
seo_title: "FCOS dans LibreYOLO\_: prédire, valider et exporter"
description: >-
  Exécutez FCOS dans LibreYOLO pour la détection d'objets sans ancres.
  Installez, prédisez, validez et exportez le portage torchvision sous licence
  BSD-3-Clause, ResNet-50/FPN.
lead: >-
  FCOS détecte les objets pixel par pixel au lieu de s'appuyer sur un ensemble
  d'ancres prédéfinies, en prédisant une boîte et un score de centerness à
  chaque position de la carte de caractéristiques. LibreYOLO porte
  l'implémentation torchvision pour la détection.
keywords:
  - FCOS
  - détection d'objets sans ancres
  - détection d'objets python
  - détecteur one-stage
  - torchvision détection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Installation

FCOS ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoient toutes les familles, si bien
que passer à un autre détecteur ne demande qu'une ligne de changement. Appeler
le modèle sans argument de seuil applique les valeurs par défaut publiées par
FCOS, `conf=0.2`, `iou=0.6` et `max_det=100` ; passez l'un des trois pour les
remplacer. FCOS conserve une étape finale de NMS sur ses prédictions par pixel.
Voir [la prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

## Variantes

Une seule taille : ResNet-50 avec une pyramide de caractéristiques, la seule
variante que cette famille reconnaît.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset dans le
format avec lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

FCOS s'exporte vers ONNX, TorchScript et OpenVINO. FCOS préserve le rapport
d'aspect de la source avant l'exécution du graphe, si bien que LibreYOLO force
`dynamic=True` pour les chemins ONNX et OpenVINO quelle que soit la valeur
passée, afin de garder le graphe valide pour des formes d'entrée complétées par
padding. Un fichier `.onnx` exporté se recharge via `LibreYOLO()` d'après son
extension et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
