---
title: YOLOv3
families:
  - yolo3
seo_title: "YOLOv3 dans LibreYOLO\_: prédire, valider et exporter"
description: "Exécutez YOLOv3 dans LibreYOLO\_: une famille historique figée, réservée à l'inférence, avec les tailles tiny, base et SPP. Prédisez, validez et exportez sous une licence du domaine public."
lead: >-
  YOLOv3 est le détecteur Darknet-53 qui a ajouté la prédiction multi-échelle et
  des classificateurs logistiques indépendants à la lignée YOLO. LibreYOLO le
  conserve comme pièce historique figée, réservée à l'inférence, dans les
  tailles tiny, base et SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - détection d'objets
  - détection multi-échelle
  - ancien modèle YOLO
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Taille SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La variante SPP ajoute un bloc de spatial pyramid pooling avant les
        # têtes de détection et s'exécute à sa propre taille d'entrée native.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLO3b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Installer

YOLOv3 ne nécessite aucun extra au-delà du package de base.

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
détecteur ne demande donc de modifier qu'une ligne. `conf` filtre selon le
seuil de confiance et `iou` selon le seuil de NMS, appliqués par échelle avant
de fusionner les bounding boxes des trois têtes. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour la validation.

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

