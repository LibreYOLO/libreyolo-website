---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS : estimation de profondeur monoculaire dans LibreYOLO'
description: >-
  Utilisez MiDaS dans LibreYOLO pour l'estimation de profondeur monoculaire.
  Installez, prédisez, validez et exportez deux variantes sous licence MIT,
  depuis les miroirs LibreYOLO.
lead: >-
  MiDaS est un modèle d'estimation de profondeur relative monoculaire entraîné
  avec une loss invariante à l'échelle et au décalage sur des datasets mélangés,
  la lignée de travaux qui a établi le protocole de transfert de profondeur
  zero-shot que les familles suivantes réutilisent. LibreYOLO le prend en charge
  pour la tâche de profondeur : prédiction et validation zero-shot, sans voie
  d'entraînement.
keywords:
  - MiDaS
  - estimation de profondeur monoculaire
  - MiDaS python
  - DPT
  - profondeur relative
  - carte de profondeur
  - profondeur zero-shot
last_verified: "1.6.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Télécharge le checkpoint en miroir au premier usage.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Variante Small
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encodeur EfficientNet-Lite3, plus petit et plus rapide que le l
        DPT-Large.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory route selon l'extension du fichier : un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 64537aa3ad3a6f8f
---

## Installation

MiDaS nécessite l'extra `midas` pour ses encodeurs timm.

```bash
pip install "libreyolo[midas]"
```

## Prédire

Les checkpoints s et l se téléchargent depuis les miroirs de LibreYOLO sous la licence MIT de l'éditeur et sont mis en cache localement.

<code-tabs name="predict" />

`result.depth_map` porte une carte dense de profondeur inverse relative : les
valeurs élevées correspondent aux points les plus proches de la caméra, et ces
valeurs n'ont ni unité métrique ni échelle commune d'une image à l'autre.
`save=True` écrit sur le disque une visualisation colorisée de cette carte ;
`Results.plot()` affiche la carte de profondeur. Voir [la prédiction](/docs/predict)
pour les sources, le streaming et le traitement des résultats.

## Variantes

Deux variantes avec des encodeurs différents, pas seulement deux échelles du
même encodeur. `s` est MiDaS v2.1 Small, un encodeur EfficientNet-Lite3. `l` est
DPT-Large, un encodeur ViT-L/16 doté du décodeur DPT que MiDaS a introduit pour
la prédiction dense. Leur prétraitement diffère aussi : `s` utilise un
redimensionnement à rapport d'aspect de type upper-bound avec une normalisation
mean/std ImageNet, `l` utilise un redimensionnement à rapport d'aspect minimal
avec une moyenne et un écart-type de 0.5. Prenez `s` pour un CNN plus léger, `l`
pour l'exactitude du décodeur transformer.

L'entraînement n'est pas proposé pour cette famille. `LibreMiDaS.train()` lève
`NotImplementedError` sans condition.

## Valider

`val()` exécute le validateur de profondeur commun : il aligne chaque prédiction
sur sa vérité terrain avec une échelle et un décalage estimés aux moindres
carrés, image par image, puis rapporte les métriques standard de profondeur
relative en zero-shot, AbsRel, RMSE et les trois seuils delta.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`, avec `depth_map` à la place des boîtes.

<code-tabs name="export" />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
