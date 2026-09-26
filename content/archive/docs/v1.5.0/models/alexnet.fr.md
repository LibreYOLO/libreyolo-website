---
title: AlexNet
families:
  - alexnet
seo_title: "AlexNet\_: exécuter le classifieur ImageNet classique dans LibreYOLO"
description: "Prédire, valider et exporter AlexNet avec LibreYOLO. Poids torchvision sous licence BSD-3-Clause\_; le fine-tuning n'est pas encore pris en charge."
lead: >-
  AlexNet est le réseau convolutif qui a remporté l'ILSVRC 2012 et qui a
  contribué à lancer l'ère du deep learning en vision par ordinateur. LibreYOLO
  fournit la révision ultérieure de l'architecture, à une seule tour, pour la
  classification d'images.
keywords:
  - AlexNet
  - ImageNet
  - réseau de neurones convolutif
  - classification d'images python
  - classifieur d'images pré-entraîné
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data est un répertoire racine avec des splits train/ et val/ en
        # dossiers par classe (format ImageFolder), pas un YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Installation

AlexNet ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

Un classifieur renvoie `result.probs` au lieu de `result.boxes` : `top1` et
`top5` donnent les indices de classe, `top1conf` et `top5conf` donnent leurs
scores de confiance. Voir [la prédiction](/docs/predict) pour les sources, le
streaming et le traitement des résultats.

## Variantes

Une seule taille. Le graphe fourni est la révision ultérieure à une seule tour
publiée par torchvision, avec 64 filtres en première couche et sans local
response normalization, et non l'architecture d'origine de 2012 à deux GPU.
LibreYOLO fournit cette famille en inférence seule : la prédiction, la
validation top-1/top-5 façon ImageNet et l'export sont pris en charge, et le
fine-tuning n'est pas implémenté.

## Valider

`val()` s'exécute sur un split de type ImageFolder (un répertoire avec des
sous-dossiers `train/` et `val/`, un dossier par classe) et renvoie
l'exactitude top-1 et top-5.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. [L'export](/docs/export) liste les
arguments que chaque format accepte, ainsi que les extras que quelques-uns
ajoutent.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>
