---
title: VGG
families:
  - vgg
seo_title: "VGG\_: exécuter des classificateurs d'images VGG-16/19 dans LibreYOLO"
description: "Prédisez, validez et exportez des classificateurs VGG avec LibreYOLO. Poids torchvision sous licence BSD-3-Clause\_; le fine-tuning n'est pas encore pris en charge."
lead: >-
  VGG est un classificateur d'images convolutionnel construit à partir
  d'empilements uniformes de petites convolutions 3x3 plutôt que de filtres plus
  grands. LibreYOLO fournit les tailles à 16 et 19 couches, sans et avec
  normalisation par batch, pour la classification d'images.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - réseau de neurones convolutionnel
  - classification d'images
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreVGG16-cls.pt")


        # data est la racine d'un répertoire avec des sous-ensembles de dossiers
        par classe

        # train/ et val/ (structure ImageFolder), et non le YAML d'un dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreVGG16-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Installer

VGG ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

Un classificateur renvoie `result.probs` au lieu de `result.boxes`\u00a0: `top1` et
`top5` donnent les indices de classes, tandis que `top1conf` et `top5conf`
donnent leurs confiances. La prédiction s'exécute avec une entrée fixe de
224\u00a0px et lève une erreur si vous transmettez une autre valeur à `imgsz`.
Consultez la [prédiction](/docs/predict) pour les sources, le streaming et la
gestion des résultats.

## Variantes

Quatre tailles\u00a0: 16 ou 19 couches convolutionnelles, chacune déclinée sans et
avec normalisation par batch. Les poids fournis proviennent de l'entraînement
ImageNet ultérieur de torchvision à partir de zéro, et non de conversions de
la version Caffe originale publiée par le groupe d'Oxford en 2014. LibreYOLO
fournit cette famille uniquement pour l'inférence\u00a0: la prédiction, la
validation top-1/top-5 de type ImageNet et l'export sont pris en charge, mais
le fine-tuning n'est pas implémenté.

## Valider

`val()` s'exécute sur un sous-ensemble de type ImageFolder (un répertoire avec
des sous-dossiers `train/` et `val/`, un dossier par classe) et renvoie
l'exactitude top-1 et top-5.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. La page [Export](/docs/export) énumère les
arguments acceptés par chaque format ainsi que les extras ajoutés par certains.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

