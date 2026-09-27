---
title: Swin Transformer
families:
  - swin
seo_title: "Swin Transformer\_: classifier des images avec LibreSwin de LibreYOLO"
description: "Prédisez, validez et exportez des classificateurs Swin Transformer avec LibreYOLO. Poids sous licence MIT\_; le fine-tuning n'est pas encore pris en charge."
lead: >-
  Swin Transformer V1 est un vision transformer hiérarchique qui calcule
  l'attention dans des fenêtres locales décalées plutôt que sur l'image entière.
  LibreYOLO fournit quatre tailles pour la classification d'images.
keywords:
  - Swin Transformer
  - vision transformer hiérarchique
  - shifted window attention
  - classification d'images
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwint-cls.pt")


        # data est la racine d'un répertoire avec des sous-ensembles de dossiers
        par classe

        # train/ et val/ (structure ImageFolder), et non le YAML d'un dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Installer

Swin ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
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
donnent leurs confiances. Chaque taille utilise une entrée fixe de 224\u00a0px, car
la dernière étape d'attention est construite pour cette résolution. La
prédiction, la validation et l'export lèvent tous une erreur si vous
transmettez une autre valeur à `imgsz`. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Quatre tailles, de tiny à large, sont construites à partir de la même tour à
fenêtres décalées et se distinguent par la largeur des embeddings et la
profondeur des étapes. La taille large est pré-entraînée sur ImageNet-22k et
affinée sur ImageNet-1k\u00a0; les trois autres sont directement entraînées sur
ImageNet-1k. LibreYOLO fournit cette famille uniquement pour l'inférence\u00a0: la
prédiction, la validation top-1/top-5 de type ImageNet et l'export sont pris en
charge, mais la recette d'entraînement ImageNet upstream n'est pas implémentée.

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

## Citation

<citation-block />

