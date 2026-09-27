---
title: ZipDepth
families:
  - zipdepth
seo_title: "ZipDepth\_: estimation de profondeur monoculaire légère dans LibreYOLO"
description: >-
  Utilisez ZipDepth dans LibreYOLO pour une estimation de profondeur monoculaire
  légère. Installez, prédisez, validez et exportez deux checkpoints sous licence
  MIT.
lead: "ZipDepth est un CNN compact reparamétrable, distillé depuis Depth Anything V2 Large, qui prédit une carte dense de profondeur inverse relative. LibreYOLO le prend en charge pour la tâche de profondeur\_: prédiction et validation zero-shot, sans entraînement."
keywords:
  - ZipDepth
  - estimation profondeur monoculaire
  - modèle profondeur edge
  - profondeur relative
  - carte de profondeur
  - CNN reparamétrable
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Même encodeur, avec une tête d'upsampling sans unfold pour les
        compilateurs dépourvus

        # de prise en charge de gather/unfold. La sortie est visuellement
        équivalente au checkpoint b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Installer

ZipDepth ne nécessite aucun extra facultatif. Tous ses imports sont inclus
dans l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

`result.depth_map` contient une carte dense de profondeur inverse relative\u00a0:
les valeurs élevées représentent les éléments les plus proches de la caméra,
et les valeurs n'ont aucune unité métrique ni échelle commune aux différentes
images. `save=True` écrit sur disque une visualisation en fausses couleurs de
cette carte. `Results.plot()` ne couvre pas cette famille, car cette méthode
est définie uniquement pour les normales de surface et les contours. Consultez
la [prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Deux checkpoints de même capacité d'encodeur ne se distinguent que par leur
tête d'upsampling entraînée. `b` utilise un upsampling convexe et s'exécute sur
GPU ou CPU. `bnpu` emploie un décodeur sans unfold pour les NPU et les
compilateurs edge dépourvus de prise en charge de gather/unfold\u00a0; sa sortie est
documentée comme visuellement équivalente à celle de `b`. Choisissez `bnpu`
lorsque la cible d'export est un runtime contraint, et `b` dans les autres cas.

Les deux checkpoints ont été distillés à partir de pseudo-étiquettes de Depth
Anything V2 Large. Cette famille constitue donc le niveau compact orienté edge
de la tâche de profondeur de LibreYOLO, aux côtés des encodeurs Depth Anything
V2 plus grands.

Cette famille ne propose aucun entraînement. `LibreZipDepth.train()` lève
toujours `NotImplementedError`\u00a0: la recette upstream distille des
pseudo-étiquettes sur un vaste ensemble d'images qui ne peut pas être
reproduit comme une exécution d'entraînement LibreYOLO. Entraînez le modèle
upstream sur [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth),
puis convertissez le résultat avec `weights/convert_zipdepth_weights.py`.

## Valider

`val()` exécute le validateur de profondeur partagé\u00a0: il aligne chaque
prédiction sur sa vérité terrain avec une échelle et un décalage par image
obtenus par moindres carrés, puis rapporte les mesures standard de profondeur
relative zero-shot, AbsRel, RMSE et les trois seuils delta.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export suit un contrat dense à résolution fixe\u00a0: l'image source est étirée
jusqu'au canevas exporté, puis la carte de profondeur renvoyée est
redimensionnée au canevas d'origine. Un artefact exporté se recharge dans
`LibreYOLO()` grâce au suffixe de son fichier. Un fichier `.onnx` ou `.ncnn`
se comporte donc comme un checkpoint et renvoie les mêmes `Results`, avec
`depth_map` à la place des bounding boxes.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
