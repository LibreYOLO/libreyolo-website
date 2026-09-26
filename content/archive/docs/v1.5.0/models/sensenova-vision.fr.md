---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: "SenseNova-Vision dans LibreYOLO\_: 7 tâches, un checkpoint"
description: >-
  Utilisez SenseNova-Vision dans LibreYOLO pour la détection, la segmentation,
  la segmentation panoptique, la pose, les points, la profondeur et l'OCR à
  partir d'un même checkpoint génératif guidé.
lead: "SenseNova-Vision est un modèle multimodal unifié qui formule les tâches de vision comme une génération guidée sur un décodeur partagé\_: les bounding boxes, points, points clés et mots OCR sont produits sous forme de texte balisé, tandis que les cartes de profondeur, de masques et panoptiques sont produites sous forme d'images rendues par un décodeur. LibreYOLO le charge par l'intermédiaire de LibreVLM et prend en charge sept tâches à partir de l'unique checkpoint 7B."
keywords:
  - SenseNova-Vision
  - SenseTime
  - modèle multimodal unifié
  - Bagel
  - détection guidée par prompt
  - perception dense
  - segmentation par expression référente
  - segmentation panoptique
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() change de tâche sur le même modèle déjà chargé.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentation par expression référente et panoptique
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # La segmentation suit une expression référente : elle exige une phrase
        cible, pas une liste de classes.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Sans vocabulaire personnalisé, la segmentation panoptique utilise les
        catégories

        # panoptiques COCO sur lesquelles le checkpoint a été affiné.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Points, pose et OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Sans vocabulaire défini, la pose utilise "person" par défaut.
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Installer

SenseNova-Vision nécessite son propre extra. Celui-ci installe `accelerate`
pour la répartition des grands modèles dont ce checkpoint a besoin et, sur les
plateformes autres que macOS, `bitsandbytes` pour le chargement en 4 bits.

```bash
pip install "libreyolo[sensenova]"
```

Le checkpoint est répliqué sur Hugging Face sous l'organisation propre à
LibreYOLO et se télécharge automatiquement à la première utilisation. Il est
sous licence CC BY-NC 4.0, réservée à un usage non commercial, et le chargeur
affiche cet avertissement avant chaque téléchargement automatique. Consultez
la section Licence ci-dessous.

## Prédire

<code-tabs name="predict" />

Chaque prédiction est un décodage par diffusion sur le backbone Bagel-MoT
partagé. Il s'agit donc d'un modèle de capacités et non d'un modèle temps
réel\u00a0: attendez-vous à une latence par image nettement supérieure à celle d'un
détecteur ou segmenteur spécialisé. `dtype="auto"` (la valeur par défaut)
charge le modèle en bf16 sur un GPU disposant de suffisamment de mémoire et se
rabat ailleurs sur une quantification NF4 en 4 bits, qui nécessite
`bitsandbytes`\u00a0; transmettez `dtype="bf16"` pour imposer la pleine précision
sur un GPU suffisamment grand. Le paramètre `noise_seed=42` à la construction
initialise le sampler de diffusion afin de rendre les sorties denses
reproductibles\u00a0; transmettez `noise_seed=None` pour désactiver cette
initialisation.

Les sept tâches partagent un seul checkpoint chargé\u00a0: `set_task()` passe de
l'une à l'autre sans rechargement. `set_classes()` définit le vocabulaire
actif\u00a0; la détection, les points, la pose et la segmentation panoptique
acceptent une liste de classes, tandis que la segmentation suit une expression
référente et exige exactement la phrase désignant l'élément à isoler. Chaque
tâche renvoie l'objet `Results` standard avec une charge utile différente\u00a0:
`boxes` pour detect, `points` pour point, `boxes` et `keypoints` pour pose,
`ocr` pour OCR, `depth_map` pour depth, `masks` pour segment et `panoptic`
(avec `segments_info`) pour panoptic. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Checkpoints

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
