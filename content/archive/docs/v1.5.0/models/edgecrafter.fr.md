---
title: EdgeCrafter
families:
  - ec
seo_title: "EdgeCrafter\_: détecter, estimer la pose et segmenter dans LibreYOLO"
description: >-
  Utilisez EdgeCrafter dans LibreYOLO pour la détection, l'estimation de pose et
  la segmentation d'instances. Installez, prédisez, validez et exportez, avec du
  code sous licence MIT.
lead: "Un transformer de vision compact pour la prédiction dense sur matériel edge, publié en amont sous la forme de trois modèles frères\_: ECDet, ECPose et ECSeg. LibreYOLO charge les trois comme une seule famille, la tâche étant portée par le checkpoint."
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - vision transformer compact
  - détection d'objets python
  - estimation de pose python
  - segmentation d'instances
  - inférence sur appareil edge
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -pose du nom de fichier sélectionne la tête de points
        # clés, aucun argument task n'est donc nécessaire ici.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Demande un dataset de points clés à une seule classe dont le data.yaml
        # déclare kpt_shape, et imgsz à la taille native du checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentation d'instances
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Demande des étiquettes polygonales, et imgsz à la taille native du
        modèle.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masques
        print(metrics["metrics/mAP50-95(B)"])   # boîtes
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory route selon l'extension du fichier, donc un artefact
        exporté

        # se charge comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreECs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Installation

EdgeCrafter ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base.

```bash
pip install libreyolo
```

Le fine-tuning par adaptateurs avec `lora=True` fait exception, et demande
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

La tâche vient du nom de fichier, si bien qu'un checkpoint `-pose` ou `-seg`
sélectionne sa propre tête et ne prend aucun argument task. Les trois renvoient
l'objet `Results` que renvoie chaque famille, avec `result.keypoints` en plus
pour la pose et `result.masks` pour la segmentation. La pose couvre une seule
classe, la personne, avec les 17 points clés COCO, et ce nombre est fixé à la
construction du modèle. Elle n'a pas de tête de boîtes, si bien que chaque boîte
de pose est l'étendue englobante de ses propres points clés, et le troisième
canal de point clé est une constante plutôt qu'un score par point.

`conf` et `max_det` filtrent la sélection des queries ; `iou` est accepté par
parité d'API mais n'a aucun effet, car les trois têtes décodent un ensemble de
queries sans étape de NMS. Voir [la prédiction](/docs/predict) pour les sources,
le streaming et le traitement des résultats.

## Variantes

Quatre tailles. Elles tournent toutes à la même résolution d'entrée, la table
les sépare donc par nombre de paramètres et par exactitude.

<benchmark-table task="detect" />

<va-embed />

En amont, ECDet, ECPose et ECSeg sont publiés comme trois modèles distincts
plutôt que comme un seul modèle à trois têtes. Ils partagent le backbone ECViT
et l'encodeur hybride et ne diffèrent que par la tête, si bien que LibreYOLO les
réunit en une seule famille et laisse le nom de fichier du checkpoint porter la
tâche. Une lettre de taille désigne donc le même backbone et le même encodeur
pour les trois, et la prédiction, la validation et l'export prennent les mêmes
arguments quel que soit celui que vous chargez.

## Entraîner

Les trois tâches s'entraînent via `train()`, qui lit la tâche depuis le
checkpoint chargé et choisit l'entraîneur correspondant.

<code-tabs name="train" />

Ce qui a été vérifié pour la détection et la segmentation : la parité
d'inférence avec l'amont à 1e-5, couche par couche et par taille, et le fait que
la loss et un unique pas d'entraînement s'exécutent sur une entrée synthétique.
Ce qui ne l'a pas été, d'après la docstring de `train()` elle-même : la
convergence d'un fine-tuning complet, l'entraînement multi-GPU, l'étape d'arrêt
de l'augmentation avec rechargement du meilleur modèle, et le remap des classes
d'Objects365 vers COCO. La voie pose suit la recette publiée de DETRPose, un
matcher hongrois sur des coûts de classe, de L1 sur les points clés et d'OKS,
avec débruitage contrastif des points clés, et sa convergence n'a pas non plus
été vérifiée de bout en bout.

Laissé à ses réglages par défaut, l'entraîneur fait 74 époques à `lr0=5e-4` avec
la précision mixte activée, en suivant la recette amont : AdamW, un schedule
cosinus plat, une EMA à 0.9999 et des entrées normalisées ImageNet. La pose et
la segmentation exigent toutes deux `imgsz` à la taille native du checkpoint,
car leur grille d'ancres d'évaluation est construite à la construction du
modèle ; une autre valeur lève une erreur avant le démarrage du run. La pose
demande en plus un dataset à une seule classe dont le `data.yaml` déclare
`kpt_shape`, avec un nombre de points clés correspondant à la tête.

`lora=True` ne s'applique qu'à la détection ; la pose et la segmentation lèvent
une `ValueError` dessus. Sur Apple silicon, l'entraîneur garde le run sur le GPU
et envoie une seule opération sur CPU, la passe arrière du grid-sample à
l'intérieur de l'attention déformable, que PyTorch n'implémente pas en Metal.

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire indexé par nom de métrique, et affiche les
résultats par classe tant que `verbose` reste actif.

<code-tabs name="val" />

La pose reporte les métriques OKS des points clés sous `metrics/keypoints_*`. La
segmentation reporte les masques sous la clé `metrics/mAP50-95` sans suffixe et
reprend les deux vues en une seule passe, les boîtes sous `(B)` et les masques
sous `(M)`.

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. La pose et la segmentation exportent
avec une entrée fixe de 640 par 640 plutôt qu'en formes dynamiques, et plusieurs
cibles de détection sont elles aussi à canevas fixe, dont OpenVINO, Paddle, MNN,
ExecuTorch et Core AI. [L'export](/docs/export) liste les arguments que chaque
format accepte, ainsi que les extras que quelques-uns ajoutent.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
