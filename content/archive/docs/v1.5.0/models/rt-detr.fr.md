---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 et RT-DETRv4 dans LibreYOLO'
description: >-
  Utilisez RT-DETR, RT-DETRv2 et RT-DETRv4 dans LibreYOLO pour la détection
  d'objets, avec en plus les boîtes orientées sur RT-DETRv2. Installez,
  prédisez, entraînez, validez et exportez avec des poids Apache-2.0.
lead: >-
  Un detection transformer conçu pour l'inférence en temps réel : il décode un
  ensemble fixe de requêtes plutôt qu'une grille dense et n'exécute donc aucune
  NMS. LibreYOLO en propose trois versions, distinguées par le checkpoint
  chargé, et la version 2 prend aussi en charge les boîtes orientées.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - detection transformer temps réel
  - DETR
  - détection d'objets
  - détection de boîtes orientées
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vidéo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La version figure dans le nom du fichier et la factory utilise le
        # checkpoint, les trois versions se chargent donc de la même façon.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # index de webcam, flux RTSP ou liste .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Boîtes orientées
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Version 2 uniquement. Le suffixe -obb sélectionne la tâche et les

        # propres tenseurs du checkpoint indiquent son orientation, aucun
        argument

        # task n'est requis. Ces poids DOTA v1.0 couvrent 15 classes à 1024 px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5) : cx, cy, w, h, radians

        print(obb.xyxyxyxy)  # mêmes lignes sous forme de quatre points d'angle

        print(result.boxes.xyxy)  # boîtes alignées sur les axes qui les
        englobent
    - label: 'Boîtes orientées, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # coco128.yaml télécharge 128 images au premier usage. Pour une
        véritable

        # exécution, faites pointer `data` vers votre propre YAML de dataset.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Nécessite l'extra lora : pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Comparer à COCO
      language: bash
      code: >
        # coco-val-only.yaml récupère les 5000 images val2017 et ignore le
        dataset

        # d'entraînement. Son script de téléchargement intégré nécessite une

        # autorisation explicite, sauf si le dataset est déjà local.

        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Boîtes orientées
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La validation orientée utilise l'IoU pivotée : une prédiction placée
        au

        # bon endroit avec le mauvais angle est donc comptée comme manquée.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        metrics = model.val(data="my-obb-dataset.yaml")


        print(metrics["metrics/mAP50-95(OBB)"])

        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Nécessite l'extra onnx : pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Boîtes orientées
      language: bash
      code: >
        # ONNX et TorchScript sont les cibles validées pour la tâche orientée,

        # en FP32, batch 1, sur un canevas fixe de 1024 par 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreRTDETRr18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Installer

RT-DETR ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base, et l'extra `rtdetr` est un nom stable qui n'y ajoute
rien.

```bash
pip install libreyolo
```

Le fine-tuning par adaptateur avec `lora=True` constitue l'exception et
nécessite l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
détecteur ne demande donc de modifier qu'une ligne. `conf` et `max_det` filtrent
un décodage top-k sur les requêtes et les classes ; aucune étape NMS n'est à
régler, et `iou` est accepté mais inutilisé. Un checkpoint orienté remplit
nativement `result.obb` ainsi que `result.boxes` avec les rectangles alignés sur
les axes qui englobent les boîtes. Consultez la [prédiction](/docs/predict) pour
les sources, le streaming et le traitement des résultats.

## Variantes

Trois versions se partagent deux tâches, et leurs codes de taille ne forment pas
une série unique. La version 1 nomme ses tailles d'après le backbone, ResNet ou
HGNetv2. La version 2 réutilise uniquement les noms ResNet : la version 1 fournit
déjà les deux tailles HGNetv2 et les résultats de la version 2 étaient assez
proches pour que LibreYOLO ne publie pas de poids en double. La version 4 utilise
une série de lettres simples qui entre en collision avec les noms HGNetv2 de la
version 1. Un code de taille seul n'identifie donc pas un modèle. La version est
inscrite dans le nom du fichier de checkpoint.

<benchmark-table task="detect" />

<va-embed />

La version 2 conserve l'architecture et la disposition du dictionnaire d'état
de la version 1, mais change la façon dont l'attention déformable échantillonne.
C'est pourquoi les métadonnées du checkpoint permettent de les distinguer, et
non sa forme. La version 4 appartient à une autre lignée : elle réutilise
l'architecture et l'entraîneur de D-FINE, et ses poids proviennent de la
distillation d'un modèle de fondation visuel DINOv3 enseignant vers un élève
HGNetv2. Dans LibreYOLO, `LibreRTDETRv4` est une sous-classe de `LibreDFINE` dont
la tête de masque est désactivée de manière fixe, elle reste donc réservée à la
détection.

### Boîtes orientées sur la version 2

La version 2 est la seule à proposer une seconde tâche. Ses tâches prises en
charge sont `detect` et `obb`, et elles ne partagent ni graphe ni série de
tailles. La détection utilise les tailles ResNet à 640 px ; la détection orientée
utilise une série HGNetv2, n, s, m, l et x, à 1024 px, et la taille d'entrée est
résolue par tâche plutôt que par famille. Les propres tenseurs d'un checkpoint
permettent de reconnaître son orientation, grâce aux têtes de boîte à cinq
coordonnées et aux paramètres d'échantillonnage de la version 2. Les poids `-obb`
se chargent donc dans le graphe orienté sans argument `task`, et une
incompatibilité entre les deux produit une erreur franche au lieu d'une
réinterprétation silencieuse.

Les fichiers publiés vont de `LibreRTDETRv2n-obb.pt` à
`LibreRTDETRv2x-obb.pt`. Ce sont les checkpoints officiels DOTA v1.0 à une seule
échelle, convertis au format LibreYOLO. Ils couvrent 15 classes aériennes, de
l'avion et du navire au port et à l'hélicoptère, et leurs noms de classes sont
inscrits dans le checkpoint. Contrairement à la détection, la tâche orientée est
réservée à l'inférence : la prédiction, la validation et l'export fonctionnent,
mais `train()` lève une erreur sur un modèle orienté. Le suivi et l'augmentation
au moment du test ne prennent pas non plus en charge les boîtes orientées. La
page [Détection orientée](/docs/tasks/oriented-detection) présente la tâche, le
format des étiquettes et les métriques.

## Entraîner

L'entraînement part d'un checkpoint publié. `pretrained` est accepté, puis
ignoré sur les trois versions. `pretrained=False` ne produit donc pas un modèle
initialisé aléatoirement. Toute cette section concerne la détection : la tâche
orientée de la version 2 est réservée à l'inférence et il n'existe aucun chemin
de transfert depuis les poids de détection, car les deux utilisent des backbones
différents.

<code-tabs name="train" />

Le learning rate est l'argument à régler correctement, et chaque version possède
sa propre valeur par défaut au lieu de celle de toute la bibliothèque. La
signature Python de `train()` la lit dans la configuration d'entraînement de la
version, et la CLI résout la même valeur lorsque `lr0` n'est pas fourni. Les
versions 1 et 2 acceptent également `lr_backbone`, dont la valeur par défaut est
un vingtième de `lr0`, conformément à la recette originale. La version 4 passe
par l'entraîneur D-FINE, qui met à l'échelle le groupe de paramètres du backbone
avec `backbone_lr_mult`.

Laissez `imgsz` à la taille native du checkpoint sauf si vous avez une raison de
la modifier. La validation et la prédiction fonctionnent à d'autres tailles,
avec un effet résiduel : une taille rectangulaire dont le nombre de tokens
correspond à celui de la taille native réutilise tout de même un embedding conçu
pour le mauvais rapport de forme.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset dans le format
utilisé pour l'entraînement.

<code-tabs name="val" />

Les lignes du tableau de benchmark ci-dessus proviennent de l'outil de benchmark
LibreYOLO ; la note sous ce tableau indique le dataset qui les a produites et
fournit les liens vers les enregistrements des exécutions.

La validation orientée passe par le même appel et rapporte les mêmes clés, plus
quatre répétées avec un suffixe `(OBB)`. La correspondance utilise l'IoU pivotée
plutôt que celle des rectangles englobants. Une erreur d'angle compte donc comme
un échec. `augment=True` est refusé pour cette tâche.

## Exporter

<export-matrix />

La matrice présente la lignée sur une seule page : lorsque les trois versions ne
s'accordent pas sur un format, la cellule indique la prise en charge la plus
faible des trois, afin de ne rien exagérer quelle que soit la version chargée.
La ligne orientée concerne uniquement la version 2. ONNX et TorchScript y sont
validés en FP32, avec un batch de 1 et un canevas fixe de 1024 par 1024 ;
OpenVINO, TensorRT et ExecuTorch se convertissent et se rechargent, mais n'ont pas
atteint la parité des sorties brutes sur l'ensemble complet de requêtes. Les
meilleures boîtes concordent à une fraction de pixel près, tandis que la fin de
la distribution dérive.

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

Le nom de fichier contient la version, puis la taille, puis la tâche. Les poids
de détection sont `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` et
`LibreRTDETRv4<size>.pt`, tous à 640 px. Les poids orientés existent uniquement
pour la version 2 et ajoutent le suffixe de tâche, de
`LibreRTDETRv2n-obb.pt` à `LibreRTDETRv2x-obb.pt`. Ils utilisent tous 1024 px et
sont entraînés sur DOTA v1.0 plutôt que COCO.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />

Le bloc ci-dessus est celui que les auteurs publient pour la détection avec les
versions 1 et 2. Les poids orientés de la version 2 possèdent une troisième
source amont, le dépôt RiO-DETR Apache-2.0 à l'adresse
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), d'où
proviennent les checkpoints DOTA ; citez ce projet si vous en avez utilisé un.
La version 4 fait l'objet d'un article distinct par un autre groupe et possède
son propre bloc de citation à l'adresse
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation) ;
citez celui-ci si vous avez utilisé un checkpoint de version 4.
