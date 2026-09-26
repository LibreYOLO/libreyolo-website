---
title: DEIM
families:
  - deim
seo_title: DEIM et DEIMv2 dans LibreYOLO
description: >-
  Utilisez DEIM et DEIMv2 dans LibreYOLO pour la détection d'objets. Installez,
  prédisez, entraînez, validez et exportez, à partir d'une taille d'un
  demi-million de paramètres.
lead: >-
  Un transformer de détection entraîné par appariement dense un-à-un, qui
  converge en bien moins d'époques que les recettes DETR dont il hérite.
  LibreYOLO en embarque deux versions, distinguées par le checkpoint que vous
  chargez.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - transformer de détection
  - DETR
  - détection d'objets temps réel
  - détection d'objets python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vidéo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La version fait partie du nom de fichier, et la factory route sur
        # le checkpoint, donc les deux se chargent de la même façon.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # index de webcam, flux RTSP ou liste .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml télécharge un échantillon de 128 images au premier
        # usage. Pointez `data` vers votre propre YAML de dataset pour un
        # vrai run.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Laissés non renseignés, epochs, batch, imgsz et lr0 viennent de la
        # recette publiée pour la taille chargée.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Demande l'extra lora : pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Sur COCO
      language: bash
      code: |
        # coco-val-only.yaml récupère les 5000 images de val2017 et ignore le
        # set d'entraînement. Il embarque un script de téléchargement, il faut
        # donc une autorisation explicite si le dataset n'est pas déjà local.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Demande l'extra onnx : pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory route selon l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Installation

Aucune des deux versions ne demande d'extra optionnel. Tout ce qu'elles
importent fait partie de l'installation de base.

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

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
remplacer le détecteur par un autre tient en une seule ligne. `conf` et
`max_det` filtrent un décodage top-k sur les queries et les classes ; il n'y a
pas d'étape de NMS à régler, et `iou` est accepté mais inutilisé. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

La version 1 propose cinq tailles, toutes à la même taille d'entrée. La
version 2 conserve ces cinq noms et en ajoute trois plus petits, `atto`,
`femto` et `pico`, dont les deux premiers sont natifs à une taille d'entrée
plus basse que les autres. Cinq codes de taille existent donc dans les deux
versions et désignent des modèles différents ; la version est inscrite dans le
nom de fichier du checkpoint.

<benchmark-table task="detect" />

<va-embed />

La version 1 conserve l'architecture de D-FINE et remplace son objectif de
classification par la fonction de perte sensible à l'appariabilité issue de la
recette dense un-à-un, si bien que les deux familles partagent presque toutes
les clés du state dict et se distinguent par les métadonnées du checkpoint. La
version 2 conserve ce contrat d'entraînement et mélange les backbones :
HGNetv2 en dessous de `s`, et un vision transformer DINOv3 avec un adaptateur
d'ajustement spatial à partir de `s`. C'est ce backbone qui ajoute une seconde
licence à ces quatre checkpoints, alors lisez [la licence](#licensing) avant
d'en mettre un en production.

## Entraîner

L'entraînement démarre depuis un checkpoint publié. `pretrained` n'atteint
jamais l'entraîneur : la version 1 avertit que la clé est inconnue et l'ignore,
la version 2 la supprime. Aucune des deux ne vous donne un modèle initialisé
aléatoirement.

<code-tabs name="train" />

Passez `lr0` vous-même sur la version 1. Sa signature Python `train()` a pour
défaut `4e-4`, le taux de la recette COCO publiée, alors que la configuration
d'entraînement de la famille porte `1e-4` comme défaut de fine-tuning, et c'est
cette valeur plus basse que la CLI résout quand l'argument est absent. La
configuration consigne la mesure qui la justifie : aux tailles de batch qu'un
fine-tuning utilise réellement, sur de petits datasets, le taux COCO dégradait
le transfert de façon mesurable.

La version 2 résout ces défauts elle-même. Laisser `epochs`, `batch`, `imgsz`
et `lr0` non renseignés lui fait lire chacun d'eux dans la recette publiée pour
la taille chargée, si bien que les petites tailles s'entraînent à leur propre
résolution d'entrée sans qu'on le leur dise, et une valeur que vous passez
l'emporte sur la recette. `imgsz` est l'argument qu'elle contraint : il doit
être un multiple positif de 32, sinon la version 2 lève une erreur avant le
démarrage du run.

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset au
format sur lequel vous avez entraîné.

<code-tabs name="val" />

Les lignes de la table de benchmark ci-dessus proviennent du harnais de
benchmark LibreYOLO ; la note sous cette table indique quel dataset les a
produites et renvoie vers les enregistrements de run.

## Exporter

<export-matrix />

La matrice couvre les deux versions sur une seule page : là où elles divergent
sur un format, la cellule affiche la plus faible des deux, si bien que rien
n'est survendu ici, quelle que soit la version que vous chargez.

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>
Les quatre tailles DEIMv2 à partir de S tirent leur backbone de DINOv3, si bien
que leurs dépôts de poids portent à la fois Apache-2.0 et la DINOv3 License de
Meta, et LibreYOLO distribue le code source du backbone DINOv3 sous ce même
accord. Le reste de cette famille, y compris toutes les tailles DEIMv2 en
dessous de S, est sous Apache-2.0 seule.
</provenance-box>

## Citation

<citation-block />

DEIMv2 est un article distinct et possède son propre bloc de citation sur
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation) ;
citez celui-là si vous avez utilisé un checkpoint de version 2.
