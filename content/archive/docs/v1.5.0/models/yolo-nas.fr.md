---
title: YOLO-NAS
families:
  - yolonas
seo_title: "YOLO-NAS\_: prédire, entraîner et exporter dans LibreYOLO"
description: "Utilisez YOLO-NAS dans LibreYOLO pour la détection et la pose. Les poids de Deci.AI sont propriétaires et réservés à un usage non commercial\_; LibreYOLO n'en publie aucun."
lead: >-
  YOLO-NAS est un détecteur convolutionnel dont le backbone et le neck sont
  issus de la recherche d'architecture de Deci.AI, construit à partir de blocs
  RepVGG adaptés à la quantification. Ses poids appartiennent à Deci.AI, leur
  licence les réserve à un usage non commercial et LibreYOLO n'en publie aucun.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - détection d'objets
  - estimation de pose
  - détecteur compatible quantification
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Un nom qui n'est pas déjà présent sur le disque est récupéré depuis le
        CDN de Deci.

        # Le téléchargement affiche d'abord les conditions de licence de Deci ;
        récupérer le fichier les accepte.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Le suffixe -pose sélectionne la tête de pose et son propre ensemble de
        poids.

        model = LibreYOLO("LibreYOLONASs-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: À partir de zéro
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Aucun checkpoint Deci n'est utilisé : le modèle part de poids
        aléatoires,

        # le résultat de l'exécution dérive donc uniquement de vos données.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Sur COCO
      language: bash
      code: >
        # Le YAML COCO fourni contient un script de téléchargement intégré, il
        nécessite donc

        # une autorisation explicite sauf si le dataset est déjà présent
        localement.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Installer

YOLO-NAS ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Un nom de checkpoint qui n'est pas déjà présent sur le disque est récupéré
depuis le CDN public de Deci, et non depuis l'organisation LibreYOLO, qui
n'héberge aucun de ces poids. Avant le début du transfert, la bibliothèque
affiche les conditions de licence de Deci une fois par processus. Avant
l'ouverture du fichier téléchargé, son SHA-256 est comparé à une valeur
épinglée. Les droits accordés par ces conditions sont décrits dans la section
[licence](#licensing).

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. `conf` définit le seuil de
confiance et `iou` le seuil de NMS. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

La détection et la pose utilisent la même architecture avec des têtes
différentes, et acceptent les mêmes arguments. Les tailles du tableau
ci-dessous concernent la détection\u00a0; la pose est publiée dans ces tailles ainsi
que dans une taille plus petite. La tête de pose prédit l'ensemble de points
clés COCO.

<benchmark-table task="detect" />

<va-embed />

## Entraîner

<code-tabs name="train" />

Lorsque vous les omettez, `epochs`, `lr0` et `amp` sont résolus pour chaque
tâche. Une exécution de pose utilise donc des valeurs par défaut différentes
d'une exécution de détection. L'optimiseur par défaut est AdamW. Le nombre de
classes provient du fichier YAML du dataset et la tête est reconstruite en
conséquence avant la première époque. Sur la tête de pose, le nombre de points
clés est géré de la même manière, ce qui vous permet d'effectuer le fine-tuning
d'un checkpoint de pose COCO sur un squelette de taille différente.

Le fine-tuning part des poids de Deci, qui sont couverts par la licence de
Deci. L'entraînement d'un modèle initialisé aléatoirement n'emploie aucun
checkpoint Deci\u00a0; il correspond au troisième extrait ci-dessus.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. Vous pouvez également exécuter le graphe dans
un runtime nu, sans installer LibreYOLO, mais vous devez alors écrire vous-même
le prétraitement et le post-traitement. Chaque format installe un extra
différent et accepte quelques arguments qui lui sont propres. Ces deux aspects
sont décrits sur la page du format concerné.

Un export est une autre copie des mêmes poids dans un conteneur différent.
L'export d'un checkpoint Deci ne modifie ni l'origine des poids ni la licence
qui les couvre.

<code-tabs name="export" />

## Checkpoints

Il n'y en a aucun à énumérer. La licence de Deci interdit la redistribution,
l'organisation LibreYOLO ne publie donc aucun poids YOLO-NAS et le
téléchargement est résolu ailleurs\u00a0: un nom de la forme
`LibreYOLONAS<size>.pt`, ou `LibreYOLONAS<size>-pose.pt` pour la pose, est
associé à l'objet correspondant sur le CDN public de Deci.

Seuls les checkpoints dont la bibliothèque épingle le SHA-256 peuvent être
récupérés de cette manière. Toute autre tentative échoue de façon sécurisée
au lieu d'ouvrir un pickle tiers non vérifié\u00a0; vous devez télécharger le
fichier manuellement et transmettre son chemin. Un fichier déjà présent sur
le disque est chargé depuis son chemin, sans téléchargement ni contrôle par
somme de vérification. Cela inclut un fichier Deci `.pth` sous son nom
d'origine, que le chargeur reconnaît.

## Licence

<provenance-box>

LibreYOLO n'héberge ni ne réplique ces poids\u00a0: cette famille n'a aucun fichier
dans l'organisation LibreYOLO sur Hugging Face. Chaque téléchargement
automatique pointe plutôt vers le CDN public de Deci, affiche les conditions
de Deci une fois par processus avant de démarrer et vérifie le fichier à
l'aide d'un SHA-256 épinglé avant de l'ouvrir.

L'autre solution consiste à entraîner un modèle initialisé aléatoirement.
L'architecture est sous licence Apache-2.0 upstream et MIT ici, un modèle
ainsi entraîné sur vos propres données ne dérive donc d'aucun checkpoint Deci.

</provenance-box>

## Citation

YOLO-NAS a été publié sans article. L'entrée ci-dessous est celle demandée par
ses auteurs et couvre SuperGradients, la bibliothèque dans laquelle il a été
publié.

<citation-block />

