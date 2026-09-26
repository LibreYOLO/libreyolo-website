---
title: Segmentation sémantique
seo_title: Segmentation sémantique dans LibreYOLO
description: >-
  Attribuer une classe à chaque pixel dans LibreYOLO : familles qui couvrent la
  tâche, format des masques denses et appels de prédiction, d'entraînement, de
  validation et d'exportation.
lead: >-
  La segmentation sémantique attribue une classe à chaque pixel d'une image sans
  distinguer les instances d'une même classe. La clé de tâche est semantic.
keywords:
  - segmentation sémantique python
  - classification pixels
  - prédiction dense
  - entraîner modèle segmentation
  - mIoU
  - bibliothèque segmentation MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Le suffixe -sem du nom de fichier sélectionne la tâche.

        # Aucun argument task n'est donc nécessaire.

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # identifiants de classes (H, W) sur le canevas
        d'origine

        print(mask.classes)      # identifiants présents triés, sans 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Une classe à la fois
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # booléen (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Une autre famille, le même appel'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Sur ADE20K
      language: bash
      code: >
        # ade20k.yaml contient un script de téléchargement intégré pour
        l'archive

        # d'environ 1 Go. Il nécessite une autorisation explicite si les données
        ne sont pas locales.

        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Définition

La segmentation sémantique étiquette des pixels et non des objets. Chaque pixel
reçoit un identifiant de classe. Deux voitures qui se touchent dans l'image
deviennent une même région de la classe voiture, sans limite entre elles. La
[segmentation d'instances](/docs/tasks/instance-segmentation) sert à compter les
instances. La [segmentation panoptique](/docs/tasks/panoptic-segmentation)
étiquette chaque pixel tout en séparant les instances.

`semantic` est la clé de tâche canonique, et le suffixe `-sem` du nom d'un
checkpoint la sélectionne. L'argument `task=` est donc inutile lors du
chargement des poids publiés.

`predict()` remplit `result.semantic_mask`. `.data` est une carte entière de
classes `(H, W)` sur le canevas de l'image d'origine, `.classes` répertorie les
identifiants présents dans l'ordre croissant et `.class_mask(id)` renvoie la
sélection booléenne `(H, W)` d'une classe. La valeur `255` est l'étiquette à
ignorer. Elle ne constitue jamais une classe, est exclue de la perte et des
métriques, et n'apparaît pas dans `.classes`.

## Modèles

Trois familles peuvent être entraînées et effectuer des prédictions :
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) et
[DINOv2](/docs/models/dinov2). SegFormer et LingBot-Vision s'exécutent avec le
paquet de base et proposent des poids publiés. DINOv2 nécessite
`pip install "libreyolo[rfdetr]"` et ne possède aucun checkpoint hébergé par
LibreYOLO. Il charge le backbone amont, tandis que sa tête dense reçoit une
initialisation aléatoire. Il constitue donc un point de départ pour
l'entraînement plutôt qu'un prédicteur prêt à l'emploi.

Quatre autres familles effectuent la prédiction, la validation et l'exportation,
mais leur méthode `train()` déclenche une `NotImplementedError` :
[FCN](/docs/models/fcn), [DeepLabv3](/docs/models/deeplabv3),
[PIDNet](/docs/models/pidnet) et [EoMT](/docs/models/eomt).

Les ensembles de classes diffèrent selon le checkpoint, et non selon la
famille. Les poids publiés proviennent de datasets dont les espaces
d'étiquettes ont peu en commun, notamment les 150 classes d'ADE20K contre les
19 de Cityscapes. Le champ `names` d'un checkpoint indique donc ce qu'il peut
étiqueter, et deux checkpoints ne sont comparables que s'ils ont été entraînés
sur le même dataset.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

La carte est un argmax par pixel. Il n'existe donc aucune étape de NMS et
`iou` n'a jamais d'effet. `conf` et `max_det` sont acceptés pour préserver la
parité de l'API et ne font rien sur SegFormer, PIDNet et les autres prédicteurs
denses. EoMT constitue l'exception, car `conf` y filtre la sélection des
requêtes. Consultez la page [prédiction](/docs/predict) pour les sources, le
streaming et la gestion des résultats.

## Format du dataset

Chaque image est associée à un masque dense monocanal plutôt qu'à un fichier
d'étiquettes `.txt`. Le masque est trouvé en remplaçant `images` par le
répertoire des masques dans le chemin de l'image.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Les masques sont des images monocanales sans perte, généralement des PNG. Les
PNG en mode palette sont lus comme des indices de palette. Chaque pixel contient
un identifiant de classe dans `0..nc-1`, la valeur `255` signifie ignorer, et la
résolution du masque doit être identique à celle de l'image associée.

Le fichier YAML ajoute deux clés au contrat partagé :

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` est le nom du répertoire substitué à `images`. Il vaut `masks` par
défaut. `label_mapping` est une association facultative
`{source_id: train_id}` appliquée aux valeurs de pixels du masque au chargement.
Elle permet par exemple de faire passer un dataset numéroté de 1 à 150 à une
plage de 0 à 149. Toute valeur source sans correspondance devient une valeur à
ignorer, et chaque identifiant d'entraînement doit se trouver dans `0..nc-1`.

Omettre `masks_dir` fait basculer le chargeur vers une solution de repli. Les
masques sont alors rastérisés au chargement depuis les annotations polygonales
résolues selon la convention habituelle `images` vers `labels`. Une classe
`background` est ajoutée après les classes d'objets, ce qui augmente `nc` d'une
unité.

Le chargeur canonique est `libreyolo.data.SemanticDataset`.

## Entraîner

<code-tabs name="train" />

`imgsz` est ici soumis à une contrainte absente des détecteurs. Chaque famille
déclare un diviseur dont son entrée doit être un multiple, défini par sa grille
de patches ou son pas de sortie. L'entraînement comme la validation déclenchent
une `ValueError` avant le début de l'exécution si `imgsz` n'est pas divisible.
Ce diviseur vaut 32 pour SegFormer, 16 pour LingBot-Vision et EoMT, 14 pour
DINOv2, et 8 pour FCN et PIDNet. Consultez la page
[entraînement](/docs/train) pour les datasets, les augmentations, le multi-GPU
et les systèmes de journalisation.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`, calculées sur la
partition `val` nommée dans le fichier YAML du dataset.

<code-tabs name="val" />

`metrics/mIoU` est l'intersection sur l'union moyenne. Pour chaque classe, le
chevauchement entre les pixels prédits et réels est divisé par leur union, puis
le résultat est moyenné sur les classes. C'est le nombre principal et celui qui
sert à choisir la meilleure époque pendant l'entraînement.
`metrics/pixel_accuracy` est la proportion de pixels qui reçoivent la bonne
classe. Une grande classe d'arrière-plan peut la gonfler. La mIoU reste donc la
valeur à comparer. Les pixels marqués `255` ne contribuent à aucune des deux.
Le dictionnaire contient aussi `fitness`, une copie de la mIoU.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie le même `Results`. La couverture des formats varie selon la famille.
La matrice de chaque page de modèle est générée depuis l'ensemble validé plutôt
que saisie manuellement. Consultez la page
[exporter et déployer](/docs/export) pour les formats, leurs extras et leurs
contraintes.
