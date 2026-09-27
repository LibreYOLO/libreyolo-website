---
title: Classification d'images
seo_title: Classification d'images dans LibreYOLO
description: >-
  Étiqueter une image entière dans LibreYOLO : familles qui couvrent la tâche,
  structure de dataset ImageFolder et appels de prédiction, d'entraînement, de
  validation et d'exportation.
lead: >-
  La classification d'images attribue une distribution d'étiquettes à une image
  entière et ne localise rien à l'intérieur. La clé de tâche est classify.
keywords:
  - classification images python
  - entraîner classificateur images
  - dataset ImageFolder
  - précision top 1
  - classification zero shot
  - bibliothèque classification MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -cls du nom de fichier sélectionne la tâche.
        # Aucun argument task n'est donc nécessaire.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Distribution complète
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)

        probs = result.probs


        # .data est le vecteur (C,) complet ; top5/top5conf sont des vues
        ordonnées.

        print(probs.data.shape)

        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, sans entraînement'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP compare l'image à des prompts textuels. L'ensemble d'étiquettes

        # est donc défini à l'appel plutôt qu'intégré au checkpoint.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # imagenette160 est un nom de dataset connu, téléchargé à la première

        # utilisation. Pour vos données, transmettez un répertoire avec une
        partition train/.

        model = LibreYOLO("LibreResNet50-cls.pt")

        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Définition

La classification d'images produit un score par classe pour l'image entière et
aucune coordonnée. Elle répond à la question de ce qui se trouve dans l'image,
jamais où. C'est ce qui la distingue de la
[détection d'objets](/docs/tasks/object-detection).

`classify` est la clé de tâche canonique, et le suffixe `-cls` du nom d'un
checkpoint la sélectionne. Ce suffixe est obligatoire, et non facultatif, pour
les familles de classification. `LibreResNet50.pt` n'est donc pas interprété
comme un classificateur. Seul `LibreResNet50-cls.pt` l'est.

`predict()` remplit `result.probs` et laisse `boxes` vide. `.data` est le
vecteur complet des scores, `.top1` l'indice du score le plus élevé et
`.top1conf` sa valeur. `.top5` contient les cinq indices les plus élevés dans
l'ordre décroissant et `.top5conf` leurs scores. Ces indices pointent vers
`result.names`. Découper un objet `Results` ne tronque jamais `probs`, car le
vecteur appartient à l'image et non à une ligne.

## Modèles

Cinq familles peuvent être entraînées et effectuer des prédictions :
[ResNet](/docs/models/resnet), [ConvNeXt](/docs/models/convnext),
[MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) et
[DINOv2](/docs/models/dinov2). Les quatre premières s'exécutent avec le paquet
de base et proposent des poids publiés. DINOv2 nécessite
`pip install "libreyolo[rfdetr]"` et ne possède aucun checkpoint hébergé par
LibreYOLO. Il charge le backbone amont avec une tête linéaire initialisée
aléatoirement. Il constitue donc un point de départ pour le fine-tuning plutôt
qu'un prédicteur prêt à l'emploi.

Cinq autres familles effectuent la prédiction, la validation et l'exportation,
mais leur méthode `train()` déclenche une `NotImplementedError` :
[ViT](/docs/models/vit), [Swin](/docs/models/swin), [VGG](/docs/models/vgg),
[AlexNet](/docs/models/alexnet) et [DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) et [SigLIP2](/docs/models/siglip2) classent sans
ensemble fixe d'étiquettes. Ils comparent l'image à des prompts textuels.
`set_classes()` définit donc les classes au moment de l'appel, sans aucune étape
d'entraînement pour un nouvel ensemble d'étiquettes. Tous deux couvrent
également la tâche `embed`.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

`conf`, `iou` et `max_det` n'ont aucun effet ici. Il n'existe aucun candidat à
filtrer par seuil ou à supprimer, seulement une distribution. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

La classification emploie une arborescence de répertoires, pas des fichiers
d'étiquettes ni un fichier YAML. `data` désigne la racine du dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` est obligatoire pour l'entraînement et définit l'association entre
classes et indices selon l'ordre des noms de dossiers. Le premier dossier par
ordre alphabétique devient donc la classe 0. `val/` est obligatoire pour la
validation. Une partition `test/` peut être présente, mais les commandes
d'entraînement et de validation par défaut ne l'utilisent pas. Toute partition
autre que `train` doit contenir les mêmes noms de dossiers de classes que
l'ensemble attendu. Une incompatibilité échoue ainsi explicitement au lieu
d'être évaluée comme une mauvaise prédiction. Les extensions d'images acceptées
sont `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` et `.tiff`.

`data` accepte trois formes : le chemin d'un répertoire contenant une partition
`train/`, l'URL d'un fichier `.zip` ou l'un des noms de datasets connus,
`imagenette160` et `smoke10`, téléchargés et mis en cache à la première
utilisation.

Le chargeur canonique est `libreyolo.data.classify_dataset`.

## Entraîner

<code-tabs name="train" />

Aucun `nc` n'est à déclarer. Le nombre de classes provient des noms de dossiers
sous `train/`, et la couche linéaire finale est reconstruite pour lui
correspondre tandis que le backbone est transféré sans modification. Consultez
la page [entraînement](/docs/train) pour les datasets, les augmentations, le
multi-GPU et les systèmes de journalisation.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`, calculées sur la
partition `val/` de la racine du dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` est la proportion d'images dont la classe au score le
plus élevé correspond à la vérité terrain. C'est le nombre principal, utilisé
par l'entraînement pour choisir la meilleure époque.
`metrics/accuracy_top5` est la proportion d'images dont la vraie classe figure
parmi les cinq scores les plus élevés. Cette valeur est d'autant moins
significative que le dataset contient peu de classes. Le dictionnaire contient
aussi `fitness`, une copie de la valeur top-1.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie le même `Results`. La couverture des formats varie selon la famille.
La matrice de chaque page de modèle est générée depuis l'ensemble validé plutôt
que saisie manuellement. Consultez la page
[exporter et déployer](/docs/export) pour les formats, leurs extras et leurs
contraintes.
