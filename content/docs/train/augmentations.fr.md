---
title: Augmentations
seo_title: 'Augmentations d''entraînement dans LibreYOLO, avec exemples'
description: >-
  Ce que fait chaque réglage d'augmentation de LibreYOLO, montré sur une image
  réelle par le propre code d'entraînement de la bibliothèque, avec sa plage, sa
  valeur par défaut et les familles qui le lisent.
lead: >-
  Les augmentations d'entraînement sont des transformations aléatoires
  appliquées à chaque échantillon d'entraînement. Elles se configurent avec des
  réglages de TrainConfig, et chaque famille de modèles exécute son propre
  pipeline, si bien qu'un réglage ne modifie les échantillons que là où ce
  pipeline le lit.
keywords:
  - augmentation de données yolo
  - augmentation mosaïque
  - mixup
  - augmentation hsv
  - transformation affine aléatoire
  - random erasing
  - détection de petits objets
  - exemples d'augmentation
  - augmentation copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # La CLI nomme mosaic_prob mosaic et mixup_prob mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX part de mosaic_scale=(0.1, 2.0), qui peut réduire un objet
        # au dixième de sa taille. (0.5, 1.5) s'arrête à la moitié.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # En CLI, une paire s'écrit avec deux nombres séparés par une virgule.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: Lire la table de prise en charge d'une famille
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Uniquement les réglages ignorés
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Ensemble de classification
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 72d22a38d647de0a
---

## Définir les réglages

Les réglages d'augmentation sont des arguments ordinaires de `train()`.

<code-tabs name="train" />

Trois d'entre eux ont une forme plus courte, acceptée aussi bien par `train()` que par la CLI : `mosaic` correspond à `mosaic_prob`, `fliplr` à `flip_prob` et, en détection, `mixup` correspond à `mixup_prob`. Passer une forme courte et son champ avec des valeurs différentes lève une erreur. `perspective` n'a pas de flag CLI dans la version 1.8.0, il se définit donc depuis Python.

Les réglages de classification sont validés à la construction de leur pipeline, et la CLI vérifie aussi `flip_prob`, `flipud`, `cutmix` et `erasing`. `TrainConfig` ne vérifie pas la plage des réglages de détection, si bien que `degrees=100` entraîne avec des rotations allant jusqu'à 100 degrés. La [matrice d'augmentation](/docs/reference/augmentation-matrix) donne la plage de chaque réglage.

## Trois états, pas deux

Qu'un réglage ait un effet ou non dépend de la famille. La bibliothèque tient une table déclarative de ce comportement, et chaque entrée prend l'un des trois états.

`used` signifie que le réglage atteint le pipeline et modifie les échantillons. `ignored` signifie qu'il n'atteint jamais le pipeline, si bien que le définir n'a aucun effet. `gated_by_mosaic` signifie qu'il ne s'applique qu'aux échantillons passés par la branche mosaïque, si bien qu'avec `mosaic_prob=0` il ne se déclenche jamais, même s'il est bien branché.

C'est ce troisième état qui surprend. Dans un pipeline de style YOLOX, la transformation affine s'exécute sur le canevas de la mosaïque et MixUp mélange un échantillon mosaïque, si bien que `mosaic_prob=0` désactive silencieusement `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` et `mixup_scale`, tous à la fois. Le trainer consigne un avertissement propre au cas de MixUp :

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI signale aussi les réglages ignorés, en ne listant que ceux que vous avez réellement saisis :

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Appelé depuis Python, `train()` ne signale pas les réglages ignorés. La [matrice d'augmentation](/docs/reference/augmentation-matrix) donne l'état de chaque réglage pour chaque famille et la valeur dont part chaque famille. La table est la spécification distribuée dans `libreyolo/data/augment/spec.py`, que les propres tests de la bibliothèque confrontent aux pipelines, et elle peut être lue directement :

<code-tabs name="support" />

## Ce que fait chaque réglage

Les figures ci-dessous sont produites par les propres transformations d'entraînement de LibreYOLO 1.8.0, exécutées sur l'image d'exemple au moyen d'un dataset d'une seule image, si bien que chaque tuile de mosaïque et chaque partenaire de MixUp est la même photo. La boîte est la personne que LibreYOLO9t détecte sur cette image. Sous chaque panneau figurent la valeur du réglage et la taille de la plus grande boîte, en pixels, à la résolution d'entraînement de 640 pixels. Sans augmentation, la personne mesure 291 × 217 px.

Plusieurs réglages géométriques sont des plages : `degrees=10` désigne un angle tiré n'importe où entre -10 et +10 degrés. Leurs panneaux fixent le tirage à la valeur indiquée dans la légende, si bien qu'ils montrent l'effet le plus fort que permet une valeur plutôt qu'un échantillon typique. Les figures de détection proviennent du pipeline YOLOv9. YOLOv7 s'entraîne avec ce pipeline sans modification, et RT-DETR et RT-DETRv2 utilisent son wrapper mosaïque avec leur propre transformation par échantillon. YOLOX, RTMDet, PicoDet et FOMO construisent la mosaïque et la transformation de la même façon dans un wrapper distinct, dont le MixUp diffère.

### mosaic_prob

La probabilité qu'un échantillon soit une mosaïque : quatre images placées sur un canevas de deux fois la taille d'entrée, que la transformation affine ramène ensuite à la taille d'entrée. Dans les pipelines de style YOLOX, la transformation et MixUp ne s'exécutent que sur les échantillons mosaïque, si bien que les deux s'arrêtent lorsque cette valeur vaut 0. RT-DETR, RT-DETRv2, PicoDet et FOMO la fixent à 0 par défaut.

<aug-figure name="mosaic" />

Le deuxième panneau fixe l'échelle de la transformation à 0.5×, où le canevas entier tient dans le cadre. Les deux derniers sont des tirages ordinaires avec les valeurs par défaut de YOLOv9, et la légende donne l'échelle tirée pour chacun.

### mosaic_scale

La plage dans laquelle est tiré le zoom de la transformation. À 1×, un objet garde la taille qu'il a à l'inférence, et à 0.5×, il fait la moitié de cette taille. La valeur par défaut de base est `(0.1, 2.0)`, que conservent YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2 et FOMO, si bien que leurs objets peuvent être réduits au dixième de leur taille. YOLOv9 utilise `(0.5, 1.5)` par défaut. YOLO-NAS et PP-YOLOE n'ont pas de mosaïque et réutilisent cette plage comme échelle de leur transformation affine par échantillon.

<aug-figure name="mosaic_scale" />

Au-delà de 1×, la sortie ne montre que la partie supérieure gauche du canevas, si bien que les objets sont coupés au bord ou sortent du cadre.

### degrees

La plage de rotation : chaque échantillon transformé tourne d'un angle tiré entre `-degrees` et `+degrees`. La rotation pivote autour du coin supérieur gauche du canevas, si bien qu'un grand angle fait sortir du cadre une grande partie de l'image. YOLOX et YOLOv7 conservent la valeur par défaut de base de 10, et YOLOv9 utilise 0 par défaut.

<aug-figure name="degrees" />

### translate

La plage de décalage, en fraction du côté de l'entrée : chaque échantillon transformé se déplace, sur chaque axe, d'une distance pouvant atteindre `translate` fois 640 px avec `imgsz=640`. La valeur par défaut de base est 0.1. YOLO-NAS et PP-YOLOE utilisent 0.25 par défaut, et RTMDet, PicoDet et FOMO 0.

<aug-figure name="translate" />

### shear

La plage de cisaillement en degrés, tirée séparément pour chaque axe. YOLOX et YOLOv7 conservent la valeur par défaut de base de 2, et toutes les autres familles qui lisent ce réglage utilisent 0 par défaut.

<aug-figure name="shear" />

### perspective

Une transformation projective appliquée en plus de la rotation, de l'échelle, du cisaillement et de la translation. Chaque coin du canevas se déplace vers l'intérieur d'une quantité aléatoire pouvant atteindre `min(perspective * 100, 0.2)` fois la moitié du côté du canevas, par axe, si bien que `perspective=0.001` déplace un coin d'au plus 5 pour cent du côté, quelle que soit la résolution, et que toute valeur à partir de `0.002` atteint le même maximum. La valeur par défaut `0` garde la transformation affine, et une valeur négative lève `ValueError` lorsque le premier échantillon est transformé. Les panneaux partagent un même tirage aléatoire, mis à l'échelle par la valeur. YOLOv9 et ses variantes E2E et P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR et RT-DETRv2 l'appliquent.

<aug-figure name="perspective" />

### hsv_prob

La probabilité d'une variation de couleur HSV. Seule la probabilité est un réglage. La variation décale la teinte d'au plus 5 sur l'échelle de teinte de 0 à 179 d'OpenCV, et la saturation et la valeur d'au plus 30 sur une échelle de 0 à 255, et chacun des trois décalages est omis avec une probabilité d'un demi. Tous les pipelines de détection qui lisent `hsv_prob` utilisent ces amplitudes. Les trois derniers panneaux montrent les tirages les plus visibles sur 119.

<aug-figure name="hsv" />

### flip_prob et flipud

Les probabilités de retournement horizontal et vertical. `flip_prob` vaut 0.5 au départ dans toutes les familles sauf FOMO, qui utilise 0 par défaut. `flipud` vaut 0 au départ. RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE et les familles de style DETR n'ont pas de retournement vertical et l'ignorent. La recette de PP-YOLOE utilise à la place une rotation aléatoire de 90 degrés.

<aug-figure name="flip" />

### mixup_prob et mixup_scale

MixUp mélange un second échantillon dans l'échantillon courant et conserve les boîtes des deux. Dans le pipeline YOLOv9, le partenaire est un échantillon entier sans mosaïque, le ratio de mélange est tiré uniformément entre 0 et 1, et `mixup_scale` n'est jamais lu, bien que la spécification de la bibliothèque le classe comme conditionné par la mosaïque. Dans le wrapper YOLOX, utilisé par YOLOX, RTMDet, PicoDet et FOMO, et dans celui de YOLO-NAS, le partenaire est d'abord remis à l'échelle par un facteur tiré de `mixup_scale`, puis les deux images sont moyennées avec un ratio fixe de 0.5. La valeur par défaut de base de `mixup_prob` est 1.0, et YOLOv9 utilise 0 par défaut.

<aug-figure name="mixup" />

Les deux derniers panneaux mélangent un second échantillon dans une mosaïque à 0.5×. Leur légende donne le ratio de mélange λ, le poids de l'échantillon mosaïque.

### no_aug_epochs

`no_aug_epochs` est `used` partout, mais il ne signifie pas la même chose partout. Dans les pipelines à mosaïque, il désactive la mosaïque et MixUp pendant les dernières époques. Dans les pipelines de style DETR, il arrête les augmentations photométriques, de zoom arrière et de recadrage, et façonne la fin du schedule. Dans les pipelines de classification, il désactive l'auto-augmentation, l'effacement, MixUp et CutMix, tandis que le recadrage et les retournements restent actifs.

## Recettes de style DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 et TinyFormer s'entraînent à travers une seule transformation torchvision, sans mosaïque ni transformation affine, tout comme EdgeCrafter (EC) en détection. Les trois opérations fortes de cette transformation sont des constantes de recette plutôt que des réglages :

- `RandomPhotometricDistort` applique chaque modification de couleur (luminosité de 0.875 à 1.125, contraste et saturation de 0.5 à 1.5, teinte ±0.05, et une permutation des canaux) avec une probabilité de 0.5.
- `RandomZoomOut`, avec une probabilité de 0.5, place l'image sur un canevas noir dont le côté peut atteindre quatre fois le sien, si bien que les objets peuvent être réduits au quart de leur taille.
- `RandomIoUCrop`, appliqué avec une probabilité de 0.8, recadre sur une région mesurant entre 0.3 et 1.0 fois chaque côté, choisie pour conserver au moins une partie des boîtes.

Après ces étapes, les boîtes dont l'un des côtés mesure 1 px ou moins sont supprimées. Les trois opérations s'arrêtent à l'époque d'arrêt de la famille, au plus tard lorsque commencent les `no_aug_epochs` dernières époques. Les seuls réglages que lit ce pipeline sont `flip_prob` et `no_aug_epochs`.

<aug-figure name="detr_ops" />

Trois échantillons du début d'un entraînement D-FINE, avec les trois opérations actives :

<aug-figure name="detr_samples" />

RF-DETR a son propre pipeline : un retournement horizontal, puis, avec la probabilité `crop_resize_prob`, un redimensionnement du côté le plus court à 400, 500 ou 600 px et un recadrage carré aléatoire de 384 à 600 px, et enfin un redimensionnement direct à la résolution d'entraînement carrée, qui étire l'image plutôt que de l'inscrire en letterbox. `crop_resize_prob` vaut 0.5 par défaut et se définit depuis Python.

<aug-figure name="rfdetr_crop" />

## Petits objets

`erasing` ne s'applique pas à la détection. Random Erasing appartient à l'ensemble de classification et aucun pipeline de détection n'a d'étape d'effacement, si bien qu'un détecteur s'entraîne de la même façon quelle que soit sa valeur, et qu'il n'y a rien à désactiver. Voici les étapes qui réduisent ou suppriment les petits objets :

- `mosaic_scale`. Dans les pipelines à mosaïque, la taille d'un objet est multipliée par l'échelle tirée, si bien que la borne inférieure fixe la plus petite taille qu'il peut prendre. Avec la valeur par défaut de base `(0.1, 2.0)`, un objet de 20 px peut se retrouver à 2 px. Le pipeline YOLOv9 supprime ensuite les boîtes dont l'un des côtés mesure 2 px ou moins, et sa transformation finale, comme celle de YOLOX, supprime les boîtes dont l'un des côtés mesure 1 px ou moins. Les pixels de l'objet restent dans l'image, sans étiquette. Une borne inférieure de 0.5, comme dans le `(0.5, 1.5)` de YOLOv9, maintient chaque objet à la moitié de sa taille d'inférence ou plus.
- `degrees`, `translate`, `shear` et `perspective` poussent les objets vers le bord, où leurs boîtes sont rognées et passent par le même filtre de taille.
- Dans les recettes de style DETR, `RandomZoomOut` peut réduire les objets au quart de leur taille et `RandomIoUCrop` peut les exclure du recadrage. Aucun des deux n'est un réglage, et tous deux s'arrêtent à l'époque d'arrêt.
- Le recadrage de RF-DETR peut couper des objets à son bord. `crop_resize_prob=0` le désactive.
- La résolution d'entraînement met chaque objet à l'échelle : le letterbox redimensionne une image d'un facteur égal à `imgsz` divisé par son côté le plus long, si bien qu'avec `imgsz=1280`, chaque objet a deux fois la largeur et la hauteur qu'il a à 640. [YOLOv9](/docs/models/yolov9) a aussi une variante P2 avec une tête de stride 4 pour les petits objets.

<code-tabs name="small" />

## L'ensemble de classification

Six réglages n'agissent que sur le pipeline de classification. Les familles de détection ignorent les six.

<code-tabs name="classify" />

`scale` est la plage d'aire du recadrage aléatoire. `scale=0.5` signifie `(0.5, 1.0)`, et une paire explicite fixe les deux bornes. Une petite borne inférieure peut exclure entièrement l'objet du recadrage, comme dans le troisième panneau.

<aug-figure name="cls_crop" />

`auto_augment` accepte `"randaugment"`, `"autoaugment"`, `"augmix"` ou `None`, et applique un tirage de cette politique à chaque échantillon.

<aug-figure name="auto_augment" />

`erasing` est la probabilité de Random Erasing, dans `[0, 1)`. Il s'exécute après la normalisation et remplit de zéros un rectangle couvrant de 2 à 33 pour cent de l'image, ce qui, après normalisation, correspond à la couleur moyenne d'ImageNet.

<aug-figure name="erasing" />

`mixup` et `cutmix` sont des probabilités par batch qui produisent des étiquettes souples. Au plus une des deux s'exécute par batch, MixUp en premier, si bien que les deux sont additives et que leur somme doit valoir au plus 1. Dans les panneaux, A et B sont deux recadrages de la même photo, et chaque légende donne l'étiquette souple sur laquelle l'image mélangée est entraînée.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` et `cutmix` sont désactivés par défaut, si bien que l'entraînement de classification reste inchangé tant que vous ne les demandez pas. `train()` et la CLI routent `mixup` selon la tâche : la classification utilise le mélange par batch, tandis que la détection utilise `mixup_prob`.

`crop_pct=None` conserve le ratio de redimensionnement d'évaluation de la famille. Une valeur explicite modifie le recadrage d'évaluation de la passe de validation exécutée pendant l'entraînement, tandis que l'export conserve le prétraitement natif de la famille.

La CLI expose `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr` et `flipud`. En classification, `mixup` désigne le mélange par batch et vaut 0.0 par défaut. Les retournements horizontaux valent 0.5 par défaut et les verticaux 0.0.

`no_aug_epochs` désactive l'auto-augmentation, l'effacement, MixUp et CutMix pendant les dernières époques, en conservant le recadrage et les retournements. Les recettes de classification fixent cette phase finale à 0 par défaut.

## Réglages propres aux familles

Certains réglages se trouvent dans la sous-classe de configuration d'une famille plutôt que dans la classe de base, si bien qu'ils n'existent que pour cette famille et n'ont pas de flag CLI.

| Famille | Réglage | Effet |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilité d'augmentation copy-paste des instances, uniquement avec `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` réutilise le même échantillon en miroir, `"mixup"` prend un second échantillon |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilité de rotation aléatoire de 90 degrés |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Plafond de vérités terrain par image dans les transformations d'entraînement, 300 par défaut |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste avec `task="segment"`, mode `"flip"` uniquement |
| RF-DETR | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire dans le pipeline natif, 0.5 par défaut |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire, uniquement avec `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilités de variation sur le chemin de pose et de transformation affine tenant compte des points clés |

`max_labels` est celui qui fait perdre des données. Les boîtes au-delà du plafond sont supprimées avec un avertissement unique plutôt qu'une erreur, si bien que les images denses, comme la photographie aérienne, exigent de l'augmenter. La mémoire d'affectation des étiquettes suit l'image la plus dense du batch, pas le plafond.

La mosaïque et MixUp sont désactivés pour l'entraînement de boîtes orientées, quels que soient les réglages, car l'augmentation tenant compte des coins pour les boîtes tournées n'est pas implémentée.

## Pages connexes

- [Matrice d'augmentation](/docs/reference/augmentation-matrix) pour l'état de chaque réglage dans chaque famille, les plages et les valeurs par défaut de chaque famille.
- [Hyperparamètres](/docs/train/hyperparameters) pour `no_aug_epochs` en tant qu'argument de schedule et le reste de `train()`.
- [Datasets](/docs/train/datasets) pour les formats d'étiquettes consommés par ces transformations.
