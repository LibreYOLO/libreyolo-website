---
title: Matrice d'augmentation
seo_title: Augmentations prises en compte par chaque famille LibreYOLO
description: >-
  Chaque réglage d'augmentation avec sa plage et son nom en CLI, l'état de
  chaque réglage dans chacune des 31 familles entraînables, et les valeurs par
  défaut fournies avec chaque famille.
lead: >-
  Définir un réglage d'augmentation ne garantit pas qu'il atteigne le pipeline.
  Cette page consigne la façon dont chaque famille entraînable traite chaque
  réglage de TrainConfig, et la valeur dont part chaque famille, le tout généré
  à partir de la bibliothèque dans la version qu'elle documente.
keywords:
  - augmentation libreyolo
  - paramètres d'augmentation
  - valeurs par défaut augmentation
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - matrice support augmentation
  - réglages TrainConfig
last_verified: 1.8.0
verification: "Généré par scripts/build-augmentation/collect.py à partir de LibreYOLO 1.8.0\_: la liste des réglages, les états et les notes depuis libreyolo/data/augment/spec.py, que les tests de la bibliothèque maintiennent en accord avec les vrais pipelines, les valeurs par défaut depuis la sous-classe TrainConfig de chaque famille, et les noms en CLI depuis la commande libreyolo train. Les plages et le seul point où la spécification et un pipeline divergent sont vérifiés par rapport à la bibliothèque par le même script."
snippets:
  usage:
    - label: Interroger directement la spécification
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: Valeurs par défaut d'une famille
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Les réglages

Ce sont des noms de champs de `TrainConfig`. `train()` et la CLI acceptent aussi les formes courtes de la colonne CLI, si bien que `mosaic` définit `mosaic_prob`. Dans la CLI, `mixup` est routé selon la tâche : mélange par batch de classification pour les classifieurs, `mixup_prob` pour les détecteurs. `perspective` n'a pas de flag CLI et se définit depuis Python.

<aug-knobs />

La valeur par défaut de base est la valeur définie sur `TrainConfig` lui-même. La plupart des familles en remplacent certaines, et [Valeurs par défaut par famille](#defaults-by-family) donne les valeurs dont part réellement un entraînement. Pour des illustrations de ce que fait chaque réglage, consultez [Augmentations](/docs/train/augmentations).

Les plages entre crochets sont des intervalles, si bien que `[0, 1)` inclut 0 et exclut 1. Les réglages de classification sont vérifiés à la construction de leur pipeline et lèvent `ValueError` hors de leur plage. Les réglages de détection ne sont pas vérifiés par `TrainConfig`, si bien qu'une valeur hors plage entraîne sans le moindre signal. Les exceptions sont une échelle tirée de 0 ou moins et un `perspective` négatif, qui lèvent `ValueError` lorsqu'un échantillon est transformé.

<code-tabs name="usage" />

## Les trois états

| État | Signification |
|---|---|
| `used` | Le réglage atteint le pipeline d'entraînement de la famille et modifie les échantillons |
| `gated_by_mosaic` | Le réglage ne s'applique qu'aux échantillons passés par la branche mosaïque, si bien qu'avec `mosaic_prob == 0` il ne se déclenche jamais |
| `ignored` | Le réglage n'atteint jamais le pipeline ; le définir n'a aucun effet |

`ignored` est l'état à vérifier avant un entraînement, car rien n'échoue. La CLI avertit lorsqu'un paramètre d'entraînement défini explicitement fait partie de ceux que la famille sélectionnée ignore, et le trainer avertit lorsque `mixup_prob > 0` ne peut pas se déclencher parce que la famille conditionne MixUp à la mosaïque et que `mosaic_prob` vaut zéro.

## Réglages pris en compte par chaque famille

<aug-matrix />

Chaque famille couverte suit l'un de six pipelines. Dans le pipeline de style YOLOX, le prétraitement par échantillon applique la variation HSV et les retournements, tandis que la transformation affine et MixUp ne s'exécutent que dans la branche mosaïque. YOLO-NAS exécute plutôt une transformation affine par échantillon toujours active, ignore la mosaïque et applique MixUp indépendamment, en réutilisant `mosaic_scale` comme plage d'échelle de la transformation affine.

Le pipeline de style DETR est une transformation directe, sans mosaïque. Sa distorsion photométrique, son zoom arrière et son recadrage IoU sont des constantes de recette plutôt que des réglages configurables, ce qui explique pourquoi `hsv_prob` et les réglages géométriques ne l'atteignent jamais. La classification utilise `flip_prob` pour les retournements horizontaux et `flipud` pour les retournements verticaux. La variation d'échelle et la variation HSV de la segmentation sémantique viennent d'attributs de classe des familles plutôt que de réglages de configuration, et les retournements de restauration sont des opérations couplées sur l'entrée et la cible, avec une probabilité fixe de 0.5.

`no_aug_epochs` est pris en compte partout, mais ce qu'il désactive varie : la mosaïque et MixUp pour le style YOLOX, la transformation affine et MixUp pour YOLO-NAS, les augmentations photométriques fortes et de recadrage ainsi que la fin du schedule de learning rate pour le style DETR, et l'auto-augmentation, l'effacement, MixUp et CutMix pour la classification. Le recadrage et les retournements de classification restent actifs.

Lorsqu'une famille diffère du reste de son pipeline, la raison est l'une des suivantes :

| Famille | Différence |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignoré : leurs transformations n'ont pas de retournement vertical |
| `fomo` | `perspective` et `flipud` ignorés |
| `ppyoloe` | `flipud` ignoré : la recette source utilise à la place une rotation aléatoire de 90 degrés |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignoré : leur MixUp mélange des images entières à un ratio aléatoire sans remettre à l'échelle ni l'une ni l'autre |
| `gtr` | `mosaic_prob` et `mixup_prob` utilisés, pendant les `mosaic_epochs` premières époques ; `degrees`, `translate` et `mosaic_scale` façonnent la transformation affine qui suit la mosaïque |
| `rfdetr` | `hsv_prob` ignoré : le pipeline natif n'a pas de variation HSV |
| `ec` | `hsv_prob`, `degrees` et `translate` utilisés, pour `task="pose"` uniquement ; detect et segment utilisent des recettes photométriques fixes |
| `dinov2` | L'ensemble de classification est utilisé, pour `task="classify"` uniquement |

`ec` et `dinov2` sont des familles multitâches, si bien qu'un réglage n'est marqué ignoré que lorsque toutes les tâches entraînables de la famille l'ignorent. Ainsi, l'avertissement de la CLI n'est jamais faux pour une tâche tout en étant juste pour une autre.

Dome-DETR hérite sans modification des transformations de D-FINE. La seule chose qu'il ne peut pas prendre en charge est l'entraînement multi-échelle, que désactive sa configuration plutôt que la spécification d'augmentation.

Trente et une familles sont couvertes. Pour une famille absente de cette liste, `ignored_aug_params` renvoie un ensemble vide, si bien qu'aucun avertissement n'est émis pour elle.

## Valeurs par défaut par famille

La valeur dont part la configuration de chaque famille de détection. Une cellule vide signifie que la famille ignore ce réglage, si bien que sa valeur par défaut ne change rien.

<aug-defaults />

Deux tendances comptent plus que n'importe quelle valeur isolée. RT-DETR, RT-DETRv2, PicoDet et FOMO partent de `mosaic_prob=0`, si bien que, sur ces familles, les réglages marqués comme conditionnés dans la matrice n'ont aucun effet tant que la mosaïque n'est pas activée. YOLOv9 part avec la rotation, le cisaillement et MixUp désactivés, et avec `mosaic_scale=(0.5, 1.5)`, là où la valeur par défaut de base est `(0.1, 2.0)`.

EdgeCrafter ne lit `hsv_prob`, `degrees` et `translate` que pour `task="pose"`, si bien que sa ligne affiche, pour ces trois réglages, les valeurs par défaut de la tâche pose.

Les familles de classification partent des valeurs par défaut de base : `flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, le `crop_pct` propre à la famille, et `auto_augment`, `erasing`, `mixup` et `cutmix` désactivés.

## Réglages propres aux familles

Certaines familles portent des réglages d'augmentation sur leur propre sous-classe `TrainConfig` plutôt que sur la classe de base. La CLI ne les expose pas ; définissez-les par l'API Python.

| Famille | Réglage | Signification |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilité d'augmentation copy-paste des instances, `task="segment"` uniquement |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Source du copy-paste : `flip` retourne le même échantillon en miroir, `mixup` utilise un second échantillon |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilité de rotation aléatoire de 90 degrés |
| `rfdetr` | `copy_paste` | Probabilité de copy-paste pour `task="segment"`, mode `flip` uniquement |
| `rfdetr` | `copy_paste_mode` | Mode de source du copy-paste pour `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire dans le pipeline natif |
| `dfine` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilité de variation de luminosité et de contraste, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilité de transformation affine tenant compte des points clés, `task="pose"` |

`rot90` s'applique à detect et à OBB sur `yolo9`.

## Interroger la spécification

| Fonction utilitaire | Valeur renvoyée |
|---|---|
| `aug_support(family)` | La table qui associe chaque réglage à son `Support`, ou `None` pour une famille inconnue |
| `ignored_aug_params(family)` | L'ensemble des noms de réglages que la famille ignore ; vide pour une famille inconnue |
| `uses_mosaic_gating(family)` | Indique si le MixUp de la famille ne se déclenche que sur les échantillons mosaïque |
| `display_name(family)` | Le nom de la famille destiné aux lecteurs, utilisé dans les avertissements |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Le texte de l'avertissement lorsque MixUp ne peut jamais se déclencher, sinon `None` |

Un `Support` est un tuple nommé composé de `status` et de `note`, où la note explique pourquoi un réglage est ignoré ou conditionné pour cette famille. Dans la version 1.8.0, la spécification indique encore `mixup_scale` comme conditionné pour les familles qui s'entraînent via le wrapper mosaïque de YOLOv9, si bien que la fonction `aug_support` le signale comme conditionné pour elles, alors que la matrice ci-dessus, vérifiée par rapport au pipeline, le marque comme ignoré.

## Le conditionnement par la mosaïque

Pour une famille de style YOLOX, `mixup_prob=0.5` avec `mosaic_prob=0` désactive entièrement MixUp, car MixUp ne s'applique qu'aux échantillons mosaïque. Cette combinaison est facile à atteindre en désactivant la mosaïque en fin d'entraînement. Le trainer consigne un avertissement qui nomme la famille, et `mixup_gating_warning` est la fonction pure qui le produit.
