---
title: Matrice d'augmentation
seo_title: Augmentations prises en compte par chaque famille LibreYOLO
description: "Prise en charge des réglages d'augmentation par famille\_: les seize réglages TrainConfig, les trois états, les six archétypes de pipeline et les réglages silencieusement ignorés par une famille."
lead: >-
  Définir un réglage d'augmentation ne garantit pas qu'il atteigne le pipeline.
  Cette page consigne la manière dont chaque famille entraînable traite chaque
  réglage de TrainConfig, d'après la table déclarative fournie par la
  bibliothèque comme source unique de vérité.
keywords:
  - augmentation libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - matrice support augmentation
  - réglages TrainConfig
last_verified: 1.5.0
verification: >-
  Liste des réglages, états, archétypes, écarts par famille et fonctions
  d'assistance lus dans libreyolo/data/augment/spec.py en v1.5.0. Cette table
  est reliée aux véritables pipelines par tests/unit/test_augment_spec.py.
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
source_hash: d2e1b9f5c81072e1
---

## Réglages

Il s'agit de noms de champs `TrainConfig`, et non de leurs formes dans le CLI.
Le CLI associe ses propres alias à ces champs, `--mosaic` définit donc
`mosaic_prob`.

| Réglage | Signification |
|---|---|
| `mosaic_prob` | Probabilité de construire un échantillon mosaic à 4 images |
| `mixup_prob` | Probabilité de mélanger un second échantillon |
| `hsv_prob` | Probabilité de variation aléatoire des couleurs HSV |
| `flip_prob` | Probabilité de retournement horizontal |
| `degrees` | Plage de rotation aléatoire de la transformation affine, en degrés |
| `translate` | Fraction de translation aléatoire de la transformation affine |
| `mosaic_scale` | Plage d'échelle aléatoire de la transformation affine |
| `mixup_scale` | Plage de variation d'échelle appliquée à l'image partenaire MixUp |
| `shear` | Plage de cisaillement aléatoire de la transformation affine, en degrés |
| `perspective` | Amplitude de la transformation projective de la transformation affine |
| `flipud` | Probabilité de retournement vertical |
| `no_aug_epochs` | Dernières époques entraînées sans augmentation forte |
| `auto_augment` | Politique AutoAugment de classification\u00a0: randaugment, autoaugment ou augmix |
| `erasing` | Probabilité RandomErasing de classification |
| `mixup` | Probabilité de batch-MixUp de classification, avec étiquettes souples |
| `cutmix` | Probabilité de batch-CutMix de classification, avec étiquettes souples |

Les quatre derniers constituent le groupe de classification. Les familles de
détection les ignorent. `mixup` est un réglage réservé à l'API\u00a0: l'option
`--mixup` du CLI est l'alias du réglage de détection `mixup_prob`.

<code-tabs name="usage" />

## Trois états

| État | Signification |
|---|---|
| `used` | Le réglage atteint le pipeline d'entraînement de la famille et modifie les échantillons |
| `gated_by_mosaic` | Le réglage s'applique uniquement aux échantillons qui empruntent la branche mosaic, il ne se déclenche donc jamais avec `mosaic_prob == 0` |
| `ignored` | Le réglage n'atteint jamais le pipeline\u00a0; le définir ne fait rien |

`ignored` est l'état à vérifier avant une exécution, car aucune erreur ne se
produit. Le CLI avertit lorsqu'un paramètre d'entraînement explicitement
défini est ignoré par la famille sélectionnée. Le trainer avertit lorsque
`mixup_prob > 0` ne peut pas se déclencher, car la famille conditionne MixUp
à mosaic et `mosaic_prob` vaut zéro.

## Archétypes de pipelines

Chaque famille couverte suit l'un de six pipelines, avec quelques écarts par
famille énumérés ci-dessous.

| Réglage | Style YOLOX | YOLO-NAS | Style DETR | Classification | Sémantique | Restauration |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

Dans le pipeline de style YOLOX, le prétraitement par échantillon applique les
variations HSV et les retournements, tandis que la transformation affine et
MixUp s'exécutent uniquement dans la branche mosaic. YOLO-NAS exécute plutôt
une transformation affine par échantillon toujours active, ignore mosaic et
applique MixUp indépendamment, en réutilisant `mosaic_scale` comme plage
d'échelle affine.

Le pipeline de style DETR utilise une transformation directe sans mosaic. Sa
distorsion photométrique, son zoom arrière et son recadrage IoU sont des
constantes de la recette plutôt que des réglages configurables. `hsv_prob` et
les réglages géométriques ne l'atteignent donc jamais. Le pipeline de
classification utilise une transformation ImageFolder dont le retournement
horizontal est fixé à 0.5 plutôt que défini par `flip_prob`. La variation
d'échelle sémantique et le HSV proviennent d'attributs de classe des familles,
et non des réglages de configuration. Les retournements de restauration sont
des opérations couplées sur l'entrée et la cible, avec une probabilité fixe de
0.5.

`no_aug_epochs` est respecté partout, mais les éléments désactivés diffèrent\u00a0:
mosaic et MixUp pour le style YOLOX, la transformation affine et MixUp pour
YOLO-NAS, les fortes augmentations photométriques et de recadrage ainsi que la
fin du learning rate pour le style DETR, et la fin du scheduler pour les autres.

## Familles par archétype

| Archétype | Familles |
|---|---|
| Style YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Style DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Classification | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Sémantique | `segformer` |
| Restauration | `nafnet` |

Vingt-cinq familles sont couvertes. Une famille absente de cette liste renvoie
un ensemble ignored vide, aucun avertissement n'est donc émis à son sujet.

## Écarts

| Famille | Différence par rapport à son archétype |
|---|---|
| `rtmdet` | `flipud` ignoré\u00a0: sa transformation ne comporte aucun retournement vertical |
| `picodet` | `flipud` ignoré |
| `rtdetr` | `flipud` ignoré |
| `rtdetrv2` | `flipud` ignoré |
| `fomo` | `perspective` et `flipud` ignorés |
| `ec` | `hsv_prob`, `degrees` et `translate` utilisés pour `task="pose"` uniquement\u00a0; detect et segment utilisent des recettes photométriques fixes |
| `dinov2` | Le groupe de classification est utilisé pour `task="classify"` uniquement |

`ec` et `dinov2` sont des familles multitâches. Un réglage n'est donc marqué
ignored que lorsque chaque tâche entraînable de la famille l'ignore. Un
avertissement du CLI n'est ainsi jamais erroné pour une tâche tout en étant
correct pour une autre.

Dome-DETR hérite sans modification des transformations de D-FINE. La seule
fonctionnalité qu'il ne peut pas accepter est l'entraînement multi-échelle,
que sa configuration désactive plutôt que la spécification d'augmentation.

## Réglages propres aux familles

Certaines familles portent des réglages d'augmentation dans leur propre
sous-classe `TrainConfig` plutôt que dans la classe de base. Le CLI ne les
expose pas\u00a0; définissez-les par l'API Python.

| Famille | Réglage | Signification |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilité d'augmentation copy-paste des instances, `task="segment"` uniquement |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Source du copy-paste\u00a0: `flip` reflète le même échantillon, `mixup` utilise un second échantillon |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilité de rotation aléatoire de 90 degrés |
| `rfdetr` | `copy_paste` | Probabilité de copy-paste pour `task="segment"`, mode `flip` uniquement |
| `rfdetr` | `copy_paste_mode` | Mode de source copy-paste pour `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire dans le pipeline natif |
| `dfine` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilité de variation de luminosité et contraste, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilité de transformation affine consciente des points clés, `task="pose"` |

`rot90` s'applique à detect et OBB sur `yolo9`.

## Interroger la spécification

| Assistant | Valeur renvoyée |
|---|---|
| `aug_support(family)` | Table qui associe les réglages à `Support`, ou `None` pour une famille inconnue |
| `ignored_aug_params(family)` | Ensemble des noms de réglages ignorés par la famille\u00a0; vide pour une famille inconnue |
| `uses_mosaic_gating(family)` | Indique si le MixUp de la famille se déclenche uniquement sur les échantillons mosaic |
| `display_name(family)` | Nom de famille destiné au lecteur et utilisé dans les avertissements |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Texte de l'avertissement lorsque MixUp ne peut jamais se déclencher, sinon `None` |

Un objet `Support` est un tuple nommé contenant `status` et `note`, cette
dernière explique pourquoi un réglage est ignoré ou conditionné pour cette
famille.

## Condition mosaic

Pour une famille de style YOLOX, associer `mixup_prob=0.5` à `mosaic_prob=0`
désactive entièrement MixUp, car celui-ci ne s'applique qu'aux échantillons
mosaic. Cette combinaison survient facilement lorsque mosaic est désactivé en
fin d'entraînement. Le trainer consigne un avertissement qui nomme la famille,
et `mixup_gating_warning` est la fonction pure qui le produit.

