---
title: Augmentations
seo_title: Augmentations d'entraînement dans LibreYOLO
description: >-
  Les paramètres d'augmentation de TrainConfig, les quatre formes de pipeline
  qui les sous-tendent et la table par famille indiquant quels paramètres sont
  utilisés, conditionnels ou ignorés.
lead: >-
  L'augmentation est configurée par les paramètres de TrainConfig, mais chaque
  famille de modèles exécute son propre pipeline d'entraînement, et un pipeline
  dépourvu de branche mosaïque ignore mosaic_prob au lieu de l'approximer.
keywords:
  - augmentation données yolo
  - augmentation mosaïque
  - mixup
  - variation hsv
  - transformation affine aléatoire
  - augmentation copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
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
  support:
    - label: Lire la table de prise en charge d'une famille
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Afficher uniquement les paramètres ignorés
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
source_hash: 47461cd13aab580c
---

## Régler les paramètres

Les paramètres d'augmentation sont des arguments ordinaires de `train()`.

<code-tabs name="train" />

Deux d'entre eux ont une forme plus courte dans la CLI : `mosaic` correspond à
`mosaic_prob` et `mixup` à `mixup_prob`. Tous les autres paramètres portent le
même nom aux deux endroits.

## Trois états, pas deux

L'effet d'un paramètre dépend de la famille. La bibliothèque conserve une table
déclarative de ce comportement, dont chaque entrée adopte l'un des trois états.

`used` signifie que le paramètre atteint le pipeline et modifie les échantillons.
`ignored` signifie qu'il n'atteint jamais le pipeline, si bien que le définir
n'a aucun effet. `gated_by_mosaic` signifie qu'il s'applique uniquement aux
échantillons passés par la branche mosaïque. Avec `mosaic_prob=0`, il ne se
déclenche donc jamais, même s'il est connecté.

Ce troisième état est celui qui surprend. Dans un pipeline de style YOLOX, la
transformation affine s'exécute sur le canevas mosaïque et MixUp fusionne un
échantillon mosaïque. `mosaic_prob=0` désactive donc silencieusement `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` et
`mixup_scale` en même temps. Le trainer journalise spécifiquement un
avertissement pour MixUp :

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI avertit aussi des paramètres ignorés en ne listant que ceux que vous
avez réellement saisis :

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Quatre formes de pipeline

Les familles se regroupent en quatre pipelines d'entraînement, et le pipeline
détermine presque toutes les réponses.

Le pipeline mosaïque de style YOLOX applique une variation HSV et des
retournements à chaque échantillon, puis exécute la transformation affine et
MixUp dans la branche mosaïque. Il couvre YOLOX, YOLOv7, YOLOv9 et ses variantes
E2E et P2, RTMDet, PicoDet, RT-DETR, RT-DETRv2 et FOMO.

Le pipeline traversant de style DETR ne contient ni mosaïque ni transformation
affine. Sa distorsion photométrique, son zoom arrière et son recadrage IoU sont
des constantes de la recette plutôt que des paramètres de configuration. Seuls
`flip_prob` et `no_aug_epochs` sont donc actifs. Il couvre D-FINE, Dome-DETR,
DEIM, DEIMv2, RT-DETRv4, EC et, avec une différence, RF-DETR.

Le pipeline de classification ImageFolder ignore tous les paramètres de
détection. Son retournement horizontal possède une valeur fixe de 0.5 que
`flip_prob` n'atteint pas. Il dispose de son propre ensemble de paramètres,
décrit ci-dessous.

YOLO-NAS constitue un cas à part : aucune mosaïque, une transformation affine
par échantillon toujours active et MixUp appliqué indépendamment plutôt que
conditionné. Sa valeur `mosaic_scale` est réutilisée comme plage d'échelle de
la transformation affine.

SegFormer et NAFNet utilisent chacun un pipeline propre à leur tâche, dont
l'aléa est fixé dans la famille au lieu d'être configurable. Pour SegFormer,
les paramètres actifs sont les attributs de classe `semantic_scale_jitter` et
`semantic_hsv_prob`, pas `mosaic_scale` et `hsv_prob`. Le recadrage et les
retournements de NAFNet sont des opérations couplées sur l'entrée et la cible,
avec une probabilité fixe de 0.5.

## Paramètres respectés par chaque famille

La table ci-dessous est la spécification distribuée dans
`libreyolo/data/augment/spec.py`, comparée aux branchements réels du pipeline
par les propres tests de la bibliothèque. Consultez-la plutôt que de déduire
le comportement de l'architecture.

<code-tabs name="support" />

Résumé par pipeline pour les paramètres de base :

| Paramètre | Style YOLOX | YOLO-NAS | Style DETR | Classification |
|---|---|---|---|---|
| `mosaic_prob` | utilisé | ignoré | ignoré | ignoré |
| `mixup_prob` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `hsv_prob` | utilisé | utilisé | ignoré | ignoré |
| `flip_prob` | utilisé | utilisé | utilisé | ignoré |
| `flipud` | utilisé | utilisé | ignoré | ignoré |
| `degrees` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `translate` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `shear` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `perspective` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `mosaic_scale` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `mixup_scale` | conditionné par la mosaïque | utilisé | ignoré | ignoré |
| `no_aug_epochs` | utilisé | utilisé | utilisé | utilisé |

Les exceptions dans ces colonnes les restreignent toutes :

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 et FOMO n'ont aucun retournement vertical,
  si bien que `flipud` est ignoré. Le wrapper mosaïque de FOMO est également
  construit sans perspective.
- Le pipeline natif de RF-DETR n'a aucune variation HSV, si bien que `hsv_prob`
  est ignoré en plus de la colonne de style DETR.
- EC respecte `hsv_prob`, `degrees` et `translate`, mais seulement pour
  `task="pose"`, dont la transformation tenant compte des points clés les lit.
  Ses chemins de détection et de segmentation utilisent des recettes
  photométriques fixes.
- DINOv2 suit la colonne de style DETR pour ses tâches de détection et de
  segmentation sémantique, et ajoute l'ensemble de classification pour
  `task="classify"`.

`no_aug_epochs` est `used` partout, mais sa signification varie. Dans les
pipelines mosaïques, il désactive la mosaïque et MixUp pour les dernières
époques. Dans les pipelines de style DETR, il arrête les augmentations
photométriques, de zoom arrière et de recadrage, et façonne la fin du schedule.
Dans les pipelines de classification et de segmentation sémantique, il ne fait
que façonner cette fin.

## L'ensemble de classification

Quatre paramètres pilotent le pipeline de classification, et rien d'autre. Les
familles de détection les ignorent tous les quatre.

<code-tabs name="classify" />

`auto_augment` accepte `"randaugment"`, `"autoaugment"`, `"augmix"` ou `None`.
`erasing` est la probabilité de RandomErasing. `mixup` et `cutmix` sont des
probabilités par batch qui produisent des étiquettes souples ; au plus une
technique s'exécute par batch, MixUp en premier. Elles sont donc additives et
leur somme ne doit pas dépasser 1.

Les quatre sont désactivées par défaut, si bien que l'entraînement de
classification reste inchangé tant que vous ne les demandez pas.

Une collision de noms mérite d'être formulée clairement : dans la CLI, `mixup`
est l'alias du paramètre de détection `mixup_prob`. Le champ de classification
`mixup` n'a pas de syntaxe CLI propre et n'est accessible que par
`model.train(mixup=...)` en Python.

## Paramètres propres aux familles

Certains paramètres se trouvent dans la sous-classe de configuration d'une
famille plutôt que dans la classe de base. Ils n'existent donc que pour cette
famille et ne disposent d'aucun flag CLI.

| Famille | Paramètre | Effet |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilité d'augmentation copy-paste des instances, uniquement pour `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` réutilise le même échantillon retourné, `"mixup"` charge un second échantillon |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilité de rotation aléatoire à 90 degrés |
| YOLOv9 | `max_labels` | Limite de vérités terrain par image dans les transformations d'entraînement, 100 par défaut |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste pour `task="segment"`, mode `"flip"` uniquement |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Probabilité de recadrage-redimensionnement aléatoire |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilités de variation sur le chemin de pose et de transformation affine tenant compte des points clés |

`max_labels` est celui qui fait perdre silencieusement des données. Les boîtes
au-delà de la limite sont supprimées sans erreur. Les images denses, comme les
photographies aériennes, exigent donc d'augmenter cette valeur.

La mosaïque et MixUp sont désactivés pour l'entraînement de boîtes orientées,
quels que soient les paramètres, car l'augmentation tenant compte des coins
pour les boîtes tournées n'est pas implémentée.

## Pages connexes

- [Hyperparamètres](/docs/train/hyperparameters) pour `no_aug_epochs` en tant
  qu'argument de schedule et le reste de `train()`.
- [Datasets](/docs/train/datasets) pour les formats d'étiquettes consommés par
  ces transformations.
