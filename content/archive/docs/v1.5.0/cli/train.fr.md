---
title: libreyolo train
seo_title: référence de la commande libreyolo train
description: "Entraînez un modèle depuis la ligne de commande\_: les 59 arguments avec leurs valeurs par défaut, comment les valeurs par défaut d'une famille les remplacent, et quels arguments une famille ignore."
lead: >-
  Entraîne un modèle sur un dataset et écrit les checkpoints, les métriques et
  les logs dans un répertoire de run. Chaque argument ci-dessous a une valeur
  par défaut issue de la définition de la commande, que la config d'entraînement
  propre à une famille de modèles peut remplacer.
keywords:
  - libreyolo train cli
  - commande libreyolo train
  - entrainer yolo en ligne de commande
  - arguments libreyolo train
  - entrainer yolo sur son propre dataset
  - geler des couches yolo
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo train
    mono: true
  - label: Requis
    value: data
    mono: true
  - label: Sortie
    value: 'Checkpoints, métriques et logs dans runs/train/exp'
snippets:
  examples:
    - label: Basique
      language: bash
      code: >
        # coco8.yaml est fourni avec le paquet et télécharge ses 8 images au
        premier usage.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Vérifier d'abord la config résolue
      language: bash
      code: >
        # Affiche ce que le run utiliserait, valeurs par défaut de la famille

        # incluses, et quitte sans entraîner ni charger de données.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Run nommé avec une recette explicite
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Synopsis

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi :
`epochs=50` et `--epochs 50` sont donc le même argument. Les booléens acceptent
`true` et `false` : `amp=false` devient `--no-amp` là où le flag possède une
forme négative.

## Arguments

### Modèle et données

| Argument | Défaut | Signification |
|---|---|---|
| `data` | | Chemin vers le YAML du dataset (format YOLO, par exemple `coco8.yaml`). Requis |
| `model` | `yolox-s` | Nom du modèle ou chemin vers des poids |
| `task` | | Forçage explicite de la tâche : `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Utiliser des poids pré-entraînés. `false` construit l'architecture et entraîne à partir de zéro |
| `allow_download_scripts` | `false` | Autoriser le Python embarqué dans les blocs de téléchargement du YAML de dataset |

### Boucle d'entraînement

| Argument | Défaut | Signification |
|---|---|---|
| `epochs` | `300` | Époques d'entraînement |
| `batch` | `16` | Taille de batch par appareil |
| `imgsz` | `640` | Taille des images d'entraînement : `640` (carré) ou `480x640` (HxW) |
| `device` | `auto` | Appareil : `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Workers du dataloader |
| `cache` | `false` | Mettre les images en cache pour accélérer le chargement des données : `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Graine aléatoire |
| `resume` | | Reprendre l'entraînement : `true`, ou un chemin vers un checkpoint |
| `amp` | `true` | Précision mixte automatique |
| `amp_dtype` | `float16` | dtype de l'AMP CUDA : `float16` ou `bfloat16` |
| `cuda_graph` | `false` | Capturer les passes avant et arrière de l'entraînement dans des CUDA graphs. GPU unique, familles prises en charge uniquement ; les autres s'exécutent en mode eager |
| `lora` | `false` | Fine-tuning LoRA, pour les familles transformer listées sous Notes |
| `freeze` | | Geler des couches : un nombre entier, une liste d'indices, ou des noms de modules |

### Distillation

| Argument | Défaut | Signification |
|---|---|---|
| `distill_model` | | Enseignant : un checkpoint de détecteur, ou un identifiant de modèle de fondation enseignant tel que `dinov2` pour la distillation des caractéristiques du backbone |
| `dis` | | Poids de la loss (fonction de perte) de distillation. La valeur par défaut publiée pour le type de loss si non défini |
| `distill_loss_type` | `mgd` | Loss de caractéristiques pour les enseignants détecteurs : `mgd`, `cwd`. Les enseignants de fondation utilisent toujours `feat_mse` |

### Optimiseur

| Argument | Défaut | Signification |
|---|---|---|
| `optimizer` | `sgd` | Optimiseur : `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate initial |
| `momentum` | `0.937` | Momentum de SGD, et coefficient du premier moment pour les optimiseurs Adam |
| `weight_decay` | `0.0005` | Régularisation L2 |
| `nesterov` | `true` | Momentum de Nesterov |

### Scheduler

| Argument | Défaut | Signification |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Type de schedule du LR |
| `warmup_epochs` | `5` | Durée du warmup |
| `warmup_lr_start` | `0.0` | LR initial du warmup |
| `min_lr_ratio` | `0.05` | Ratio de LR minimal |
| `lr_drop` | `100` | Époque de baisse du LR par paliers pour RF-DETR |

### Augmentation

| Argument | Défaut | Signification |
|---|---|---|
| `mosaic` | `1.0` | Probabilité du mosaic |
| `mixup` | `1.0` | Probabilité du mixup |
| `hsv_prob` | `1.0` | Probabilité du jitter HSV |
| `flip_prob` | `0.5` | Probabilité de retournement horizontal |
| `degrees` | `10.0` | Amplitude de rotation, en plus et en moins, en degrés |
| `translate` | `0.1` | Ratio de translation |
| `shear` | `2.0` | Angle de cisaillement |
| `mosaic_scale` | `(0.1,2.0)` | Plage d'échelle du mosaic |
| `mixup_scale` | `(0.5,1.5)` | Plage d'échelle du mixup |
| `no_aug_epochs` | `15` | Désactiver l'augmentation pendant les N dernières époques |

### EMA

| Argument | Défaut | Signification |
|---|---|---|
| `ema` | `true` | Moyenne mobile exponentielle |
| `ema_decay` | `0.9998` | Facteur de décroissance de l'EMA |

### Validation pendant l'entraînement

| Argument | Défaut | Signification |
|---|---|---|
| `val` | `true` | Valider pendant l'entraînement |
| `eval_interval` | `10` | Valider toutes les N époques |
| `max_det` | `300` | Nombre maximum de prédictions par image après le NMS de validation |
| `eval_max_det` | | Plafond de l'évaluateur COCO. La convention AP@100 de pycocotools si non défini |
| `faster_coco_eval` | `true` | Utiliser le backend C++ faster-coco-eval pour les métriques COCO s'il est installé ; repli sur pycocotools |
| `save_plots` | `false` | Enregistrer les graphiques de validation finale pendant l'entraînement |
| `patience` | `50` | Patience de l'early stopping. `0` le désactive |

### Sortie

| Argument | Défaut | Signification |
|---|---|---|
| `project` | `runs/train` | Racine du répertoire de sortie |
| `name` | `exp` | Nom de l'expérience |
| `exist_ok` | `false` | Réutiliser un répertoire de sortie existant |
| `save_period` | `10` | Enregistrer un checkpoint toutes les N époques |
| `log_interval` | `10` | Journaliser la loss tous les N batchs |

### Flags agent

| Argument | Défaut | Signification |
|---|---|---|
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprimer stderr |
| `dry_run` | `false` | Résoudre et afficher la config sans exécuter |
| `help_json` | `false` | Dumper le schéma de la commande en JSON et quitter |

## Exemples

<code-tabs name="examples" />

## Notes

### Les valeurs par défaut ci-dessus ne sont pas toujours celles utilisées

Chaque famille de modèles porte sa propre config d'entraînement, et là où cette
config diffère de la config de base, sa valeur remplace la valeur par défaut de
la commande pour tout argument que vous n'avez pas défini explicitement. Définir
l'argument vous-même l'emporte toujours. `libreyolo cfg` affiche les valeurs par
défaut de base et les surcharges par famille, ce qui est le moyen de voir ce
qu'une famille donnée utilisera réellement.

`imgsz` est l'argument pour lequel cela compte le plus. La valeur par défaut de
la commande est `640`, qui n'est pas l'entrée native de tous les checkpoints :
les tailles de détection publiées pour RF-DETR sont 384, 512, 576 et 704, et les
checkpoints YOLOX `n` et `t` sont en 416. RF-DETR et DEIMv2 sont traités en ne
transmettant `imgsz` que s'il a été défini explicitement, si bien que leur propre
taille reste sinon en vigueur. Les autres familles reçoivent la valeur telle
quelle et s'entraînent à cette taille. FOMO est la famille stricte : chaque
taille n'accepte que son entrée native (96, 192 et 224), donc un run FOMO exige
que `imgsz` soit réglé en conséquence, sinon il s'arrête sur une erreur. RF-DETR
impose en plus que la valeur soit divisible par sa taille de patch multipliée par
son nombre de fenêtres, et signale les deux tailles légales les plus proches
lorsque ce n'est pas le cas.

### Les arguments qu'une famille ignore

Toutes les familles ne lisent pas tous les arguments, et c'est sur ceux
d'augmentation que cela se voit. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 et
DINOv2 s'entraînent via des pipelines pass-through sans mosaic, sans mixup et
sans déformation affine, si bien que `mosaic`, `mixup`, `hsv_prob`, `degrees`,
`translate`, `shear`, `mosaic_scale` et `mixup_scale` n'atteignent rien chez
elles. EC partage ce pipeline mais lit bien `hsv_prob`, `degrees` et `translate`
quand sa tâche est pose. Les familles de classification, SegFormer et NAFNet
ignorent tout cet ensemble et `flip_prob` avec lui, parce que leur retournement
s'applique à une probabilité fixe plutôt que configurable. YOLO-NAS ignore
`mosaic` seul, puisqu'il augmente à la place avec une transformation affine par
échantillon toujours active. RF-DETR en ignore trois autres au-delà de cette
liste : `optimizer`, `momentum` et `nesterov`.

Définir l'un d'eux n'est pas une erreur. Le run écrit sur stderr une ligne qui
nomme la famille et les arguments qu'elle ignorera, puis entraîne, et cette ligne
est la liste faisant autorité pour la version installée. C'est aussi le seul
signal, donc un run scripté avec `quiet=true` supprime cet avertissement en même
temps que tout le reste sur stderr.

`val=false` est un cas voisin. Il met `eval_interval` à `0` pour la plupart des
familles ; RF-DETR ne peut pas désactiver la validation ainsi et journalise qu'il
a ignoré la demande.

### Autres comportements à connaître

`lora=true` est accepté par RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 et v4,
EC et ConvNeXt. Toute autre famille se termine avec `config_unsupported` plutôt
que d'entraîner sans.

`pretrained=false` combiné avec `resume` est refusé pour les familles qui
prennent en charge l'entraînement à partir de zéro, puisque les deux demandent
des choses opposées.

`mosaic` et `mixup` sont les orthographes en ligne de commande des champs de
config `mosaic_prob` et `mixup_prob`. Sur les familles dont le mixup ne
s'applique qu'aux échantillons mosaic, `mixup` au-dessus de zéro avec `mosaic` à
zéro ne se déclenche jamais, et le run le signale.

`dry_run=true` résout la référence du modèle, applique les valeurs par défaut de
la famille et affiche la config avec laquelle il entraînerait. Il ne charge pas
le dataset, c'est donc le moyen peu coûteux de confirmer qu'un argument a bien
atteint la valeur que vous attendiez.

stdout porte l'objet de résultat final ; la progression et les avertissements
vont sur stderr. Le code de sortie est `0` en cas de succès, `2` pour une erreur
d'usage ou de configuration, `3` quand le dataset est introuvable ou illisible,
`4` quand le modèle ne peut pas être chargé, et `1` pour les autres échecs
d'exécution.

À voir aussi : [`libreyolo doctor`](/docs/cli/doctor) pour vérifier un dataset
avant de lancer un run, [`libreyolo monitor`](/docs/cli/monitor) pour suivre un
run dans le navigateur, [`libreyolo val`](/docs/cli/val) pour mesurer le
résultat.
