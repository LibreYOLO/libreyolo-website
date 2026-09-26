---
title: libreyolo val
seo_title: Référence de la commande libreyolo val
description: "Évaluer un checkpoint sur un split de dataset en ligne de commande\_: chaque argument avec sa valeur par défaut, et les clés de métriques que chaque tâche renvoie."
lead: >-
  Évalue un seul modèle sur un seul split de dataset et affiche les métriques.
  L'ensemble de métriques dépend de la tâche du modèle, et les chiffres sont
  ceux à partir desquels une ligne de benchmark est construite.
keywords:
  - libreyolo val cli
  - commande validation libreyolo
  - évaluer un modèle yolo en ligne de commande
  - calculer mAP50-95 yolo
  - arguments libreyolo val
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo val
    mono: true
  - label: Requis
    value: 'model, data'
    mono: true
  - label: Sortie
    value: Métriques sur stdout. Graphiques et JSON COCO sous runs/val/exp si demandé
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Graphiques et JSON COCO
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Lisible par une machine
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Synopsis

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi :
`batch=8` et `--batch 8` sont donc le même argument.

## Arguments

| Argument | Défaut | Signification |
|---|---|---|
| `model` | | Chemin du fichier de poids ou nom CLI. Requis |
| `data` | | Chemin vers le YAML du dataset (format YOLO, par exemple `coco8.yaml`). Requis |
| `data_dir` | | Répertoire du dataset indiqué directement, en court-circuitant le chemin du YAML |
| `split` | `val` | Split du dataset : `val`, `test`, `train` |
| `batch` | `16` | Taille de batch |
| `imgsz` | | Taille d'image : `640` (carré) ou `480x640` (HxL). La taille d'entrée propre au modèle si non défini |
| `conf` | `0.001` | Seuil de confiance |
| `iou` | `0.6` | Seuil IoU du NMS |
| `max_det` | `300` | Nombre maximal de prédictions par image après le NMS |
| `eval_max_det` | | Plafond de l'évaluateur COCO. La convention AP@100 de pycocotools si non défini |
| `faster_coco_eval` | `true` | Utilise le backend C++ faster-coco-eval pour les métriques COCO lorsqu'il est installé ; repli sur pycocotools |
| `half` | `false` | Inférence FP16 |
| `amp_dtype` | `float16` | dtype de l'autocast CUDA quand `half=true` : `float16` ou `bfloat16` |
| `save_json` | `false` | Enregistre les résultats JSON au format COCO |
| `save_plots` | `false` | Enregistre les graphiques de validation : métriques, AP par classe, matrice de confusion, échantillons |
| `workers` | `4` | Workers du dataloader |
| `device` | `auto` | Appareil |
| `project` | `runs/val` | Racine du répertoire de sortie |
| `name` | `exp` | Nom de l'expérience |
| `exist_ok` | `false` | Réutilise le répertoire de sortie |
| `allow_download_scripts` | `false` | Autorise le Python embarqué dans les blocs de téléchargement du YAML du dataset |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprime stderr |
| `verbose` | `true` | Sortie verbeuse |
| `help_json` | `false` | Affiche le schéma de la commande en JSON et quitte |

## Exemples

<code-tabs name="examples" />

## Notes

### Ce que sont les métriques

L'ensemble affiché dépend de la tâche du modèle, et la sortie JSON utilise les
mêmes clés.

La détection, la segmentation et les boîtes orientées renvoient `mAP50`,
`mAP50_95`, `precision` et `recall`. Lorsqu'un modèle prédit plusieurs types de
sorties, les groupes par type apparaissent à côté sous les noms `box_metrics`,
`mask_metrics` et `obb_metrics`, chacun portant les quatre mêmes clés.

La classification renvoie `accuracy_top1` et `accuracy_top5`. La détection de
points renvoie `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` et
`mAP_sweep`. La profondeur renvoie `abs_rel`, `rmse`, `delta1`, `delta2` et
`delta3`. La segmentation sémantique renvoie `mIoU` et `pixel_accuracy`. La
restauration renvoie `PSNR` et `SSIM`.

Le résultat JSON porte aussi `eval_backend`, qui nomme la bibliothèque
d'évaluation COCO et la version ayant produit les chiffres, de sorte que deux
exécutions peuvent être comparées en sachant si le même backend les a évaluées
toutes les deux.

### Seuils

Les valeurs par défaut sont ici des valeurs d'évaluation, pas des valeurs de
prédiction : `conf` vaut `0.001` et `iou` vaut `0.6`, là où
[`libreyolo predict`](/docs/cli/predict) utilise `0.25` et `0.45`. Monter `conf`
à un seuil d'affichage fait baisser le rappel et avec lui la mAP : un chiffre
obtenu de cette façon n'est pas comparable à un chiffre publié.

`imgsz` n'est pas défini par défaut, ce qui signifie la taille d'entrée propre
au modèle. Le définir évalue à la taille indiquée, et c'est ainsi qu'un
checkpoint se mesure en dehors de sa résolution native.

### Les datasets qui se téléchargent

Un YAML de dataset dont le champ `download` est une URL se télécharge à la
première utilisation, sans permission supplémentaire. Celui qui embarque un
script de téléchargement Python exige `allow_download_scripts=true`, et la
commande avertit sur stderr que l'exécution de code local a été activée. Les
`coco8.yaml` et `coco128.yaml` fournis reposent sur des URL : ils n'ont besoin
de rien.

### Sortie et codes de retour

stdout porte les métriques ; la progression va sur stderr. `json=true` affiche
un seul objet avec `schema_version`, et `quiet=true` fait taire stderr.

Le code de retour est `0` en cas de succès, `2` pour une erreur d'usage ou de
configuration, `3` quand le dataset est introuvable, `4` quand le modèle ne peut
pas être chargé, et `1` pour les autres échecs à l'exécution.

À lire aussi : [`libreyolo train`](/docs/cli/train), qui lance cette même
évaluation à son propre rythme via `eval_interval`.
