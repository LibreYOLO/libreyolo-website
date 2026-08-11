---
title: Validation et métriques
seo_title: Validation et métriques dans LibreYOLO
description: >-
  Exécuter val() sur tout modèle, lire les clés de métriques renvoyées par
  chaque tâche, choisir un backend d'évaluation et activer une perte de
  validation en plus de la métrique de précision.
lead: >-
  La validation exécute un modèle sur une partition de dataset avec val() et
  renvoie un dictionnaire plat de clés de métriques et de valeurs flottantes.
  Les clés sont des chaînes littérales et dépendent de la tâche, pas de la
  famille.
keywords:
  - map50-95
  - évaluation coco
  - métriques validation
  - faster-coco-eval
  - pycocotools
  - perte validation
  - miou
  - qualité panoptique
  - précision top1
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Sur une autre partition
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Écrire les prédictions au format COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Exécuter une validation

`val()` reçoit le dataset et renvoie les métriques.

<code-tabs name="val" />

La valeur renvoyée est un dictionnaire ordinaire `dict[str, float]`. Chaque clé
est littérale. Lisez-la par son nom plutôt que par sa position.

Les principaux arguments sont `data`, `split`, `batch`, `imgsz`, `conf`,
`iou`, `workers`, `device`, `augment`, `save_json` et `verbose`. `conf` vaut
`0.001` et `iou` `0.6` par défaut, des seuils bien plus permissifs que ceux de
la prédiction, car un balayage mAP a besoin de la queue à faible confiance.
`imgsz` prend par défaut la taille d'entrée propre au modèle plutôt qu'un nombre
fixe. `split` accepte uniquement `val`, `test` ou `train`.

Tout autre champ de la configuration de validation passe comme argument nommé,
notamment `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`,
`cache` et `save_plots`.

## Clés de métriques par tâche

La détection renvoie la famille de nombres COCO :

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Deux constituent des pièges. `metrics/precision` et `metrics/recall` sont des
alias conservés pour la rétrocompatibilité. Ils contiennent respectivement les
valeurs mAP 50-95 et AR@100, et non une paire précision-rappel. Utilisez les
clés nommées.

La segmentation d'instances renvoie les valeurs mAP et AR ci-dessus comme
nombres de masques sous les clés sans suffixe, avec les versions de boîtes sous
`(B)` et une répétition des versions de masques sous `(M)`. Pour cette tâche,
la précision et le rappel existent uniquement avec les suffixes
`metrics/precision(B)`/`metrics/recall(B)` et
`metrics/precision(M)`/`metrics/recall(M)`. Les deux paires contiennent les
mêmes alias que pour la détection : mAP50-95 et AR@100 des boîtes pour `(B)`,
des masques pour `(M)`.

| Tâche | Clés |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, plus les répartitions par taille et de rappel ci-dessus |
| segment | versions masque des clés de détection ci-dessus, les clés sans suffixe désignent le masque ; `precision`/`recall` existent uniquement sous `(B)`/`(M)` et sont des alias identiques |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L` et les clés `keypoints_AR` correspondantes |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, plus les copies suffixées `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, plus une clé de balayage mAP |

Pour OBB, `metrics/precision` et `metrics/recall` ne sont pas des alias. Il
s'agit de la précision et du rappel réels à une IoU de 0,50, au point de
fonctionnement le plus permissif, donc pour chaque prédiction qui franchit
`conf`, égal à `0.001` par défaut. Les copies suffixées `(OBB)` répètent les
mêmes quatre valeurs sous un nom propre à la tâche, selon la même convention que
`(B)` et `(M)`.

`accuracy_top5` est en réalité top-`min(5, num_classes)`. Sur un dataset à trois
classes, il s'agit donc du top-3, satisfait par chaque échantillon et égal à 1,0.

La clé de balayage de la tâche de points est construite depuis les seuils de
distance. Avec les valeurs par défaut, elle est
`metrics/mAP@[0.01:0.10]`, et la clé à seuil unique est
`metrics/mAP@0.01`. Transmettre `dist_thresholds` change les deux chaînes.

La plupart des tâches renvoient aussi une clé `fitness`, le nombre unique
employé par défaut pour choisir le meilleur checkpoint. La détection, la
segmentation et OBB n'en contiennent pas. Leurs familles se sélectionnent sur
`metrics/mAP50-95`, présent dans leurs dictionnaires. La pose ne renvoie ni
`fitness` ni `metrics/mAP50-95`. Ses programmes d'entraînement définissent
plutôt `best_metric_key` à `metrics/keypoints_mAP50-95`.

## Clés de vitesse

Chaque validateur ajoute des temps :

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Il s'agit de millisecondes par image, moyennées sur l'exécution. Elles décrivent
la machine et les réglages employés. Un chiffre n'a donc de sens qu'avec le
matériel, la taille de lot et la précision qui l'accompagnent.

## Backend d'évaluation

Les métriques de détection et de segmentation sont calculées par un évaluateur
COCO. `faster_coco_eval=True`, valeur par défaut, sélectionne le backend C++
lorsque le paquet `faster-coco-eval` est installé. Sinon, l'exécution revient à
pycocotools avec un avertissement par processus :

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Le backend réellement exécuté est enregistré dans `last_eval_backend` sur le
modèle, et la CLI le rapporte dans sa sortie pour les tâches de type détection.
Définissez `LIBREYOLO_FASTER_COCO_EVAL` pour remplacer la valeur de
configuration depuis l'environnement.

`iou_thresholds` est respecté uniquement dans le parcours OBB. Le parcours
COCO emploie son propre balayage fixe de 0,50 à 0,95 et ignore cette valeur.

## Perte de validation

Par défaut, la validation rapporte uniquement la précision. `val_loss=True`
calcule aussi l'objectif d'entraînement de la famille sur les lots de
validation.

<code-tabs name="valloss" />

Cette option émet `metrics/loss` et une clé `metrics/loss/<component>` par terme,
pondérés exactement comme pendant l'entraînement afin que les composants
s'additionnent au total. Dans un système de journalisation, ils deviennent
`val/loss` et `val/loss/<component>`, tandis que `libreyolo monitor` superpose
`metrics/loss` à `train/loss`.

Les composants sont propres à la famille :

| Tâche | Familles | Composants |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Cette option est désactivée par défaut, car l'affectation des cibles ajoute du
temps et de la mémoire à la validation. Le validateur réutilise la sortie déjà
produite pour la métrique de précision au lieu d'effectuer une seconde
propagation. Il s'exécute sous `no_grad` sur le modèle d'évaluation ou EMA. En
multi-GPU, le calcul reste local au rang 0 sans opérations collectives. La
sélection du meilleur checkpoint reste fondée sur la métrique de précision.

Trois opérations sont volontairement absentes. La perte n'inclut jamais les
termes de débruitage contrastif, car ceux-ci exigent la vérité terrain pendant
la propagation, alors que les propagations de validation n'en reçoivent pas.
Elle rapporte le modèle en mode évaluation. Lorsque les propagations
d'entraînement et d'évaluation d'une famille diffèrent réellement, par les
statistiques BatchNorm ou la profondeur stochastique, le nombre reflète donc le
mode évaluation, comme prévu. Enfin, une tâche non implémentée par une famille
déclenche une erreur de configuration à la préparation au lieu d'être
silencieusement ignorée :

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO est l'exception qui ne change rien. Son validateur calculait déjà cette
perte, et `val_loss=True` modifie uniquement les clés sous lesquelles elle est
publiée.

La validation augmentée et la perte de validation ne peuvent pas être combinées.
Les demander ensemble déclenche une erreur.

## Fichiers écrits par une validation

`val()` écrit toujours `config.yaml` dans son répertoire de sortie. Sans
`save_dir`, celui-ci vaut par défaut
`runs/val/<model>_<size>_<timestamp>`.

<code-tabs name="json" />

`save_json=True` écrit `predictions.json` pour la détection, et
`predictions_bbox.json` avec `predictions_masks.json` pour la segmentation.
OBB ne le prend pas en charge et le signale.

`save_plots=True` écrit dans un sous-répertoire `plots/`. La détection produit
`box_metrics.png`, des graphiques AP et rappel par classe, des courbes
précision-rappel et confiance, une matrice de confusion et des images
d'échantillons annotées lorsque OpenCV est installé. La segmentation ajoute les
copies de chaque graphique pour les masques, et la pose possède son propre
ensemble de métriques et de courbes. Les autres validateurs n'implémentent aucun
graphique. La classification, la segmentation sémantique et panoptique, la
profondeur, les normales, les contours, la restauration, le cache, l'OCR, OBB
et les points n'y écrivent rien. Un échec de tracé produit un avertissement sans
jamais interrompre l'exécution.

## Validation pendant l'entraînement

L'entraînement valide toutes les `eval_interval` époques sur la partition `val`
du dataset. Les métriques obtenues pilotent la sélection de `best.pt`, l'arrêt
anticipé `patience` et les clés `val/` de chaque système de journalisation. La
validation emploie les poids EMA lorsque l'EMA est active.

Consultez les [hyperparamètres](/docs/train/hyperparameters) pour
`eval_interval`, `patience` et `save_plots`, ainsi que les
[systèmes de journalisation](/docs/train/loggers) pour la destination des
nombres.

## Voir aussi

- [Datasets](/docs/train/datasets) pour les clés de partitions et les formats
  lus par les validateurs.
