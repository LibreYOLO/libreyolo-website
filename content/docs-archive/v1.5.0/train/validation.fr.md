---
title: Validation et mesures
seo_title: Validation et mesures dans LibreYOLO
description: >-
  Exécutez val() sur n'importe quel modèle, lisez les clés de mesures renvoyées
  par chaque tâche, choisissez un backend d'évaluation et activez une loss de
  validation en plus de la mesure d'exactitude.
lead: >-
  La validation exécute un modèle sur un split de dataset par val() et renvoie
  un dictionnaire plat de clés de mesures et de valeurs flottantes. Les clés
  sont des chaînes littérales, et celles que vous obtenez dépendent de la tâche,
  pas de la famille.
keywords:
  - map50-95
  - évaluation coco
  - métriques validation
  - faster-coco-eval
  - pycocotools
  - loss validation
  - miou
  - qualité panoptique
  - exactitude top1
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
    - label: Sur un autre split
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
    - label: Écrire des prédictions au format COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Exécuter une validation

`val()` accepte le dataset et renvoie les mesures.

<code-tabs name="val" />

La valeur de retour est un simple `dict[str, float]`. Chaque clé est littérale,
lisez-la donc par son nom plutôt que par sa position.

Les principaux arguments sont `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` et `verbose`. `conf` vaut `0.001`
par défaut et `iou` vaut `0.6`, deux valeurs bien plus permissives que celles
de la prédiction, car un balayage mAP exige la fin à faible confiance. `imgsz`
utilise par défaut la taille d'entrée propre au modèle plutôt qu'un nombre fixe.
`split` accepte uniquement `val`, `test` ou `train`.

Tout autre champ de la configuration de validation est transmis comme argument
nommé, notamment `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`,
`cache` et `save_plots`.

## Clés de mesures par tâche

La détection renvoie la famille de valeurs COCO :

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Deux de ces valeurs sont trompeuses. `metrics/precision` et `metrics/recall`
sont des alias conservés pour la rétrocompatibilité : ils contiennent les
valeurs mAP 50-95 et AR@100, pas une paire précision-rappel. Utilisez les clés
nommées.

La segmentation d'instances renvoie les valeurs mAP et AR ci-dessus pour les
masques sous les clés sans suffixe, les versions des boîtes sous le suffixe
`(B)` et de nouveau les versions des masques sous `(M)`. Pour cette tâche, la
précision et le rappel existent uniquement sous les formes suffixées
`metrics/precision(B)`/`metrics/recall(B)` et
`metrics/precision(M)`/`metrics/recall(M)`, et les deux paires contiennent les
mêmes alias que pour la détection : la paire `(B)` est la mAP50-95 des boîtes et
l'AR@100 des boîtes, la paire `(M)` la mAP50-95 des masques et l'AR@100 des
masques.

| Tâche | Clés |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, ainsi que les répartitions par taille et rappel ci-dessus |
| segment | versions masque des clés de détection ci-dessus (les clés sans suffixe désignent les masques) ; `precision`/`recall` n'existent qu'avec `(B)`/`(M)`, toutes deux avec les mêmes alias |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, et les clés `keypoints_AR` correspondantes |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, ainsi que des copies suffixées `(OBB)` |
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

Les valeurs `metrics/precision` et `metrics/recall` d'OBB ne sont pas des
alias : ce sont la précision et le rappel réels à une IoU de 0.50, au point de
fonctionnement le plus permissif, soit chaque prédiction qui dépasse `conf`,
`0.001` par défaut. Les copies suffixées `(OBB)` répètent les quatre mêmes
valeurs sous un nom propre à la tâche, selon la même convention que `(B)` et
`(M)` ci-dessus.

`accuracy_top5` correspond réellement au top-`min(5, num_classes)`. Sur un
dataset à trois classes, il s'agit donc du top-3, satisfait par chaque
échantillon et par conséquent égal à 1.0.

La clé de balayage de la tâche point est construite à partir des seuils de
distance. Avec les valeurs par défaut, elle est `metrics/mAP@[0.01:0.10]`, et la
clé à seuil unique est `metrics/mAP@0.01`. Passer `dist_thresholds` modifie les
deux chaînes.

La plupart des tâches renvoient aussi une clé `fitness`, l'unique valeur
utilisée par défaut pour sélectionner le meilleur checkpoint. La détection, la
segmentation et OBB n'en possèdent pas ; leurs familles sont sélectionnées selon
`metrics/mAP50-95`, que leurs dictionnaires renvoient. La pose ne renvoie ni
`fitness` ni `metrics/mAP50-95` ; ses trainers définissent à la place
`best_metric_key` sur `metrics/keypoints_mAP50-95`.

## Clés de vitesse

Chaque validateur ajoute les temps suivants :

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Il s'agit de millisecondes par image, moyennées sur l'exécution. Elles décrivent
la machine et les paramètres utilisés. Une valeur qui en est tirée n'a donc de
sens que si elle est rapportée avec le matériel, la taille de batch et la
précision.

## Backend d'évaluation

Les mesures de détection et de segmentation sont calculées avec un évaluateur
COCO, et `faster_coco_eval=True`, la valeur par défaut, sélectionne le backend
C++ lorsque le package `faster-coco-eval` est installé. Dans le cas contraire,
l'exécution revient à pycocotools avec un avertissement par processus :

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Le backend réellement utilisé est consigné sur le modèle dans
`last_eval_backend`, et la CLI le rapporte dans sa sortie pour les tâches de
style détection. Définissez `LIBREYOLO_FASTER_COCO_EVAL` pour remplacer la
valeur de configuration depuis l'environnement.

`iou_thresholds` est respecté uniquement sur le chemin OBB. Le chemin COCO
effectue son propre balayage fixe de 0.50 à 0.95 et ignore cette valeur.

## Loss de validation

Par défaut, la validation rapporte uniquement l'exactitude. `val_loss=True`
calcule aussi l'objectif d'entraînement de la famille sur les batchs de
validation.

<code-tabs name="valloss" />

Cette option émet `metrics/loss` ainsi qu'une clé `metrics/loss/<component>` par
terme, avec exactement les mêmes pondérations que pendant l'entraînement, si
bien que les composants s'additionnent pour former le total. Dans un logger,
ils apparaissent sous `val/loss` et `val/loss/<component>`, et `libreyolo
monitor` superpose `metrics/loss` à `train/loss`.

Les composants sont ceux de chaque famille :

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
temps et de la mémoire à la validation. Le validateur réutilise la sortie du
modèle déjà produite pour la mesure d'exactitude au lieu d'exécuter une seconde
passe forward. Il s'exécute sous `no_grad` sur le modèle d'évaluation ou EMA et,
pendant un entraînement multi-GPU, est calculé localement sur le rang 0 sans
opération collective. La sélection du meilleur checkpoint reste fondée sur la
mesure d'exactitude.

Cette option ne fait volontairement pas trois choses. Elle n'inclut jamais les
termes de débruitage contrastif, car ceux-ci nécessitent la vérité terrain
pendant la passe forward et les passes de validation ne la fournissent pas.
Elle rapporte le modèle en mode évaluation. Lorsque les passes d'entraînement et
d'évaluation d'une famille diffèrent réellement, par leurs statistiques
BatchNorm ou leur profondeur stochastique, la valeur reflète donc le mode
évaluation ; c'est la comparaison voulue. Enfin, une tâche pour laquelle une
famille ne l'a pas implémentée provoque une erreur de configuration au démarrage
au lieu d'être silencieusement ignorée :

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO est l'exception pour laquelle rien ne change : son validateur calculait
déjà toujours cette loss, et `val_loss=True` modifie uniquement les clés sous
lesquelles elle est publiée.

La validation augmentée et la loss de validation ne peuvent pas être combinées.
Les demander toutes les deux provoque une erreur.

## Fichiers écrits par une validation

`val()` écrit toujours `config.yaml` dans son répertoire de sortie, qui vaut par
défaut `runs/val/<model>_<size>_<timestamp>` lorsque `save_dir` n'est pas
fourni.

<code-tabs name="json" />

`save_json=True` écrit `predictions.json` pour la détection, et
`predictions_bbox.json` ainsi que `predictions_masks.json` pour la segmentation.
OBB ne le prend pas en charge et l'indique.

`save_plots=True` écrit dans un sous-répertoire `plots/`. La détection reçoit
`box_metrics.png`, des graphiques d'AP et de rappel par classe, des courbes
précision-rappel et confiance, une matrice de confusion et des exemples
d'images annotées lorsque OpenCV est installé. La segmentation ajoute les
copies côté masque de chacun, et la pose reçoit son propre ensemble de mesures
et de courbes. Les autres validateurs n'implémentent pas les graphiques :
classification, segmentation sémantique, segmentation panoptique, profondeur,
normales, contours, restauration, matting, OCR, OBB et points n'écrivent rien à
cet emplacement. Un échec de génération de graphique émet un avertissement et
n'interrompt jamais l'exécution.

## Validation pendant l'entraînement

L'entraînement effectue une validation toutes les `eval_interval` époques sur
le split `val` du dataset, et les mesures obtenues pilotent la sélection de
`best.pt`, l'early stop défini par `patience` et les clés `val/` de chaque
logger. La validation utilise les poids EMA lorsque l'EMA est active.

Consultez les [hyperparamètres](/docs/train/hyperparameters) pour
`eval_interval`, `patience` et `save_plots`, et les
[loggers d'expériences](/docs/train/loggers) pour la destination des valeurs.

## Pages connexes

- [Datasets](/docs/train/datasets) pour les clés de splits et les formats lus
  par les validateurs.
