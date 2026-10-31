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
  - matrice de confusion
  - nms agnostique aux classes
last_verified: 1.6.0
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
  confusion:
    - label: Matrice de confusion
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")

        metrics = model.val(data="coco8.yaml")


        cm = metrics.confusion_matrix

        print(cm.matrix.shape)      # (81, 81) : 80 classes plus l'arrière-plan

        print(cm.class_accuracy())  # nom de classe : part des objets trouvés
        recevant cette classe

        tp, fp = cm.tp_fp()
  curves:
    - label: Courbes par classe
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        box = metrics.box
        print(box.px.shape, box.f1_curve.shape)  # (1000,) et (classes, 1000)
        for index, f1 in zip(box.ap_class_index, box.f1):
            print(model.names[int(index)], round(float(f1), 3))
source_hash: a5704354163337cf
---

## Exécuter une validation

`val()` accepte le dataset et renvoie les mesures.

<code-tabs name="val" />

La valeur de retour est un `dict` plat de clés de mesures vers des nombres.
Chaque clé est littérale, lisez-la donc par son nom plutôt que par sa position.
Pour la détection, la segmentation et la classification, c'est une sous-classe
de `dict` qui porte aussi des résultats par classe en attributs, décrits
ci-dessous.

Les principaux arguments sont `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` et `verbose`. `conf` vaut `0.001`
par défaut et `iou` vaut `0.6`, deux valeurs bien plus permissives que celles
de la prédiction, car un balayage mAP exige la fin à faible confiance. `imgsz`
utilise par défaut la taille d'entrée propre au modèle plutôt qu'un nombre fixe.
`split` accepte uniquement `val`, `test` ou `train`. Sans `data`, un checkpoint
entraîné est validé sur le dataset sur lequel il a été entraîné ; les poids
publiés n'en portent aucun et ont besoin de `data=`.

Tout autre champ de la configuration de validation est transmis comme argument
nommé, notamment `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`,
`cache`, `save_plots`, `single_cls` et `agnostic_nms`. Avec `half=True` sur CUDA,
la validation s'exécute sous autocast en `amp_dtype`, et les sorties bfloat16
sont converties en float32 avant le post-traitement ; l'entraînement avec
`amp_dtype="bfloat16"` valide de la même manière.

`agnostic_nms=True` ne garde que la boîte au score le plus élevé parmi celles qui
se chevauchent au-delà de `iou`, quelles que soient leurs classes. Elle
s'exécute sur les détections terminées de la famille, après `classes=`, si bien
que la mAP, la matrice de confusion, les courbes et les comptes par image
évaluent tous les mêmes boîtes. Elle couvre la détection et la segmentation, et
toute autre tâche lève `ValueError`. Le côté prédiction se trouve dans
[Seuils et filtrage](/docs/predict/thresholds).

`single_cls=True` évalue la détection comme une seule classe. La vérité terrain
est ramenée à la classe 0, et les détections du modèle sont limitées à
`classes=` dans l'espace d'étiquettes du modèle, puis réétiquetées 0, si bien
qu'un checkpoint entraîné sur plusieurs classes est évalué comme un détecteur à
une seule classe.

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
Les résultats de classification exposent aussi ces deux valeurs sous les
attributs `top1` et `top5`, à côté de `confusion_matrix`.

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

La classification ImageFolder ajoute les valeurs macro `metrics/precision`, `metrics/recall` et `metrics/f1`, moyennées sur les classes présentes dans les cibles de validation. L'exactitude top-1 reste le score de sélection par défaut. La détection renvoie aussi `metrics/best_conf` et `metrics/best_conf_f1`, en choisissant le seuil optimal du micro-F1 à IoU 0.50, et place les seuils indexés par nom de classe dans `metrics.box.best_conf_per_class`. Les détections de score égal restent groupées ; les égalités choisissent le seuil le plus élevé. Sans F1 positif, la valeur est 0.0. La segmentation n'expose pas ces clés de seuil.

## Matrice de confusion

Les résultats de détection, de segmentation et de classification portent
`results.confusion_matrix`. Elle est calculée à chaque validation, que les
graphiques soient écrits ou non, et sa `matrix` est un tableau d'entiers indexé
`matrix[predicted, true]`.

<code-tabs name="confusion" />

Pour la détection et la segmentation, les prédictions à la confiance 0.25, ou
`conf` si cette valeur est plus élevée, sont appariées une à une avec les boîtes
de vérité terrain de n'importe quelle classe à IoU 0.5, en commençant par l'IoU
le plus élevé. Une paire compte à sa classe prédite et à sa vraie classe, si bien
qu'une boîte bien placée avec la mauvaise classe tombe hors de la diagonale. La
matrice possède une ligne et une colonne `background` supplémentaires : une
vérité terrain non appariée compte dans la ligne `background`, une prédiction non
appariée dans la colonne `background`. La segmentation apparie des boîtes, pas
des masques. Avec `classes=`, la vérité terrain des autres classes est écartée.

Les régions COCO crowd (`iscrowd=1`) suivent la règle qu'applique la mAP. Une
prédiction qu'aucune vérité terrain de sa classe n'a appariée, et qui se trouve
au moins à moitié dans une région crowd de la même classe (intersection sur la
propre aire de la prédiction), est ignorée plutôt que comptée.
`box.image_metrics` et les images `visualize=True` utilisent les mêmes seuils et
la même règle crowd. Les étiquettes au format YOLO n'ont aucune région crowd.

Pour la classification, chaque image compte une fois, à sa classe top-1 face à
son étiquette, et il n'y a pas de `background`.

| Méthode | Renvoie |
|---|---|
| `summary(normalize=False)` | Un dict par classe prédite, `{"Predicted": name, <true class>: value}` ; avec `normalize`, les parts de chaque vraie classe |
| `to_csv()`, `to_json()` | Le résumé sous forme de texte CSV ou JSON |
| `to_df()` | Le résumé sous forme de DataFrame Polars ; nécessite `pip install polars` |
| `tp_fp()` | Vrais positifs par classe (la diagonale) et faux positifs (le reste de la ligne prédite) |
| `nonzero()` | Les cellules remplies sous forme de tableaux alignés `(predicted, true, count)` |
| `class_accuracy()` | Par classe, la part de ses instances recevant cette classe ; pour la détection, de ses seuls objets trouvés |
| `plot(normalize=True, save_dir="")` | Écrit `confusion_matrix_normalized.png`, ou `confusion_matrix.png` avec `normalize=False`, et renvoie le chemin |

Au-delà de 2 000 classes, la matrice de classification dense n'est pas
construite : `matrix`, `summary()`, les exports `to_*()` et `plot()` lèvent une
erreur, tandis que `tp_fp()`, `nonzero()` et `class_accuracy()` fonctionnent
toujours.

La classe est `libreyolo.validation.ConfusionMatrix`, construite comme
`ConfusionMatrix(nc, iou_thres=0.5, conf_thres=0.25, *, names=None, task="detect")`,
où `task` vaut `"detect"` ou `"classify"`.

## Précision, rappel et F1 par classe

Les résultats de détection et de segmentation portent la précision, le rappel et
le F1 de chaque classe selon les seuils de confiance, appariés à IoU 0.5.

<code-tabs name="curves" />

`box.px` contient les 1000 confiances, régulièrement espacées sur `[0, 1]`.
`box.p_curve`, `box.r_curve` et `box.f1_curve` sont des tableaux
`(classes, 1000)`, une ligne par entrée de `box.ap_class_index`, les indices de
classe qui ont une vérité terrain dans le split, par ordre croissant. `box.p`,
`box.r` et `box.f1` sont les valeurs par classe à la confiance qui maximise le F1
moyen sur ces classes. La segmentation les calcule toutes à partir des boîtes.
Ce sont des tableaux vides lorsque l'exécution n'a produit aucune courbe.

## Métriques de boîtes par image

Les résultats de détection et de segmentation restent compatibles avec un dictionnaire et exposent aussi `results.box.image_metrics`. Chaque nom de fichier est associé à `precision`, `recall`, `f1`, `tp`, `fp` et `fn` selon la règle d'appariement de la visualisation, même si celle-ci est désactivée. La segmentation compte ici les boîtes. Les noms de base en double utilisent des chemins complets après la première occurrence. Les dénominateurs nuls donnent 0.0. Ces enregistrements ne sont pas regroupés entre les rangs distribués.

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
fourni. `project`, `name` et `exist_ok` le choisissent comme le fait
l'entraînement : `project/name`, avec un suffixe incrémenté sauf si
`exist_ok=True`.

<code-tabs name="json" />

`save_json=True` écrit `predictions.json` pour la détection, et
`predictions_bbox.json` ainsi que `predictions_masks.json` pour la segmentation.
OBB ne le prend pas en charge et l'indique.

`save_plots=True`, ou son alias `plots=True`, écrit dans un sous-répertoire
`plots/` et nécessite matplotlib, qu'installe `pip install "libreyolo[plots]"`.
La détection reçoit `box_metrics.png`, des graphiques d'AP et de rappel par
classe, des courbes précision-rappel et confiance, les courbes de F1 par classe
dans `f1_conf_box.png`, la matrice de confusion sous forme de comptes dans
`confusion_matrix.png` et sous forme de parts de chaque vraie classe dans
`confusion_matrix_normalized.png`, et des exemples d'images annotées lorsque
OpenCV est installé. La segmentation ajoute des copies côté masque du graphique
de mesures, des graphiques par classe et des courbes précision-rappel et
confiance ; les courbes de F1 et la matrice de confusion restent au niveau des
boîtes. La pose reçoit son propre ensemble de mesures et de courbes. La
classification écrit les deux fichiers de matrice de confusion, sauf au-delà de
2 000 classes. Les autres validateurs n'implémentent pas les graphiques :
segmentation sémantique, segmentation panoptique, profondeur,
normales, contours, restauration, matting, OCR, OBB et points n'écrivent rien à
cet emplacement. Un échec de génération de graphique émet un avertissement et
n'interrompt jamais l'exécution.

`visualize=True` écrit des images TP/FP/FN de boîtes tenant compte des classes pour la détection et la segmentation, ou des images comparant l'étiquette à la prédiction top-1 pour la classification ImageFolder, dans `visualize/errors/` et `visualize/correct/`. L'appariement utilise IoU 0.5 et la confiance `max(0.25, conf)`. Les valeurs par défaut sont `visualize=False`, `show_labels=True` et `show_conf=True`. Les tâches non prises en charge et la validation de clips V-JEPA 2 refusent la visualisation.

`plot_samples=8` limite le tracé d'exemples distinct ; 0 le désactive et -1 conserve toutes les images. Cela ne modifie ni les métriques ni les sorties de visualisation.

## Validation pendant l'entraînement

L'entraînement effectue une validation toutes les `eval_interval` époques, et
toujours après la dernière époque, sur le split `val` du dataset, et les mesures
obtenues pilotent la sélection de `best.pt`, l'early stop défini par `patience`
et les clés `val/` de chaque logger. La validation utilise les poids EMA lorsque
l'EMA est active. Ses fichiers vont dans un répertoire `val/` à l'intérieur de
l'exécution. `val=False` désactive la validation pendant l'entraînement,
dernière époque comprise.

Consultez les [hyperparamètres](/docs/train/hyperparameters) pour
`eval_interval`, `patience` et `save_plots`, et les
[loggers d'expériences](/docs/train/loggers) pour la destination des valeurs.

## Pages connexes

- [Datasets](/docs/train/datasets) pour les clés de splits et les formats lus
  par les validateurs.
