---
title: Mettre à niveau vers la version 1.7.0
seo_title: Mettre à niveau LibreYOLO 1.6.0 vers 1.7.0
description: >-
  Étapes de migration des checkpoints, des noms de modules et des recettes
  d'entraînement YOLO9, des dossiers d'exécution de la CLI, de la matrice de
  confusion et des nouvelles erreurs dans LibreYOLO 1.7.0.
lead: >-
  La version 1.7.0 continue de charger les anciens checkpoints YOLO9 avec les
  mêmes boîtes flottantes. Elle modifie l'entraînement de la famille YOLO9, les
  noms que le code voit à l'intérieur du réseau, et quelques contrats de
  validation et de prédiction.
keywords:
  - mise à niveau libreyolo
  - migration libreyolo 1.6.0
  - changements incompatibles libreyolo
  - noms de modules yolo9
  - clip_max_norm
  - matrice de confusion
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## De 1.6.0 à 1.7.0

- Les checkpoints YOLO9, YOLO9-E2E et YOLO9-P2 enregistrés par la version 1.7.0 utilisent de nouveaux noms de tenseurs et ne se chargent pas en 1.6.x : la version 1.6.x charge ceux de YOLO9 et YOLO9-P2 avec un avertissement de clés manquantes et ne prédit aucune boîte, et lève une erreur d'incohérence `model_family` pour ceux de YOLO9-E2E. Mettez à niveau chaque environnement qui les lit. Les checkpoints flottants, quantifiés et d'entraînement écrits par les versions précédentes se chargent en 1.7.0, et les exports ONNX ne sont pas affectés.

- Le code qui accède à `model.model` d'un modèle de la famille YOLO9 voit les noms de modules de MultimediaTechLab/YOLO, sans alias. `DDetect` devient `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` devient `RepConv`, `RepNBottleneck` devient `Bottleneck`, `DFL` devient `Anchor2Vec` (`anchor2vec`), `nc` devient `num_classes`, et les sous-couches des blocs deviennent `conv1`/`conv2`/`bottleneck`. Dans YOLO9-E2E, `YOLO9E2EDetect` devient `YOLO9E2EHead` et `one2one_cv2` / `one2one_cv3` deviennent `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` et `yolo9.utils.decode_boxes` sont supprimés. Le [changelog](/docs/changelog) liste chaque renommage.

- Les noms de nœuds ONNX suivent les nouveaux noms de modules dans la tête, le backbone et le neck, par exemple `/head/anchor_convs.0/...`. Mettez à jour les outils qui s'appuient sur les noms de nœuds, comme les configurations de nœuds de fin Hailo. Les entrées, les sorties et les noms de sortie du modèle ne changent pas.

- Un fine-tuning YOLO9 à partir de poids sans tenseurs PGI, ce qui inclut les `LibreYOLO9{t,s,m,c}.pt` publiés, n'entraîne que la tête principale, comme en 1.5.0 ; la version 1.6.0 attachait une branche PGI initialisée aléatoirement avec `aux_weight=0.25`. Les poids qui contiennent des tenseurs PGI, et les runs à partir de zéro, s'entraînent toujours avec PGI. Passez `aux_weight=0.25` pour attacher malgré tout une nouvelle branche, ou `aux_weight=0` pour la désactiver de force.

- Lorsque PGI est attaché, YOLO9-m et YOLO9-c construisent la branche auxiliaire de MultimediaTechLab : 12.8M (m) et 25.7M (c) de paramètres réservés à l'entraînement avec 80 classes, contre 7.5M et 10.2M auparavant, si bien que l'entraînement PGI de ces tailles demande plus de mémoire et de temps par pas. Un checkpoint d'entraînement m/c de la version 1.6.0 reprend et fait son fine-tuning avec la branche sur laquelle il a été entraîné. L'inférence et YOLO9-t/s ne changent pas.

- YOLO9, YOLO9-E2E et YOLO9-P2 écrêtent la norme L2 du gradient à 10, et leur assignateur d'entraînement évalue les ancres en fp32, si bien que les positifs choisis sous AMP fp16 peuvent différer. Passez `clip_max_norm=0` pour retrouver l'écrêtage de la version 1.6.0 ; un run démarré en 1.6.0 et repris en 1.7.0 écrête aussi, sauf si vous le passez. L'assignation en fp32 n'a aucun interrupteur.

- Un `perspective` non nul déforme par déplacement des coins dans chaque famille qui utilise la transformation affine partagée (YOLO9 et ses variantes E2E et P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR et RT-DETRv2), y compris les runs repris depuis un checkpoint 1.6.0 qui le définissait. Les valeurs de `0.002` et plus donnent la même déformation maximale, et les valeurs négatives lèvent `ValueError`. Aucun réglage ne reproduit la déformation de la version 1.6.0 ; la valeur par défaut `perspective=0` ne change pas.

- Le pipeline mosaic de YOLO9 tire le ratio de mixup selon Beta(1, 1) au lieu de Beta(32, 32), sans réglage pour l'ancienne valeur. L'entraînement YOLOv7 par défaut change, car YOLOv7 active mixup par défaut. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR et RT-DETRv2 ne changent que lorsque mixup est activé.

- `libreyolo train` sans `name=` écrit les runs EC, D-FINE, SegFormer et LingBot-Vision dans `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` et `lingbotvision_exp` au lieu de `runs/train/exp`, les sondes V-JEPA 2 dans `runs/train/train`, et les runs YOLO9-P2 dans `runs/train/yolo9_exp`. Passez `name=exp` pour l'ancien chemin, ou `name=yolo9_p2_exp` pour YOLO9-P2.

- `libreyolo train` démarre D-FINE, SegFormer et LingBot-Vision à partir de leurs propres recettes. Pour D-FINE, cela donne 132 époques, `lr0=2e-4`, `weight_decay=1e-4`, 2 époques de warmup et aucune augmentation mosaic, mixup, HSV ou affine. Pour envoyer les valeurs que la version 1.6.0 envoyait, passez `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` indexe `matrix[predicted, true]` (c'était `[actual, predicted]`), compte les détections à `conf_thres=0.25` (c'était 0.15), prend `names=` et `task=` uniquement en arguments nommés, et son `matrix` ne peut plus être réassigné. Transposez le code qui indexait l'ancienne matrice, et passez `conf_thres=0.15` pour retrouver les anciens comptes. `plots=True` écrit les comptes dans `confusion_matrix.png` et le graphique normalisé dans `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` supprime les boîtes qui se chevauchent entre classes au lieu d'émettre un avertissement. Les tâches sans boîtes lèvent `ValueError`, `LibreEnsemble` et HRNet lèvent `NotImplementedError`, et le niveau SAM rejette le mot-clé avec `TypeError` ; retirez l'argument de ces appels.

- Un `interpolation` de classifieur inconnu lève `ValueError`, y compris les modes torchvision `LANCZOS`, `BOX`, `HAMMING` et `NEAREST_EXACT`. Utilisez `bilinear`, `bicubic`, `nearest` ou `bilinear_no_antialias`.

- Relancez la validation avant de comparer avec les chiffres de la version 1.6.0 pour `val(single_cls=True)` sur un modèle entraîné sur plusieurs classes, les datasets de segmentation qui mélangent des lignes avec boîtes seules et avec polygones, et RTMDet-Ins, qui conserve jusqu'à `max_det=300` masques par image par défaut ; passez `max_det=100` pour le réglage de MMDetection.

- Sans `libreyolo[rfdetr]`, les appels CLI avec un nom de modèle RF-DETR ou DINOv2 se terminent avec le code d'erreur JSON `model_load_failed` au lieu de `model_not_found`, et le message nomme `pip install libreyolo[rfdetr]`.

Consultez le [changelog](/docs/changelog) pour la version complète.

## De 1.5.0 à 1.6.0

- Mettez à niveau les environnements limités à une version d'ONNX Runtime inférieure à 1.18 vers `onnxruntime>=1.18.0` avant d'installer l'extra ONNX.

- SAM 3D Body accepte l'ensemble vérifié de ressources du snapshot. Installez `libreyolo[hf]`, obtenez l'accès restreint et utilisez l'acquisition automatique, ou fournissez le répertoire de snapshot inchangé et la ressource MHR épinglée sur un système de fichiers local de confiance.

- Relancez la validation après les corrections de redimensionnement RF-DETR hors pose et de normalisation des classificateurs. Revérifiez les seuils de confiance déployés avant de comparer les résultats à ceux de la version 1.5.0 ; aucune option ne restaure l'ancien prétraitement.

- D-FINE, DEIM, RT-DETRv4 et la détection YOLO-NAS activent FP16 AMP par défaut. Passez `amp=False` pour conserver FP32.

- Pour retrouver les anciens choix de fine-tuning YOLO9, définissez `aux_weight=0`, `max_labels=100` et `warmup_momentum=0.937`. Définissez `letterbox_pad="topleft"` lorsqu'une nouvelle conversion marquée center doit reproduire l'ancienne géométrie. Les anciens checkpoints à une seule tête reprennent avec leur graphe d'origine. Les nouveaux entraînements YOLO9 consomment davantage de mémoire GPU, réduisez donc `batch` si une taille de batch utilisée en 1.5.0 dépasse la mémoire disponible.

- RF-DETR et DINOv2 créent des répertoires d'exécution incrémentés portant le nom de la famille. Mettez à jour les consommateurs des chemins d'artefacts, ou définissez `output_dir="runs/train", exist_ok=True` pour conserver l'ancien emplacement et son comportement de réutilisation.

- Gardez `mixup + cutmix <= 1` pour la classification ; les combinaisons invalides lèvent désormais une erreur pendant la configuration.

- QAT désactive EMA, SyncBatchNorm et le moyennage des checkpoints. Utilisez les checkpoints best/last de QAT sans dépendre de ces états.

- Les chargeurs personnalisés avec des hooks de mutation du dataset doivent utiliser `persistent_workers=False` ou recréer les workers après la mutation. Les copies persistantes incompatibles de plusieurs workers lèvent désormais une erreur.

- Les tableaux d'images NumPy sont lus en BGR, l'ordre d'OpenCV. Passez `color_format="rgb"` là où vous transmettez un tableau RGB comme `np.asarray(pil_image)` ; la sortie de `cv2.imread()` et les images vidéo ne demandent aucun changement.

- Un tableau NumPy ou un tenseur 4D est un batch : `predict()` renvoie une liste avec un `Results` par image au lieu de n'utiliser que la première image.

- `train(resume=True)` et `train(resume="path/to/last.pt")` restaurent les arguments d'entraînement enregistrés de l'exécution et continuent d'écrire dans son répertoire d'exécution ; les arguments que vous passez explicitement l'emportent. Reprendre des poids publiés, ou une exécution qui a déjà atteint ses `epochs`, lève `ValueError`.

- Lorsque la validation est activée, la dernière époque est toujours validée, si bien que les exécutions plus courtes que `eval_interval` rapportent désormais des métriques et écrivent `best.pt`. `val=False` désactive la validation, dernière époque comprise. La validation pendant l'entraînement écrit dans `<run>/val` au lieu de `runs/val/`.

- Les échecs de la commande CLI `train` indiquent leur type d'erreur, si bien que les erreurs de configuration se terminent avec le code 2 au lieu du code 1 avec `io_error`.

Consultez le [changelog](/docs/changelog) pour la version complète et l'[importation de poids](/docs/migrate) pour la conversion de checkpoints.

## De 1.4.0 à 1.5.0

Cette page concerne la mise à niveau de LibreYOLO lui-même. Si vous cherchez à
charger un checkpoint issu d'un projet amont, consultez plutôt la page
[importer des poids existants](/docs/migrate).

L'entrée complète de cette version figure dans le
[journal des modifications](/docs/changelog). La suite présente uniquement ce
qui demande une intervention de votre part.

### Modifications de code obligatoires

#### `allow_experimental=True` n'existe plus

Le mécanisme d'acceptation a disparu, ainsi que le mécanisme
`ddp_aware(experimental_key=...)` sur lequel il reposait. L'entraînement et
l'exportation d'EC, RTMDet, PicoDet et FOMO exigeaient auparavant cet argument.
Tout script qui entraîne l'une de ces familles est donc concerné.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0 : supprimez l'argument
model.train(data="data.yaml", epochs=100)
```

Il n'existe aucune couche de compatibilité pour la dépréciation. Un appel qui
transmet encore cet argument reçoit un avertissement
`Unknown training config keys (ignored)`, et l'argument n'a aucun effet.
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` a été supprimé avec lui. Le hook
`get_download_notice()` subsiste et reste redéfini par MiDaS, SegFormer et
YOLO9-P2.

Les niveaux de prise en charge sont toujours publiés, mais ne constituent plus
un argument. Consultez les [niveaux de stabilité](/docs/reference/stability-tiers).

#### Le niveau d'exportation `"experimental"` n'existe plus

```python
from libreyolo.export.support import Tier

# 1.4.0 : Literal["validated", "experimental", "blocked"]
# 1.5.0 : Literal["validated", "available", "blocked"]
```

Le code qui bifurque en fonction de la chaîne du niveau doit lire
`"available"` là où il lisait `"experimental"`. `BaseExporter` n'émet plus de
`RuntimeWarning` pour ces formats. L'état de chaque format figure dans la
[matrice d'exportation](/docs/reference/export-matrix).

#### `pretrained=False` avec `resume` est désormais refusé

Cette combinaison produisait auparavant un comportement incohérent. Elle
déclenche maintenant :

```
ValueError: pretrained=False cannot be combined with resume.
```

Choisissez l'un ou l'autre. `pretrained=False` démarre à partir d'une nouvelle
initialisation déterministe, qui fonctionne dans la version 1.5.0 pour toutes
les familles entraînables au lieu de trois seulement. `resume` reprend une
exécution interrompue depuis son checkpoint. Les deux options sont documentées
dans la section [entraînement](/docs/train).

#### `--imgsz` de la CLI est une chaîne et non un entier

Le changement est plus limité qu'il n'y paraît. Ces deux formes restent
inchangées :

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # toujours valide
```

```python
model.predict("img.jpg", imgsz=640)   # toujours valide
```

Seul le code qui appelle directement depuis Python les fonctions de commande de
la [CLI](/docs/cli) doit changer. En effet, `predict`, `train` et `val` ont
élargi `--imgsz` de `int` à `str` afin d'accepter les tailles rectangulaires :

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, et "480x640" fonctionne désormais aussi
```

La valeur par défaut de `train` est désormais la chaîne `"640"`.
`export --imgsz` était déjà une chaîne et `profile` ne change pas.

### Nombres qui changent

Trois modifications influent sur les métriques avec les réglages par défaut. Si
vous suivez les résultats entre les versions, lisez cette section avant de
comparer une exécution 1.5.0 à une exécution 1.4.0.

#### faster-coco-eval devient le backend par défaut des métriques COCO

`val()` et la validation effectuée à chaque époque pendant l'entraînement
calculent désormais les métriques COCO avec le backend C++ faster-coco-eval au
lieu de pycocotools.

Ce choix repose sur des mesures de parité effectuées sur les 100 partitions de
test RF100-VL : 1 381 des 1 400 valeurs de métriques étaient identiques au bit
près, l'écart maximal atteignait 2,22e-16, les écarts des métriques principales
étaient exactement nuls, pour une vitesse globale 15,6 fois supérieure et
56 fois supérieure sur les datasets riches en détections. Vos nombres ne
devraient pas changer. Ils sont néanmoins produits par une autre
implémentation, raison pour laquelle ce point figure dans cette liste.

pycocotools reste la solution de repli automatique lorsque faster-coco-eval
n'est pas installé. Pour l'imposer :

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` produit le même effet globalement. Le backend
réellement utilisé est journalisé au niveau INFO, exposé dans
`model.last_eval_backend` après `val()` et inclus sous `eval_backend` dans la
charge utile JSON de la [CLI](/docs/cli/val). Installez le parcours rapide avec
`pip install libreyolo[fast-eval]`.

#### Les checkpoints YOLOX entraînés avant la version 1.5.0 nécessitent de redéfinir eps

C'est le piège de cette version. Lisez cette section si vous avez affiné
[YOLOX](/docs/models/yolox).

YOLOX définit BatchNorm avec `eps=1e-3` et `momentum=0.03`. Jusqu'à la version
1.5.0, ces valeurs étaient appliquées sous forme de correction après coup, qui
ne survivait pas à la reconstruction de la tête effectuée par `train()` lorsque
le `nc` de votre dataset différait de celui du checkpoint. Un tel fine-tuning
s'entraînait et signalait sa validation interne avec la valeur `eps=1e-5` par
défaut de torch, puis se rechargeait pour l'inférence avec `1e-3` : les mêmes
tenseurs soumis à une normalisation différente.

Les tailles à convolution ordinaire changent à peine. La taille depthwise `n`
change fortement, car sa `running_var` par canal est suffisamment faible pour
qu'eps domine. Sur la partition `ball` de RF100-VL, le même checkpoint nano
atteint une mAP50-95 de **0,566** lorsqu'il est évalué avec l'eps de son
entraînement, contre **0,151** après un rechargement standard.

Un checkpoint entraîné avant la version 1.5.0 possède la sémantique eps=1e-5.
Pour obtenir des nombres fidèles, évaluez-le en redéfinissant l'eps BN à 1e-5 :

```python
import torch
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Vous pouvez aussi intégrer une fois
`sqrt((var + 1e-3) / (var + 1e-5))` aux poids BN, puis enregistrer le
résultat. Les checkpoints entraînés avec la version 1.5.0 ou une version
ultérieure ne nécessitent aucune de ces opérations.

#### L'entraînement multi-échelle de D-FINE emploie la recette amont propre à chaque taille

`base_size_repeat` était fixé à 3 pour toutes les tailles. Il est désormais
résolu par taille, conformément à la spécification amont : **n** s'entraîne à
taille fixe sans multi-échelle, **s** vaut 20, **m** 6, **l** 4 et **x** 3.
Seule la taille x correspondait auparavant. Les tailles n, s, m et l voient donc
une distribution d'échelles différente et convergent vers d'autres métriques.

Pour rétablir l'ancien comportement, définissez-le explicitement :

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM conserve la valeur 3 codée en dur. Les détails de la famille figurent sur
la page [D-FINE](/docs/models/d-fine).

### À connaître, sans intervention nécessaire

- **Les résultats avec un `imgsz` rectangulaire ont changé parce qu'ils étaient
  auparavant incorrects.** Les coordonnées des boîtes, le redimensionnement des
  masques RTMDet, le changement d'échelle de YOLO-NAS et celui des vérités
  terrain du validateur utilisent maintenant la hauteur et la largeur propres à
  chaque axe, et non une valeur scalaire unique. Avec un `imgsz` carré, rien ne
  change au bit près. L'inférence ou la validation rectangulaire était mal mise
  à l'échelle dans la version 1.4.0. YOLO-NAS refuse désormais un `imgsz`
  rectangulaire au lieu de produire silencieusement une sortie incorrecte.
- **Les dictionnaires de métriques contiennent de nouvelles clés.** Il s'agit
  de `max_det`, `ar_max_det` et `AR_max_det` provenant de l'évaluateur COCO,
  ainsi que de `metrics/loss` et `metrics/loss/ce` pour FOMO. Les valeurs avec
  les réglages par défaut ne changent pas, mais tout code qui parcourt les clés
  des métriques, y compris les [systèmes de journalisation](/docs/train/loggers)
  personnalisés et les en-têtes CSV, voit de nouvelles colonnes.
- **Les exécutions YOLO9 déterministes qui reconstruisent une tête** commencent
  depuis une initialisation différente, car la graine est désormais appliquée
  avant la reconstruction plutôt qu'après. Un fine-tuning déterministe réalisé
  avec la version 1.4.0 sur un nombre de classes différent ne peut pas être
  reproduit bit à bit avec la version 1.5.0.
- **`libreyolo[hub-kernels]` sur CUDA active désormais réellement le kernel
  MS-deform-attn natif.** La version 1.4.0 le plaçait derrière une condition
  que RF-DETR n'empruntait jamais, de sorte que le kernel ne s'exécutait pas.
  Les prédictions de RF-DETR et des autres familles à attention déformable
  peuvent changer dans les limites de tolérance des nombres flottants. Les
  installations standard ne sont pas concernées et
  `LIBREYOLO_HUB_KERNELS=0` le désactive.
- **`libreyolo predict` ignore les options non prises en charge au lieu de
  déclencher une erreur.** La CLI filtre les arguments nommés selon la signature
  `__call__` du modèle. Une option que la famille n'accepte pas est ainsi
  ignorée au lieu de déclencher une `TypeError`. Un nom d'option
  inconnu est toujours rejeté avec `No such option`.
- **Les sources en direct modifient la forme de la sortie JSON.** Les webcams,
  les flux RTSP et la capture d'écran activent implicitement le streaming, qui
  émet un enregistrement par image plutôt qu'un seul pour l'appel. Ces
  [sources](/docs/predict/sources) sont nouvelles dans la version 1.5.0. Aucun
  script 1.4.0 n'est donc concerné.
- **Réexporter `rfdetr-pose` ou `yolonas-pose` vers ONNX produit des noms de
  sorties différents.** La version 1.4.0 interprétait à tort leurs têtes de pose
  à plusieurs tenseurs comme de la segmentation, au moyen d'une heuristique
  fondée sur le nombre de sorties. Les fichiers `.onnx` existants sur le disque
  ne changent pas.
- **Dans une installation sans torch**, les résultats contiennent des tableaux
  numpy plutôt que des `torch.Tensor`. `.boxes.data` renvoie donc un autre type
  et le départage des égalités par la NMS peut différer de torchvision. Lorsque
  torch est installé, le comportement reste identique octet par octet.
  Consultez l'[installation légère](/docs/lightweight-install).
- **Les objets de configuration sont davantage validés à leur construction.**
  `TrainConfig` possède désormais une méthode `__post_init__` dont il était
  auparavant dépourvu. Une configuration déjà non valide déclenche donc une
  erreur immédiatement au lieu d'échouer tard pendant une exécution. La
  sérialisation de `ValidationConfig` contient une nouvelle clé
  `edge_thresholds`, qui empêche un aller-retour strict
  `ValidationConfig(**dump)` depuis un export de la version 1.4.0.
- **Les noms de fichiers de poids des familles dotées d'un suffixe de tâche sont
  résolus différemment.** `segformer-b0` se résout désormais en
  `LibreSegformerb0-sem.pt`. Cette correction élimine les erreurs 404 de
  téléchargement automatique, mais rompt tout script qui avait codé en dur
  l'ancien nom sans suffixe.
- **Le marqueur pytest `experimental_backend` s'appelle désormais
  `extended_backend`.** Ce point ne vous concerne que si vous exécutez la suite
  de tests avec `-m`.

### Checkpoints et datasets

Les checkpoints écrits par la version 1.4.0 se chargent sans modification. Le
[schéma](/docs/reference/checkpoint-schema) contient désormais `imgsz_h` et
`imgsz_w` pour les modèles rectangulaires, et écrit toujours la valeur scalaire
`imgsz = max(h, w)` pour les anciens lecteurs. Les exportations
[ExecuTorch](/docs/export/executorch) et [MNN](/docs/export/mnn) nécessitent
maintenant un fichier annexe, respectivement `<program>.pte.json` et
`<model>.mnn.json`, tandis que les exportations HRNet contiennent
`pose_input: "person_crop"`. Les formats de datasets ne changent pas.
