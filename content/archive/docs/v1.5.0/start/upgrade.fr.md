---
title: Mettre à niveau vers la version 1.5.0
seo_title: Mettre à niveau LibreYOLO 1.4.0 vers 1.5.0
description: >-
  Les quatre modifications de code exigées par la version 1.5.0, les trois
  changements qui modifient les métriques et les évolutions de comportement à
  connaître avant de comparer les exécutions.
lead: >-
  Aucun élément n'a été retiré de l'API publique des modèles : toutes les
  classes et fonctions utilisables avec la version 1.4.0 s'importent encore.
  Quatre arguments ont changé de forme et trois valeurs par défaut modifient les
  nombres que vous pourriez comparer.
keywords:
  - mise à niveau libreyolo
  - migration libreyolo 1.5.0
  - suppression allow_experimental
  - changements incompatibles libreyolo
  - yolox bn eps
  - faster coco eval défaut
last_verified: 1.5.0
meta:
  - label: S'applique à
    value: 1.4.0 vers 1.5.0
  - label: Modifications de code requises
    value: 'Quatre, toutes ciblées'
  - label: Résultats modifiés
    value: 'Backend COCO, eps BN de YOLOX, multi-échelle de D-FINE'
  - label: Suppressions de l'API publique
    value: Aucune
source_hash: ab38d8ef7b53f596
---

Cette page concerne la mise à niveau de LibreYOLO lui-même. Si vous cherchez à
charger un checkpoint issu d'un projet amont, consultez plutôt la page
[importer des poids existants](/docs/migrate).

L'entrée complète de cette version figure dans le
[journal des modifications](/docs/changelog). La suite présente uniquement ce
qui demande une intervention de votre part.

## Modifications de code obligatoires

### `allow_experimental=True` n'existe plus

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
transmet encore cet argument déclenche une `TypeError`.
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` a été supprimé avec lui. Le hook
`get_download_notice()` subsiste et reste redéfini par MiDaS, SegFormer et
YOLO9-P2.

Les niveaux de prise en charge sont toujours publiés, mais ne constituent plus
un argument. Consultez les [niveaux de stabilité](/docs/reference/stability-tiers).

### Le niveau d'exportation `"experimental"` n'existe plus

```python
from libreyolo.export.support import Tier

# 1.4.0 : Literal["validated", "experimental", "blocked"]
# 1.5.0 : Literal["validated", "available", "blocked"]
```

Le code qui bifurque en fonction de la chaîne du niveau doit lire
`"available"` là où il lisait `"experimental"`. `BaseExporter` n'émet plus de
`RuntimeWarning` pour ces formats. L'état de chaque format figure dans la
[matrice d'exportation](/docs/reference/export-matrix).

### `pretrained=False` avec `resume` est désormais refusé

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

### `--imgsz` de la CLI est une chaîne et non un entier

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

## Nombres qui changent

Trois modifications influent sur les métriques avec les réglages par défaut. Si
vous suivez les résultats entre les versions, lisez cette section avant de
comparer une exécution 1.5.0 à une exécution 1.4.0.

### faster-coco-eval devient le backend par défaut des métriques COCO

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

### Les checkpoints YOLOX entraînés avant la version 1.5.0 nécessitent de redéfinir eps

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
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Vous pouvez aussi intégrer une fois
`sqrt((var + 1e-3) / (var + 1e-5))` aux poids BN, puis enregistrer le
résultat. Les checkpoints entraînés avec la version 1.5.0 ou une version
ultérieure ne nécessitent aucune de ces opérations.

### L'entraînement multi-échelle de D-FINE emploie la recette amont propre à chaque taille

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

## À connaître, sans intervention nécessaire

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
  ignorée au lieu de déclencher une `TypeError`. Une faute de frappe dans le
  nom d'une option est maintenant ignorée silencieusement.
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

## Checkpoints et datasets

Les checkpoints écrits par la version 1.4.0 se chargent sans modification. Le
[schéma](/docs/reference/checkpoint-schema) contient désormais `imgsz_h` et
`imgsz_w` pour les modèles rectangulaires, et écrit toujours la valeur scalaire
`imgsz = max(h, w)` pour les anciens lecteurs. Les exportations
[ExecuTorch](/docs/export/executorch) et [MNN](/docs/export/mnn) nécessitent
maintenant un fichier annexe, respectivement `<program>.pte.json` et
`<model>.mnn.json`, tandis que les exportations HRNet contiennent
`pose_input: "person_crop"`. Les formats de datasets ne changent pas.
