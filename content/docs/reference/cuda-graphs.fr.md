---
title: Graphes CUDA
seo_title: Matrice de prise en charge des graphes CUDA par LibreYOLO
description: >-
  Familles qui capturent leur passe forward pendant la prédiction et leurs
  passes forward et backward pendant l'entraînement, garanties numériques,
  points de séparation d'une capture et raison pour laquelle une famille non
  compatible lève une erreur.
lead: >-
  Un graphe CUDA enregistre une exécution d'une séquence fixe de kernels et la
  rejoue en un lancement unique. LibreYOLO capture l'inférence sur 39 familles
  vérifiées et l'entraînement sur 24, toujours famille par famille, toujours
  après un contrôle de parité bit à bit et jamais comme repli silencieux.
keywords:
  - graphe cuda libreyolo
  - cuda_graph=True
  - matrice support cuda graph
  - entraînement torch cuda graph
  - capture_error_mode thread_local
  - cuda graph identique bits
last_verified: 1.5.0
verification: >-
  Liste des familles d'inférence dérivée de la matrice CAPTURABLE dans
  tests/e2e/test_cuda_graph_families.py en v1.5.0. Liste des familles
  d'entraînement, classes de parité et mesures lues dans
  docs/training_cuda_graphs.md. API et NotImplementedError lus dans
  BaseModel._require_cuda_graph_support, cuda_graph_scope et capture_graph dans
  libreyolo/models/base/model.py, avec la variable de classe
  SUPPORTS_CUDA_GRAPH. Séparations aux jointures lues dans les redéfinitions de
  _get_graph_runner des familles depth_anything3, birefnet, ppocr, sam et
  sensenova, ainsi que dans libreyolo/models/base/detr_cuda_graph.py.
  capture_error_mode lu dans libreyolo/models/base/cuda_graph.py et
  libreyolo/training/cuda_graph.py. Repli de l'entraînement lu dans
  libreyolo/training/trainer.py et option --cuda-graph lue dans
  libreyolo/cli/commands/train.py.
meta:
  - label: Familles d'inférence
    value: '39'
  - label: Familles d'entraînement
    value: '24'
  - label: Paramètre d'inférence
    value: predict(cuda_graph=True)
    mono: true
  - label: Paramètre d'entraînement
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Prédire
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # True capture à la première utilisation de chaque forme d'entrée.

        # "auto" attend qu'une forme se répète avant de payer le coût de
        capture.

        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Entraîner
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Entraîner depuis le CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Éléments capturés

Un graphe enregistre une séquence fixe de kernels et les adresses mémoire
qu'ils lisent et écrivent. Il n'enregistre ni les valeurs, ni les formes, ni le
flux de contrôle. Le rejeu demande un seul lancement au lieu de centaines. Le
gain est donc maximal sur les petits réseaux et les petits batchs, où le coût
d'une étape provient surtout du lancement plutôt que des calculs.

Les deux points d'entrée capturent des quantités de travail différentes.

| | Dans le graphe | Eager |
|---|---|---|
| Inférence | Passe forward du réseau, `model._forward(x)` | Prétraitement, NMS, tout le post-traitement |
| Entraînement | Passes forward et backward du réseau | Loss, étape de l'optimiseur, écrêtage des gradients, EMA, planning du learning rate |

Ni la NMS ni la loss de détection ne peuvent être capturées. Toutes deux
sélectionnent avec des masques booléens, exécutent un appariement hongrois ou
un assigner et bifurquent selon le résultat, soit exactement ce qu'un graphe
ne peut pas enregistrer. Les maintenir hors du graphe garantit la sécurité de
la capture et ne constitue pas une limitation à contourner.

<code-tabs name="usage" />

`cuda_graph` accepte trois valeurs lors de la prédiction. `False` est la valeur
par défaut. `True` capture la première occurrence de chaque forme d'entrée.
`"auto"` attend qu'une forme se répète, les tâches ponctuelles ou aux formes
variables ne paient donc jamais le coût d'une capture inutilisée.
`capture_graph(imgsz=None, batch=1, dtype=None)` retire ce coût de la première
requête, `graph_info()` rapporte les graphes capturés et les nombres de rejeux,
et `release_graphs()` les libère.

Pendant l'entraînement, le paramètre est un simple booléen et se nomme
`--cuda-graph` dans le CLI. Consultez les
[performances de prédiction](/docs/predict/performance) et les
[performances d'entraînement](/docs/train/performance) pour les réglages
connexes.

## Prise en charge de l'inférence

La prise en charge est définie par famille au moyen de la variable de classe
`SUPPORTS_CUDA_GRAPH`. Une famille n'est marquée compatible qu'après une
capture et un rejeu identiques bit à bit sur deux entrées de test issues de
distributions différentes. Cette matrice de parité partagée couvre 39 familles
et neuf tâches.

| Tâche | Familles |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Plusieurs familles apparaissent sous plusieurs tâches, la matrice exécute donc
plus de lignes qu'elle ne contient de familles distinctes. Trois autres
familles effectuent une capture par des chemins propres à la famille avec
leurs tests dédiés plutôt que par la matrice partagée, et ne font pas partie
des 39\u00a0: PP-OCR, SAM et SenseNova.

La vérification est bit à bit, et non approximative. Une ancienne version du
protocole évaluait la parité selon l'amplitude relative et avait rétrogradé à
tort trois familles saines, YOLOX, EfficientNetV2 et YOLOv7, dont la différence
entre eager et graphe mesure environ 1e-7 tout en restant identique bit à bit
sur l'entrée de test pertinente.

## Prise en charge de l'entraînement

Dans cette version, la capture de l'entraînement est passée de deux à 24
familles, réparties entre cinq tâches.

| Tâche | Familles |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Tout le reste s'entraîne en mode eager\u00a0: les autres tâches de ces mêmes
familles, les familles absentes de la liste, les exécutions distribuées et les
exécutions de distillation. La capture est aussi omise tant qu'une forme est
nouvelle. Le chemin d'entraînement attend qu'une forme d'entrée se répète trois
fois avant de la capturer, `multi_scale=True` peut donc ne jamais capturer.

## Deux réponses pour une famille non compatible

Le chemin d'inférence lève une erreur. `predict(cuda_graph=True)` sur une
famille qui n'a pas activé la fonctionnalité lève `NotImplementedError` en
nommant la famille, au lieu d'exécuter le mode eager et de vous laisser croire
à un gain inexistant. Une mauvaise capture n'échoue pas bruyamment\u00a0: le rejeu
d'une passe forward contenant une opération non capturable renvoie
silencieusement des valeurs erronées. La prise en charge doit donc être une
affirmation explicite par famille et non une tentative avec repli.

Le chemin d'entraînement consigne un message. `train(cuda_graph=True)` peut
toujours être transmis sans risque. Une famille, tâche ou configuration non
capturable écrit une ligne et s'entraîne en mode eager sans autre changement.
Une capture qui échoue en cours d'exécution fait aussi passer tout le reste de
l'exécution en mode eager au lieu de l'interrompre. Cette asymétrie est
délibérée\u00a0: un appel de prédiction peut être corrigé sur son site d'appel,
tandis qu'une exécution d'entraînement ne doit pas mourir à la sixième heure à
cause d'une optimisation facultative.

## Séparation aux jointures

Certaines familles ne peuvent pas être capturées entièrement, car une étape
effectue réellement une opération impossible à enregistrer dans un graphe.
Plutôt que de supprimer la famille, la capture est divisée à une jointure
vérifiée\u00a0: la partie capturable est rejouée, le reste s'exécute en mode eager,
et la sortie combinée est identique à une exécution entièrement eager.

| Famille | Capturé | Eager et raison |
|---|---|---|
| Depth Anything 3 | Le réseau | Étape du ciel, travail visible par l'hôte après la passe forward |
| BiRefNet | Encodeur, `forward_enc` | Décodeur, dont `deform_conv2d` produit un résultat différent lors du rejeu |
| PP-OCR | Étape de détection, `forward_det` | Reconnaissance, car la largeur des recadrages varie par ligne |
| SAM | Encodeur d'image | Chemin des prompts, exécuté plusieurs fois par encodage |
| SenseNova | Tour de vision | Génération autorégressive, avec un cache KV qui grandit à chaque étape |
| Détecteurs encodeur-décodeur | Backbone et encodeur | Décodeur et critère hongrois |

La séparation de BiRefNet mérite une attention particulière\u00a0: le mauvais
comportement de `deform_conv2d` sous capture se reproduit sur un appel isolé
hors de tout modèle. Son remplacement par un équivalent PyTorch pur a été
refusé, car il aurait aussi modifié les prédictions eager, or les valeurs eager
constituent le contrat.

Le cas encodeur-décodeur couvre D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 et EC. Leur décodeur construit des requêtes de débruitage contrastif
depuis la vérité terrain, dont le nombre provient du plus grand nombre
d'éléments de vérité terrain dans le batch. Le nombre de tokens du décodeur
change donc d'un batch à l'autre, ce qu'un graphe ne peut pas tolérer. Le
backbone et l'encodeur représentent environ un cinquième à un quart d'une étape
pour ces familles, d'où leur position en bas du tableau des accélérations.

PP-OCR capture un graphe par forme d'entrée de détection, dans la limite du
cache du runner, et renvoie le résultat eager lorsqu'aucune portée de capture
n'est active.

## Calculs numériques

La plupart des familles sont identiques bit à bit. Dans les autres cas, la
raison est nommée explicitement. À l'étape zéro de l'entraînement, la loss est
identique bit à bit pour les 24 familles et aucun buffer BatchNorm ne diffère.
La comparaison des gradients sépare les catégories.

| Classe | Familles | Signification |
|---|---|---|
| Exact | La plupart des 24 | Chaque gradient est identique bit à bit |
| 1 ULP | fomo, lingbotvision | Dernier bit du float32, environ 1e-7 relatif, dû à un ordre de sommation différent |
| Bruit eager | Lignée DETR | L'écart entre graphe et eager ne dépasse pas celui entre deux exécutions eager |
| Arrondi des flottants | rtmdet | 137 gradients sur 139 identiques bit à bit, deux diffèrent d'environ 3e-4 |
| Flux RNG propre | segformer | La profondeur stochastique se trouve dans la région capturée |

La classe de bruit eager doit être interprétée correctement. Pour ces familles,
deux exécutions eager ayant la même seed divergent déjà. L'égalité bit à bit
n'est donc pas une exigence manquée par l'exécution avec graphe, mais une
exigence qu'aucune exécution ne satisfait. Le phénomène est plus large avec
`amp=False`, où une non-déterminisme relatif mesuré à 3.2e-7 dans un gradient
de poids fp32 s'accumule\u00a0: deux exécutions eager YOLOv9-t avec la même seed
divergent de 36\u00a0% en 20 étapes, et la désactivation de TF32 ne corrige pas le
problème.

## Mémoire épinglée

La capture utilise `capture_error_mode="thread_local"`. Avec le mode
`"global"` par défaut de PyTorch, le thread de mémoire épinglée d'un DataLoader
qui prépare le batch suivant appelle `cudaHostAlloc`. Cela invalide la capture
en cours et empoisonne le thread, si bien que l'exécution meurt lors de la
récupération du batch suivant avec une erreur levée dans le thread. Cette
association a été observée deux fois lors d'une véritable campagne
d'entraînement avant son diagnostic.

Le mode thread-local ne limite que le thread de capture. Le thread de mémoire
épinglée ne touche jamais le flux de capture, rien de ce qu'il effectue
n'appartient donc au graphe. L'entraînement va plus loin et remplace
temporairement `torch.cuda.CUDAGraph` par une sous-classe qui impose ce mode,
car `make_graphed_callables` n'expose aucun argument correspondant. Le
remplacement est protégé par un verrou afin que deux captures simultanées ne
puissent pas le laisser installé.

## Gain obtenu

Mesures effectuées sur une RTX 5070 Ti sous AMP, un processus par branche, avec
rejeu d'un batch réel pour exclure le dataloader, et meilleur temps parmi 24
étapes après warmup. Détection à 640\u00a0px, classification à 224\u00a0px.

| Famille | Batch | Accélération |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Tout le reste | variable | 1.04x à 1.26x |

Le gain d'une exécution complète est inférieur, car un graphe ne peut accélérer
ni le dataloader ni la validation. Un fine-tuning de YOLOv9-t pendant 20
époques sur 406 images est passé de 428.4\u00a0s à 367.7\u00a0s, soit un gain de 1.16x,
avec une mAP50-95 identique de 0.6394 dans les deux branches et des loss
identiques à chaque époque.

La limite dépend de la part d'une étape consacrée au réseau. Sur le même
matériel à 640\u00a0px avec un batch de 8, elle atteint 84\u00a0% pour YOLOv9-t mais
seulement 26\u00a0% pour RTMDet-t, qui consacre l'essentiel de l'étape à son
assigner d'étiquettes. Le coût de lancement est maximal sous Windows, les gains
sous Linux atteignent donc environ un tiers à la moitié de ce tableau. Une
exécution limitée par le dataloader ne change pas du tout en temps réel. Le pic
de mémoire varie entre une baisse de 5\u00a0% et une hausse de 19\u00a0%.

## Précautions

Un graphe enregistre des adresses et non des valeurs. Toute opération qui
déplace les paramètres le supprime. Le changement d'appareil par
`predict(device=...)`, la quantification et la déquantification invalident
tous les graphes capturés.

La taille de batch compte davantage que la famille\u00a0: RT-DETR-r18 gagne 1.19x
avec un batch de 2 et 1.04x avec un batch de 8, car un grand batch est limité
par les calculs et offre moins de coût de lancement à supprimer.

La suite de parité d'inférence a été exécutée sans le package facultatif
`kernels`. Elle ne couvre donc pas la sécurité de capture lorsque des kernels
compilés du Hub sont actifs. Définissez `LIBREYOLO_HUB_KERNELS=0` pour les
écarter pendant l'analyse d'un problème de capture. Consultez la page sur les
[kernels](/docs/reference/kernels).
