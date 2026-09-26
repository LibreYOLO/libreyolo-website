---
title: Performances d'entraînement
seo_title: 'Entraînement plus rapide : graphes CUDA, AMP, profileur'
description: >-
  Accélérez un entraînement : capturez l'étape dans des graphes CUDA, choisissez
  un type AMP et utilisez le profileur intégré pour trouver où le temps est
  réellement dépensé.
lead: >-
  Trois leviers modifient la vitesse d'une étape d'entraînement : la précision
  mixte, la capture des passes forward et backward du réseau dans un graphe
  CUDA, et ce qui, selon le profileur, ralentit réellement l'étape.
keywords:
  - entraînement graphes cuda
  - accélérer entraînement
  - entraînement précision mixte
  - entraînement bfloat16
  - profileur pytorch
  - goulot dataloader
  - latence lancement kernels
  - utilisation gpu
last_verified: 1.5.0
snippets:
  profile:
    - label: Profiler puis poursuivre l'entraînement
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Profile une courte fenêtre d'étapes réelles, affiche un verdict, puis
        # poursuit l'exécution après avoir retiré les hooks.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 'Mesurer uniquement, puis arrêter'
      language: bash
      code: >
        # Définit no_aug_epochs=0 et exécute juste assez d'époques pour remplir
        la fenêtre.

        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Examiner le résultat en détail
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Mesurer avant toute modification

Les trois leviers ci-dessous corrigent des problèmes différents, et appliquer
le mauvais ne change rien. Le profileur indique lequel vous rencontrez.

<code-tabs name="profile" />

`profile=True` mesure une fenêtre d'étapes d'entraînement réelles, cinq écartées
puis vingt mesurées par défaut, affiche un rapport, écrit ses artefacts, puis
poursuit l'entraînement après avoir retiré les hooks. Cette option ne coûte rien
lorsqu'elle est désactivée et est ignorée pendant l'entraînement distribué.

Le rapport se termine par l'un des quatre verdicts suivants :

| Verdict | Signification | Leviers |
|---|---|---|
| `dataloader` | le GPU attend les données d'entrée | plus de `workers`, `cache="ram"` ou `"disk"`, augmentation plus légère, batch plus grand |
| `host / launch` | le GPU est alimenté trop lentement, nombreux petits kernels | batch plus grand, graphes CUDA, moins de synchronisations hôte par étape |
| `compute` | le GPU est saturé | AMP ou bfloat16, ou l'accepter |
| `memory-pressure` | sollicitation excessive de l'allocateur, VRAM à la limite | batch plus petit ; les valeurs d'utilisation ne sont pas fiables ici |

La valeur d'utilisation est le temps d'activité des kernels divisé par le temps
d'étape non synchronisé. La fenêtre est volontairement divisée : la première
moitié s'exécute sans synchronisation supplémentaire afin que le verdict reflète
le vrai chevauchement, et seule la seconde moitié encadre chaque phase d'une
synchronisation pour attribuer le temps GPU. Synchroniser chaque phase donne du
répit aux workers du dataloader et masque le manque de données. Les valeurs de
composition ne servent donc jamais à choisir le verdict.

Quatre fichiers sont placés dans le répertoire d'exécution : `timeline.html`,
qui s'ouvre seul dans un navigateur, `profile_trace.json` pour Perfetto ou
Nsight, `profile_summary.json` et `profile.json`, le fichier autonome à copier
et à retransmettre aux sous-commandes `libreyolo profile`.

Deux caractéristiques de `profile run` sont à connaître. Il définit
`no_aug_epochs=0`, car le profileur mesure l'époque 0 et une courte exécution
avec la valeur `no_aug_epochs` par défaut profilerait le dataloader plus léger
sans augmentation plutôt que celui réellement utilisé par l'entraînement. De
plus, `--repeat N` rapporte la moyenne et l'écart-type, ce qui importe car une
étape limitée par les lancements est assez bruitée pour qu'une seule exécution
induise en erreur. L'option écrit les répertoires de chaque essai `prof_1`,
`prof_2`, etc., ainsi qu'un fichier agrégé `profile_repeat.json`.

## Précision mixte

`amp=True` est la valeur par défaut pour la plupart des familles et exécute la
passe forward sous autocast CUDA. `amp_dtype` choisit `float16` ou `bfloat16`.

<code-tabs name="amp" />

Float16 nécessite une mise à l'échelle dynamique de la loss et reçoit un
gradient scaler actif ; la plage d'exposants plus large de bfloat16 la rend
inutile, son scaler est donc désactivé. Quatre familles sont fournies avec
`amp=False` : D-FINE, DEIM, YOLO-NAS et FOMO, et le paramètre DEIM est hérité
par RT-DETRv4. D-FINE en précise la raison : son décodeur borne les activations
à 65504, la plus grande valeur float16 finie.

La sémantique des arguments, notamment le comportement d'une demande bfloat16
sur du matériel qui ne le prend pas en charge, figure dans les
[hyperparamètres](/docs/train/hyperparameters).

## Graphes CUDA

`cuda_graph=True` capture les passes forward et backward d'entraînement du
réseau dans un graphe CUDA, supprimant la surcharge de lancement des kernels à
chaque étape.

<code-tabs name="graph" />

Le flag peut toujours être passé sans risque. Une famille, une tâche ou une
configuration impossible à capturer journalise une ligne et s'entraîne en mode
eager, sans autre changement.

Seul le réseau est capturé. La loss reste volontairement en mode eager, car les
loss de détection effectuent des sélections avec des masques booléens, exécutent
un appariement hongrois et bifurquent selon les résultats d'affectation, ce
qu'un graphe ne peut pas enregistrer. L'étape de l'optimiseur, l'écrêtage des
gradients, la mise à jour EMA et le schedule du learning rate restent aussi en
mode eager.

Cela limite le gain à la part du réseau dans une étape, qui varie beaucoup.
Mesurée sur une RTX 5070 Ti à 640 px avec un batch de 8, cette part est de 84 %
pour une étape YOLOv9-t, 44 % pour YOLOv7-b, 31 % pour YOLOX-t et 26 % pour
RTMDet-t. Les deux derniers passent la majeure partie d'une étape dans leurs
mécanismes d'affectation d'étiquettes. La capture du réseau leur profite donc
le moins.

### Gain obtenu

Conditions de toutes les valeurs ci-dessous : RTX 5070 Ti, Windows, AMP, un
processus par branche depuis un état enregistré partagé, avec rejeu d'un vrai
batch afin d'exclure le dataloader, et la plus rapide de 24 étapes après warmup.
Détection à 640 px, classification à 224 px. La taille de batch est indiquée par
ligne.

| Famille | Taille | Batch | Eager | Avec graphe | Accélération |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Ces valeurs isolent l'étape GPU. Un fine-tuning complet paie aussi le coût du
dataloader et de la validation. Sur la même machine, YOLOv9-t sur un dataset de
détection de 406 images, 20 époques, batch 8, 640 px et 4 workers de dataloader
a pris 428.4 s en temps réel en mode eager contre 367.7 s avec graphe, soit un
gain de 1.16x, avec une mAP50-95 de 0.6394 dans les deux branches.

Trois facteurs modifient ces valeurs. Les petits batchs sont limités par les
lancements et les grands par le calcul. RT-DETR-r18 gagne donc 1.19x avec un
batch de 2 et 1.04x avec un batch de 8. La surcharge de lancement est la plus
élevée sous Windows, et les gains sous Linux représentent environ un tiers à
la moitié de ceux de la table. Enfin, une exécution limitée par le dataloader
ne voit aucun changement du temps réel, d'où l'importance de commencer par le
profileur.

La capture s'active de la même façon avec `amp=False`, mais les kernels fp32
s'exécutent plus longtemps. Une étape est donc moins limitée par les lancements
et la plupart des familles gagnent moins. Sur le même matériel, MobileNetV4-s
avec un batch de 16 passe d'un gain de 2.74x sous AMP à 3.61x en fp32, tandis
que YOLOv9-t avec un batch de 8 passe de 1.99x à 1.69x et RT-DETR-r18 avec un
batch de 4 de 1.12x à 0.99x.

### Périmètre de la capture

| Tâche | Familles |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Tout le reste revient au mode eager avec une seule ligne de journal : les autres
tâches de ces familles, les familles non listées, les exécutions distribuées et
les exécutions avec distillation. Un échec de capture pendant l'exécution fait
aussi revenir le reste de celle-ci au mode eager au lieu de la faire échouer.

Pour les détecteurs encodeur-décodeur D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 et
v4, et EC, seuls le backbone et l'encodeur sont capturés. Leur décodeur lit la
vérité terrain pour construire des requêtes de débruitage contrastif, et le
nombre de ces requêtes suit le plus grand nombre de vérités terrain du batch.
Le nombre de tokens varie donc d'un batch à l'autre.

### Formes

Un graphe n'est valide que pour la forme d'entrée avec laquelle il a été
capturé. Le trainer compte les formes de batch et effectue une capture après
trois répétitions d'une forme. Les batchs de toute autre forme s'exécutent en
mode eager : batchs multi-échelles et dernier batch partiel d'une époque.

C'est le piège des familles DETR, qui redimensionnent chaque batch par défaut.
Avec `multi_scale=True`, une courte exécution peut ne jamais rencontrer une
forme assez souvent pour la capturer. Passez `multi_scale=False` lorsque
l'accélération est l'objectif.

YOLOX modifie ce que calcule la région capturée au cours d'une exécution en
activant sa branche de régression L1 lorsque la mosaïque se termine à
`no_aug_epochs`. Le trainer invalide alors la capture et en effectue une nouvelle
une fois la nouvelle forme stabilisée.

### Résultats numériques et mémoire

La plupart des familles reproduisent leur trajectoire de loss eager bit pour bit
sous AMP. FOMO et LingBot-Vision diffèrent sur le dernier bit float32 à cause
d'un ordre de sommation différent. Les détecteurs à attention déformable D-FINE,
DEIM, DEIMv2, RT-DETR, RF-DETR et EC ne reproduisent pas non plus leurs propres
exécutions eager, car leur passe backward accumule avec des opérations atomiques
et les convolutions TF32 choisissent un ordre de réduction à chaque lancement ;
l'exécution avec graphe reste dans cette dispersion. RTMDet présente une
différence relative d'environ 3e-4 sur deux gradients parmi 139, car il partage
les convolutions de tête entre les niveaux de pyramide et les deux chemins
backward additionnent trois contributions dans un ordre différent. SegFormer
possède une profondeur stochastique dans la région capturée. Un graphe rejoué
tire donc son propre flux aléatoire et est statistiquement équivalent au mode
eager plutôt qu'identique ; le gestionnaire le journalise une fois au moment de
la capture.

Avec `amp=False`, une identité bit pour bit n'est possible pour rien sur ce
matériel, avec ou sans capture. Deux exécutions eager identiques de YOLOv9-t avec
la même seed divergent de 36 % en valeur relative sur 20 étapes, et celles de
YOLOX-t de 2.6 %, car cuDNN choisit un algorithme non déterministe pour le
gradient des poids de certaines formes de convolution fp32.

Un graphe capturé épingle des buffers statiques d'entrée, de sortie et d'espace
de travail. Le pic de VRAM augmente donc d'environ un jeu d'activations
supplémentaire. Pour les familles ci-dessus, le pic d'allocation a varié de
-5 à +19 %. Le coût relatif est le plus élevé pour les petits modèles de
classification, dont les activations sont déjà petites : ResNet-18 à 224 px,
batch 16, est passé de 0.48 Go en mode eager à 0.57 Go avec graphe. Si cette
hausse fait dépasser la limite à une exécution, réduisez le batch ou laissez le
flag désactivé.

## Pages connexes

- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `nbs`, `cache` et
  `workers`.
- [Entraînement multi-GPU](/docs/train/multi-gpu), où les graphes CUDA et le
  profileur sont indisponibles.
- [Graphes CUDA](/docs/reference/cuda-graphs) pour la matrice combinée de prise
  en charge de l'inférence et de l'entraînement, les séparations des segments et
  le contrat numérique.
