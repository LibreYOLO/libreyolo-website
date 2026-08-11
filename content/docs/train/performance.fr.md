---
title: Performances d'entraînement
seo_title: 'Accélérer l''entraînement : graphes CUDA, AMP et profileur'
description: >-
  Accélérer une exécution d'entraînement : capturer l'étape dans des graphes
  CUDA, choisir un type AMP et employer le profileur intégré pour trouver où le
  temps est réellement dépensé.
lead: >-
  Trois leviers modifient la vitesse d'une étape d'entraînement : la précision
  mixte, la capture de la propagation avant et arrière du réseau dans un graphe
  CUDA, et les mesures du profileur sur ce qui ralentit réellement l'étape.
keywords:
  - entraînement graphes cuda
  - vitesse entraînement
  - entraînement précision mixte
  - entraînement bfloat16
  - profileur pytorch
  - limite chargeur données
  - coût lancement kernels
  - utilisation gpu
last_verified: 1.5.0
snippets:
  profile:
    - label: Profiler puis continuer l'entraînement
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
    - label: Examiner le résultat
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

Les trois leviers ci-dessous corrigent des problèmes différents. Le mauvais
choix ne change rien. Le profileur indique lequel vous rencontrez.

<code-tabs name="profile" />

`profile=True` mesure une fenêtre d'étapes d'entraînement réelles, par défaut
cinq ignorées puis vingt mesurées, affiche un rapport, écrit ses artefacts, puis
poursuit l'entraînement après avoir retiré les hooks. Il ne coûte rien lorsqu'il
est désactivé et est ignoré en entraînement distribué.

Le rapport se termine par l'un de quatre verdicts :

| Verdict | Signification | Leviers |
|---|---|---|
| `dataloader` | le GPU attend les données d'entrée | davantage de `workers`, `cache="ram"` ou `"disk"`, augmentations plus légères, lot plus grand |
| `host / launch` | le GPU est alimenté trop lentement, beaucoup de petits kernels | lot plus grand, graphes CUDA, moins de synchronisations avec l'hôte par étape |
| `compute` | le GPU est saturé | AMP ou bfloat16, ou accepter la limite |
| `memory-pressure` | saturation de l'allocateur, VRAM à la limite | réduire le lot, les chiffres d'utilisation sont ici peu fiables |

Le taux d'utilisation est le temps d'activité des kernels divisé par le temps
d'étape non synchronisé. La fenêtre est volontairement divisée. La première
moitié s'exécute sans synchronisation supplémentaire afin que le verdict
reflète le chevauchement réel. Seule la seconde encadre chaque phase d'une
synchronisation pour attribuer le temps GPU. Synchroniser chaque phase donne du
répit aux workers du chargeur de données et masque leur manque d'alimentation.
Les nombres de composition ne servent donc jamais à choisir le verdict.

Quatre fichiers sont écrits dans le répertoire d'exécution :
`timeline.html`, autonome dans un navigateur, `profile_trace.json` pour Perfetto
ou Nsight, `profile_summary.json` et `profile.json`, le fichier autonome à
copier et à transmettre aux sous-commandes `libreyolo profile`.

Deux points sur `profile run` sont à connaître. Il définit
`no_aug_epochs=0`, car le profileur mesure l'époque 0. Une courte exécution avec
la valeur `no_aug_epochs` par défaut profilerait le chargeur allégé sans
augmentation plutôt que celui réellement utilisé par l'entraînement.
`--repeat N` rapporte la moyenne et l'écart-type, utiles car une étape limitée
par les lancements varie assez pour qu'une seule mesure soit trompeuse. Cette
option écrit les répertoires `prof_1`, `prof_2`, etc., ainsi qu'un fichier
global `profile_repeat.json`.

## Précision mixte

`amp=True` est la valeur par défaut de la plupart des familles et exécute la
propagation avant sous autocast CUDA. `amp_dtype` sélectionne `float16` ou
`bfloat16`.

<code-tabs name="amp" />

Float16 exige une mise à l'échelle dynamique de la perte et reçoit un gradient
scaler actif. La plage d'exposants plus large de bfloat16 n'en a pas besoin et
désactive son scaler. Quatre familles sont publiées avec `amp=False` : D-FINE,
DEIM, YOLO-NAS et FOMO. Le réglage DEIM est hérité par RT-DETRv4. D-FINE en
précise la raison : son décodeur limite les activations à 65 504, la plus grande
valeur float16 finie.

La sémantique de l'argument, notamment le comportement d'une demande bfloat16
sur du matériel incompatible, figure dans les
[hyperparamètres](/docs/train/hyperparameters).

## Graphes CUDA

`cuda_graph=True` capture la propagation avant et arrière du réseau dans un
graphe CUDA, supprimant le coût de lancement des kernels à chaque étape.

<code-tabs name="graph" />

Cette option peut toujours être transmise sans risque. Une famille, une tâche ou
une configuration impossible à capturer journalise une ligne et poursuit en
mode eager sans autre changement.

Seul le réseau est capturé. La perte reste volontairement en mode eager, car
les pertes de détection sélectionnent avec des masques booléens, effectuent une
association hongroise et bifurquent selon les résultats d'affectation, ce
qu'aucun graphe ne peut enregistrer. L'étape de l'optimiseur, l'écrêtage des
gradients, la mise à jour EMA et le planning du taux d'apprentissage restent
aussi en mode eager.

Le gain est limité par la part du réseau dans une étape, très variable. Mesures
sur une RTX 5070 Ti à 640 pixels, lot 8 : le réseau représente 84 % d'une étape
YOLOv9-t, 44 % d'une étape YOLOv7-b, 31 % d'une étape YOLOX-t et 26 % d'une
étape RTMDet-t. Les deux dernières passent l'essentiel de leur temps dans leurs
mécanismes d'affectation d'étiquettes. La capture du réseau les aide donc moins.

### Gain attendu

Conditions des chiffres ci-dessous : RTX 5070 Ti, Windows, AMP, un processus
par branche depuis un état enregistré commun, relecture d'un véritable lot pour
exclure le chargeur de données, meilleure des 24 étapes après préchauffage.
Détection à 640 pixels, classification à 224 pixels. La taille de lot est
indiquée par ligne.

| Famille | Taille | Lot | Eager | Capturé | Accélération |
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

Ces mesures isolent l'étape GPU. Un fine-tuning complet paie aussi le chargeur
de données et la validation. Sur la même machine, YOLOv9-t entraîné sur
406 images de détection pendant 20 époques, avec un lot de 8, 640 pixels et
4 workers, prend 428,4 s en eager contre 367,7 s avec graphe, soit un gain de
1,16x, avec une mAP50-95 de 0,6394 dans les deux cas.

Trois facteurs déplacent ces nombres. Les petits lots sont limités par les
lancements et les grands par le calcul. RT-DETR-r18 gagne ainsi 1,19x avec un
lot de 2 et 1,04x avec un lot de 8. Le coût de lancement est le plus élevé sous
Windows, et les gains sous Linux atteignent environ un tiers à la moitié du
tableau. Enfin, une exécution limitée par le chargeur de données ne voit aucun
changement de temps réel, d'où l'importance de profiler en premier.

La capture s'active de la même façon avec `amp=False`, mais les kernels fp32
s'exécutent plus longtemps. Une étape dépend moins des lancements et la plupart
des familles gagnent moins. Sur le même matériel, MobileNetV4-s avec un lot de
16 passe de 2,74x sous AMP à 3,61x en fp32, YOLOv9-t avec un lot de 8 de 1,99x
à 1,69x et RT-DETR-r18 avec un lot de 4 de 1,12x à 0,99x.

### Champ de la capture

| Tâche | Familles |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Tout le reste revient au mode eager avec une ligne de journal : autres tâches
de ces familles, familles non répertoriées, exécutions distribuées et
distillation. Un échec de capture à l'exécution fait aussi passer tout le reste
de l'exécution en mode eager au lieu de l'interrompre.

Pour les détecteurs encodeur-décodeur D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 et
v4, et EC, seuls le backbone et l'encodeur sont capturés. Leur décodeur lit la
vérité terrain pour construire des requêtes de débruitage contrastif, dont le
nombre suit le plus grand nombre de vérités terrain du lot. Son nombre de tokens
change donc d'un lot à l'autre.

### Formes

Un graphe n'est valide que pour la forme d'entrée capturée. Le programme
d'entraînement compte les formes des lots et effectue la capture après trois
répétitions. Les lots d'une autre forme s'exécutent en eager, notamment les lots
multi-échelles et le dernier lot partiel d'une époque.

C'est le piège des familles DETR, qui redimensionnent chaque lot par défaut.
Avec `multi_scale=True`, une courte exécution peut ne jamais revoir une forme
assez souvent pour la capturer. Transmettez `multi_scale=False` lorsque
l'accélération est l'objectif.

YOLOX modifie le calcul de la région capturée pendant l'exécution en activant sa
branche de régression L1 lorsque la mosaïque s'arrête à `no_aug_epochs`. Le
programme d'entraînement invalide alors la capture et la refait lorsque la
nouvelle forme se stabilise.

### Calculs numériques et mémoire

La plupart des familles reproduisent leur trajectoire de perte eager bit à bit
sous AMP. FOMO et LingBot-Vision diffèrent sur le dernier bit float32 du fait
d'un ordre de sommation différent. Les détecteurs à attention déformable,
D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR et EC, ne reproduisent pas non plus leurs
propres exécutions eager, car la rétropropagation accumule avec des opérations
atomiques et les convolutions TF32 choisissent un ordre de réduction à chaque
lancement. L'exécution capturée reste dans cette dispersion. RTMDet diffère
d'environ 3e-4 en relatif sur deux des 139 gradients, car ses convolutions de
tête sont partagées entre les niveaux de pyramide et les deux parcours arrière
additionnent trois contributions dans un ordre différent. SegFormer possède une
profondeur stochastique dans la région capturée. Un graphe rejoué tire donc son
propre flux aléatoire et est statistiquement équivalent au mode eager plutôt
qu'identique. Le gestionnaire le journalise une fois à la capture.

Avec `amp=False`, aucune exécution bit à bit identique n'est disponible sur ce
matériel, avec ou sans capture. Deux exécutions eager YOLOv9-t identiques et
déterministes divergent de 36 % en relatif sur 20 étapes, et YOLOX-t de 2,6 %,
car cuDNN choisit un algorithme non déterministe pour le gradient des poids de
certaines convolutions fp32.

Un graphe capturé fixe en mémoire les tampons statiques d'entrée, de sortie et
d'espace de travail. Le pic de VRAM augmente donc d'environ un ensemble
d'activations supplémentaire. Pour les familles ci-dessus, l'allocation
maximale a varié de -5 à +19 %. Le coût relatif est le plus élevé pour les
petits modèles de classification, dont les activations sont déjà petites :
ResNet-18 à 224 pixels, lot 16, passe de 0,48 Go en eager à 0,57 Go avec graphe.
Si cette hausse dépasse la limite, réduisez le lot ou désactivez l'option.

## Voir aussi

- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `nbs`, `cache`
  et `workers`.
- [Entraînement multi-GPU](/docs/train/multi-gpu), où les graphes CUDA et le
  profileur sont indisponibles.
- [Graphes CUDA](/docs/reference/cuda-graphs) pour la matrice combinée de prise
  en charge de l'inférence et de l'entraînement, les séparations des régions et
  le contrat numérique.
