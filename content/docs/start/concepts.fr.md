---
title: Concepts fondamentaux
seo_title: Concepts fondamentaux de LibreYOLO
description: >-
  Comprendre l'articulation entre les tâches, les familles de modèles, les
  tailles et les noms de fichiers de checkpoints dans LibreYOLO, ainsi que les
  garanties de chaque niveau de prise en charge.
lead: >-
  Quatre notions décrivent chaque modèle dans LibreYOLO : la tâche qu'il
  accomplit, la famille à laquelle il appartient, sa taille au sein de cette
  famille et le niveau de prise en charge de la famille. Le nom du fichier de
  checkpoint encode les trois premières.
keywords:
  - concepts libreyolo
  - tâches libreyolo
  - familles de modèles libreyolo
  - nommage checkpoints libreyolo
  - niveaux prise en charge libreyolo
last_verified: 1.5.0
meta:
  - label: Schéma de nommage
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Tâches canoniques
    value: 17
  - label: Niveaux de prise en charge
    value: >-
      Vedette, Principal, Pris en charge, Inférence uniquement, Musée, Niveau
      parallèle
snippets:
  inspect:
    - label: Lister les familles
      language: bash
      code: |
        # Tâches, tailles et résolutions d'entrée de chaque famille enregistrée.
        libreyolo models
    - label: Un modèle
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Choisir une tâche
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Les alias sont normalisés à l'entrée de l'API : "keypoints" devient

        # "pose", "det" devient "detect", "semantic-segmentation" devient
        "semantic".

        model = LibreYOLO("LibreYOLO9t.pt", task="det")

        print(model.task)
source_hash: 23d045463a6a8411
---

## Tâches

Une tâche correspond à ce que renvoie un modèle. LibreYOLO compte dix-sept noms
de tâches canoniques, chacun désignant le champ de l'objet `Results` qui porte
sa sortie.

| Tâche | Valeur renvoyée |
|---|---|
| `detect` | Boîtes alignées sur les axes, avec une classe et un score de confiance |
| `segment` | Masques par instance, à raison d'un masque par objet détecté |
| `semantic` | Une étiquette de classe par pixel, sans séparation des instances |
| `panoptic` | Une étiquette sans chevauchement par pixel, réunissant les objets dénombrables et les régions amorphes |
| `pose` | Points clés par instance, dont les lignes sont alignées sur les boîtes |
| `classify` | Une probabilité sur un ensemble d'étiquettes pour l'image entière |
| `obb` | Boîtes orientées, avec un angle de rotation |
| `point` | Une coordonnée dans l'image par détection, plutôt qu'une boîte |
| `depth` | Une carte dense de profondeur inverse relative |
| `normal` | Un champ dense de normales de surface sous forme de vecteurs unitaires |
| `edge` | Une carte dense de probabilité des contours |
| `restore` | Une image RVB restaurée, pour le défloutage, le débruitage ou la super-résolution |
| `matte` | Une carte progressive du premier plan de 0 à 1, pour supprimer l'arrière-plan |
| `ocr` | Quadrilatères de texte accompagnés de leur transcription, dans l'ordre de lecture |
| `embed` | Un vecteur normalisé en L2 dont le produit scalaire mesure la concordance |
| `gaze` | Une direction du regard par visage détecté |
| `mesh` | Un corps 3D articulé par personne détectée |

Ces noms figurent dans les métadonnées et les noms de fichiers des checkpoints.
Les alias courants sont acceptés partout où une tâche est fournie et sont
normalisés avant toute autre opération : `detection` et `det` deviennent
`detect`, `keypoints` devient `pose`, `cls` devient `classify`, tandis que
`deblur`, `denoise` et `super-resolution` deviennent tous `restore`, et
`face-recognition` et `reid` deviennent `embed`. Un nom inconnu déclenche une
erreur au lieu d'être remplacé silencieusement par la valeur par défaut.

`segment`, `semantic` et `panoptic` sont trois tâches distinctes, pas trois
termes pour désigner la même chose. Les masques d'instances, les étiquettes par
pixel et la carte fusionnée des objets et des régions ont des vérités terrain,
des métriques et des champs de résultats différents.

## Familles de modèles

Une famille est une lignée d'architectures disposant de son propre code de
chargement, de prétraitement et de post-traitement. Chaque famille déclare un
identifiant `FAMILY` tel que `yolo9`, `rfdetr` ou `dfine`, les tâches qu'elle
prend en charge et la résolution d'entrée de chaque taille publiée.

`LibreYOLO()` est une fabrique plutôt qu'une classe. À partir d'un chemin, elle
charge le fichier, identifie la famille grâce aux métadonnées du checkpoint ou,
à défaut, grâce aux clés des tenseurs, puis renvoie une instance du modèle de
cette famille. C'est pourquoi changer de détecteur ne demande qu'une ligne :
l'objet obtenu expose la même interface `predict`, `train`, `val` et `export`,
et renvoie le même type `Results`.

<code-tabs name="inspect" />

Une famille qui couvre plusieurs tâches publie généralement un checkpoint
distinct par tâche, souvent avec un ensemble de tailles différent pour chacune.
Quelques familles partagent toutefois un même artefact entre deux tâches à
l'exécution. Dans les deux cas, la liste des tâches prises en charge est fixe.
Demander une tâche qui n'y figure pas déclenche une erreur indiquant la liste
prise en charge, au lieu de charger une approximation.

La liste complète, avec les benchmarks par famille et les poids publiés, se
trouve sur la page [tous les modèles](/docs/models).

## Tailles

Une taille est une variante au sein d'une famille. Elle s'écrit sous la forme
d'un code en minuscules directement accolé au préfixe de la famille. Les lettres
courantes sont `n` pour nano, `t` pour tiny, `s` pour small, `m` pour medium,
`l` pour large et `x` pour xlarge. Ces codes restent toutefois propres à chaque
famille, et plusieurs familles utilisent des conventions entièrement
différentes : des codes portant le nom du backbone, tels que `r50` ou `r101`,
où la taille correspond à la profondeur d'un ResNet, des codes de mise à
l'échelle composée allant de `b0` à `b3`, ou un nom qui identifie l'unique
checkpoint publié. YOLOv9 emploie `c` pour compact là où d'autres familles
emploient `l`.

La taille fixe également la résolution d'entrée. Pour les familles à plusieurs
tâches, cette résolution peut varier selon la tâche. Ces deux informations sont
lues depuis la famille et ne sont jamais supposées. `libreyolo models` les
affiche.

## Noms de fichiers des checkpoints

Tous les fichiers de poids publiés suivent le même schéma :

```text
Libre<FAMILY><size>[-<task>].pt
```

Le préfixe de famille est une chaîne fixe propre à chaque famille. La taille est
en minuscules et accolée sans séparateur, tandis que le suffixe de tâche est
précédé d'un trait d'union. La détection n'a pas de suffixe, conformément à la
convention historique des checkpoints YOLO. Ainsi, `LibreYOLO9t.pt` est un
détecteur et `LibreRFDETRn-seg.pt` est un modèle de segmentation de la même
famille.

| Tâche | Suffixe |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Une famille dépourvue de tâche sans suffixe peut rendre ce dernier obligatoire.
Un nom sans suffixe n'est alors pas accepté comme checkpoint valide pour cette
famille. Lorsqu'une famille publie des poids entraînés sur un dataset différent
de celui par défaut, elle ajoute le nom du dataset comme suffixe supplémentaire.
Cette variante reste intégrée au nom du dépôt depuis lequel le fichier est
téléchargé.

Trois niveaux échappent à ce schéma. Les familles de segmentation guidée, les
familles vision-langage et les détecteurs à vocabulaire ouvert ne sont pas
enregistrés dans la fabrique de checkpoints et ne produisent aucun fichier
`Libre<FAMILY><size>.pt`. Leur préfixe désigne à la place un snapshot Hugging
Face téléchargé ou un checkpoint de segmentation guidée. La casse de la marque
amont y est volontairement conservée.

## Déterminer la tâche

Lorsque plusieurs signaux peuvent désigner une tâche, ils sont consultés dans
un ordre fixe et le premier présent l'emporte : l'argument `task` transmis, puis
la tâche inscrite dans les métadonnées du checkpoint, le suffixe de tâche du nom
de fichier et enfin la tâche par défaut de la famille. Le résultat est comparé
aux tâches prises en charge par la famille avant la construction du modèle. Une
incompatibilité échoue donc au chargement au lieu de produire plus tard un
résultat incorrect.

## Niveaux de prise en charge

Chaque famille appartient à un seul niveau. Un niveau décrit l'attention portée
par l'équipe d'ingénierie, pas la précision : il indique où une nouvelle
fonctionnalité arrive en premier et ce qui doit rester opérationnel.

| Niveau | Signification |
|---|---|
| Vedette | Les fonctionnalités sont conçues et entièrement validées sur GPU ici en premier |
| Principal | Détecteurs principaux entraînables. Les fonctionnalités suivent celles des modèles vedettes dans la même vague de publication |
| Pris en charge | Familles entraînables complémentaires. Maintenues opérationnelles dans la CI, avec des fonctionnalités ajoutées au gré des occasions |
| Inférence uniquement | Prédiction, validation et exportation. Les fonctionnalités d'entraînement ne s'appliquent pas |
| Musée | Une pièce figée. Corrections de bogues uniquement |
| Niveau parallèle | Une surface produit distincte, avec sa propre fabrique et son propre contrat |

La page de chaque modèle indique le niveau de sa famille dans son en-tête. Les
deux familles vedettes sont [YOLOv9](/docs/models/yolov9) pour les détecteurs CNN
et [RF-DETR](/docs/models/rf-detr) pour les détecteurs Transformer. Commencez
par elles, sauf besoin contraire.

« Inférence uniquement » indique ce qui manque, à savoir une boucle
d'entraînement dans LibreYOLO. La prédiction, la validation et, lorsque la
famille le permet, l'exportation fonctionnent toutes. Appeler `train()` sur une
telle famille déclenche une `NotImplementedError` qui en précise la raison.
