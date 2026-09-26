---
title: Niveaux de stabilité
seo_title: Signification de chaque niveau de prise en charge de LibreYOLO
description: "Le vocabulaire des niveaux LibreYOLO\_: les trois niveaux de prise en charge de l'export, les quatre niveaux d'API, les six groupes de couverture et ce qu'aucun ne garantit."
lead: "LibreYOLO emploie le mot niveau pour trois notions distinctes\_: les preuves étayant un chemin d'export, le contrat d'appel auquel répond une famille de modèles et le groupe de couverture auquel une famille appartient. Cette page définit chacun d'eux et précise ce qu'il n'implique pas."
keywords:
  - niveau support libreyolo
  - validated available blocked
  - niveaux support export
  - groupes couverture libreyolo
  - g0 g1 g2 g3 g4
  - niveaux modèles
last_verified: 1.5.0
verification: "Niveaux d'export lus dans docs/adr/0011-export-support-tiers.md et libreyolo/export/support.py\_; groupes de couverture et nombres par famille lus dans MODEL_GROUPS dans libreyolo/models/registry.py\_; contrôle de l'entraînement à partir de zéro lu dans libreyolo/models/base/model.py et libreyolo/cli/commands/train.py\_; inventaire du CLI lu dans libreyolo/models/inventory.py\_; niveaux d'API lus dans les docstrings des packages libreyolo/models/sam/, openvocab/ et vlm/ ainsi que dans les contrats de base.py, le tout en v1.5.0. Les noms de groupes destinés au lecteur (Flagship, Core, Supported, Inference only, Museum, Sibling tier) sont le vocabulaire propre au site pour les mêmes groupes, lu dans src/data/docs/registry.json."
snippets:
  usage:
    - label: Lire les deux classifications d'une famille
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Niveaux de prise en charge de l'export

Ce niveau détermine si un appel aboutit. Il s'applique au triplet
`(family, task, format)` et chaque combinaison en possède exactement un.

| Niveau | Signification | Comportement de `export()` |
|---|---|---|
| `validated` | La parité numérique est couverte par la CI ou une exécution nocturne documentée | S'exécute |
| `available` | La conversion est implémentée, mais aucune preuve de parité numérique dans le runtime n'a été consignée | S'exécute |
| `blocked` | Aucun chemin pris en charge | Lève `NotImplementedError` lors des vérifications préalables, avec la raison |

Les niveaux validated et available poursuivent tous deux sans demande de
confirmation ni avertissement général. La différence porte sur les preuves,
pas sur l'autorisation\u00a0: une entrée validated s'appuie sur un test de parité et
une version `since`, contrairement à une entrée available. Une conversion
CoreML sans exécution de prédiction sous macOS est par exemple available, et
non validated.

Une combinaison blocked échoue avant la vérification des dépendances, le
chargement de la calibration, le traçage ou la création de l'artefact. Aucun
élément partiel n'est donc écrit.

Chaque cellule validated contient une contrainte qui décrit la configuration
ayant produit la mesure de parité, généralement un canevas d'entrée fixe, un
batch de 1, le FP32 et une version nommée du runtime. Interprétez-la comme une
affirmation sur cette configuration et non sur le format en général. Les
règles qui remplissent les cellules sans entrée explicite figurent sur la page
de la [matrice d'export](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Niveaux d'API

Ce niveau détermine la forme d'un appel. Une famille appartient exactement à
un niveau, choisi selon son contrat d'appel et non son architecture.

| Niveau | Fabrique | Contrat |
|---|---|---|
| Fabrique de détecteurs | `LibreYOLO` | Une passe forward sans prompt renvoie tous les objets trouvés avec des scores calibrés. Les membres s'enregistrent en reconnaissant un checkpoint |
| Segmentation guidable | `LibreSAM` | Une passe forward n'a aucun sens sans prompt spatial ou conceptuel propre à l'image, transmis lors de l'appel. Interactive et à état\u00a0: encoder une fois, guider plusieurs fois |
| Détection à vocabulaire ouvert | `LibreOpenVocab` | Détecteurs discriminatifs conditionnés par du texte. La liste de classes est un prompt défini par `set_classes` |
| Vision-langage | `LibreVLM` | Modèle génératif utilisé comme détecteur. La liste de classes est un prompt et la confiance une valeur factice |

Les trois niveaux frères ne s'enregistrent délibérément pas dans la fabrique
de détecteurs, c'est pourquoi `LibreYOLO("some-alias")` ne les atteint pas. Ils
se chargent par alias de taille et téléchargement automatique plutôt que par
inspection de checkpoint.

Tous les quatre renvoient les mêmes `Results`, le code downstream reste donc
inchangé. Les méthodes disponibles diffèrent\u00a0: les niveaux frères lèvent
`NotImplementedError` pour `train()`, `val()` et `export()`, tandis que les
niveaux SAM et vocabulaire ouvert le lèvent aussi pour `track()`. Chaque page
de niveau énumère ses propres exclusions.

## Groupes de couverture

Cette classification détermine les familles incluses dans une exécution de
tests inter-familles. C'est aussi celle qu'un lecteur rencontrera le plus
souvent sur une page de modèle. Chaque famille enregistrée appartient à un
groupe exactement, et un test échoue si une famille enregistrée n'est affectée
à aucun groupe. `GROUPS` dans `libreyolo/models/registry.py` est la source de
la colonne Signification ci-dessous. `MODEL_GROUPS` dans le même fichier
affecte chaque famille, et la colonne Familles compte directement ces
affectations. La colonne Nom contient le nom plus court employé par le site
pour le même groupe dans l'en-tête d'une page de modèle.

| Groupe | Nom | Familles | Signification |
|---|---|---|---|
| `g0` | Flagship | 2 | Piliers principaux requis dans la couverture des fonctionnalités partagées |
| `g1` | Core | 10 | Ensemble de couverture des détecteurs entraînables |
| `g2` | Supported | 14 | Ensemble de couverture supplémentaire des familles entraînables |
| `g3` | Inference only | 35 | Familles sans implémentation d'entraînement |
| `g4` | Museum | 5 | Familles historiques avec couverture de l'inférence |
| `s` | Sibling tier | 21 | API sœurs (SAM, vocabulaire ouvert, VLM, zero-shot) couvertes séparément |

Cela représente 87 familles réparties entre six groupes. `g3` contient à lui
seul plus de familles que tous les autres groupes réunis, car la majeure partie
du registre couvre des lignées réservées à l'inférence et des modèles
historiques plutôt que des détecteurs activement entraînés.

Pour choisir un modèle, le groupe indique où attendre l'attention de
l'ingénierie et non l'exactitude d'une famille. `g0` et `g1` accueillent la
conception et la première intégration d'une nouvelle fonctionnalité. `g2` reste
vert dans la CI, mais une fonctionnalité y arrive selon les occasions plutôt
que dans la même vague de version. `g3` énonce une absence et non une limite\u00a0:
la prédiction, la validation et, lorsque la famille le permet, l'export
fonctionnent toujours. `train()` sur une famille `g3` ou `g4` lève
`NotImplementedError` en donnant la raison au lieu d'exécuter un comportement
partiel silencieux. Les familles `s` n'entrent pas du tout dans ce compromis,
car elles utilisent leur propre fabrique plutôt que `LibreYOLO()`. Consultez
les [concepts fondamentaux](/docs/concepts) pour comprendre la place du groupe
aux côtés de la tâche, de la famille et de la taille dans le nom d'un
checkpoint.

Un groupe n'accorde ni ne restreint à lui seul une fonctionnalité visible par
l'utilisateur. La prise en charge vient de l'API implémentée par la famille et
des contrôles de capacité propres au format, jamais de la seule appartenance à
un groupe. Les groupes classent les familles et non les tâches. Une exécution
de couverture limitée à une tâche nomme donc explicitement celle-ci, comme
dans «\u00a0g1 detect\u00a0».

Deux éléments lisent le groupe à l'exécution et pas seulement dans les tests.
`collect_model_inventory()` dans `libreyolo/models/inventory.py` joint le
groupe à chaque entrée affichée par l'inventaire du CLI, et `pretrained=False`
ne déclenche le chemin spécial de réinitialisation à partir de zéro que pour
les familles de `g0` et `g1`. Hors de ces deux groupes, le contrôle dans
`libreyolo/models/base/model.py` est entièrement ignoré. `pretrained=False`
atteint alors la méthode `train()` propre à la famille comme un argument nommé
ordinaire.

## Entraînement

Une famille de `g3` ou `g4` ne possède aucune implémentation d'entraînement et
l'appel à `train()` lève une erreur. Il s'agit d'une propriété du code de la
famille, et non de son groupe\u00a0: le groupe enregistre ce fait sans le provoquer.

Pour une famille entraînable, le fait qu'un réglage d'augmentation atteigne le
pipeline est une question distincte avec son propre vocabulaire à trois
valeurs, `used`, `gated_by_mosaic` et `ignored`. Consultez la
[matrice d'augmentation](/docs/reference/augmentation-matrix).

## Ce qu'un niveau n'indique pas

Un niveau n'est pas une affirmation d'exactitude. Un export validated indique
que l'artefact reproduit le modèle natif dans un seuil donné, mais pas les
performances du modèle natif sur un dataset. Les mesures de benchmark figurent
sur les pages des modèles.

Un niveau n'est pas non plus une indication de licence. Les licences des poids
varient au sein d'une famille et le dépôt qui héberge un checkpoint précis fait
autorité. La présence d'une famille dans la fabrique de détecteurs ne dit rien
sur l'autorisation d'utiliser commercialement ses poids publiés.
