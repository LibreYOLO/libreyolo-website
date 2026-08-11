---
title: Datasets
seo_title: Datasets d'entraînement dans LibreYOLO
description: >-
  Le fichier YAML de dataset lu par LibreYOLO, l'arborescence attendue, le
  fonctionnement du téléchargement automatique et la commande doctor qui
  contrôle un dataset avant l'entraînement.
lead: >-
  Un dataset LibreYOLO est un fichier YAML qui nomme une racine, ses partitions
  et ses classes. Tout le reste, notamment l'emplacement des fichiers
  d'étiquettes, en est dérivé par convention.
keywords:
  - format dataset yolo
  - data.yaml
  - entraînement dataset personnalisé
  - format étiquettes yolo
  - dataset json coco
  - téléchargement automatique dataset
  - libreyolo doctor
  - vérifier déséquilibre classes
  - fuite partitions train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Un nom inclus, un chemin relatif ou un chemin absolu fonctionnent
        tous.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Contrôler un dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Faire aussi échouer la CI sur les avertissements
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Ignorer le décodage des images
      language: bash
      code: >
        # Lit uniquement les étiquettes et le YAML. Les contrôles de corruption,

        # de doublons et de fuite entre partitions nécessitent les pixels et
        sont ignorés.

        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Indiquer un dataset à `train`

`data=` reçoit le chemin d'un fichier YAML ou le nom d'une configuration
incluse dans le paquet.

<code-tabs name="train" />

Le nom est résolu dans un ordre fixe : un chemin absolu existant, puis le nom
tel quel relativement au répertoire de travail, le même nom avec `.yaml` ajouté,
et enfin le répertoire des configurations incluses. Si rien ne correspond,
l'erreur nomme chaque répertoire recherché et répertorie les configurations
incluses.

## Configurations incluses

Treize configurations de datasets sont fournies dans
`libreyolo/config/datasets/`.

| Configuration | Tâche | Remarques |
|---|---|---|
| `coco8.yaml` | detect | 8 images, téléchargées depuis une URL simple |
| `coco128.yaml` | detect | 128 images |
| `coco1000.yaml` | detect | 800 pour l'entraînement, 200 pour la validation |
| `coco5000.yaml` | detect | 4 000 pour l'entraînement, 1 000 pour la validation |
| `coco.yaml` | detect | COCO 2017 complet |
| `coco-val-only.yaml` | detect | val2017 uniquement |
| `coco8-pose.yaml` | pose | 8 images, points clés COCO-17 |
| `coco-pose.yaml` | pose | Points clés COCO 2017 |
| `ade20k.yaml` | semantic | 150 classes |
| `cityscapes.yaml` | semantic | 19 classes, téléchargement manuel |
| `cocostuff.yaml` | semantic | 182 classes, téléchargement manuel |
| `gopro.yaml` | restore | Paires de défloutage |
| `sr8.yaml` | restore | Paires de super-résolution |

Seuls `coco8.yaml` et `coco128.yaml` contiennent une URL de téléchargement
simple. Les autres contiennent soit un bloc de téléchargement Python qui
nécessite l'autorisation décrite plus bas, soit supposent que les données sont
déjà présentes sur le disque.

## Emplacement d'un dataset sur le disque

La clé YAML `path` nomme la racine du dataset. Un `path` absolu est utilisé tel
quel. Un chemin relatif est d'abord recherché sous le répertoire des datasets,
puis à côté du fichier YAML. Un dataset sur le point d'être téléchargé est
placé sous le répertoire des datasets.

Ce répertoire est `~/datasets`, sauf remplacement par la variable
d'environnement `LIBREYOLO_DATASETS_DIR`. Aucun fichier de réglages ne le
contrôle.

## Clés YAML

```yaml
path: my-dataset        # racine du dataset
train: images/train     # obligatoire pour l'entraînement
val: images/val         # obligatoire pour la validation
test: images/test       # facultatif
nc: 3                   # facultatif ; doit correspondre à names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # facultatif
```

`train`, `val` et `test` acceptent chacun un répertoire d'images, un fichier
`.txt` qui contient un chemin d'image par ligne ou une liste qui mélange les
deux. Les lignes d'une liste `.txt` peuvent être relatives. Elles se résolvent
alors depuis le propre répertoire du fichier de liste. Les lignes qui commencent
par `#` sont ignorées.

`names` peut être une liste ou une association à clés entières. `nc` est
facultatif. Lorsque les deux sont présents et ne correspondent pas, doctor le
signale comme une erreur.

## Arborescence et fichiers d'étiquettes

La détection, la segmentation, la pose et les boîtes orientées partagent une
structure. Le chemin de l'étiquette dérive du chemin de l'image en remplaçant un
composant de répertoire `images` par `labels` et l'extension par `.txt` :

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Seul un composant de chemin `images` entier est remplacé. Un répertoire nommé
`images_old` reste inchangé.

Une ligne de détection comporte cinq champs, tous normalisés dans `[0, 1]` selon
la largeur et la hauteur de l'image d'origine :

```text
<class_id> <cx> <cy> <w> <h>
```

Un fichier d'étiquettes absent ou vide signifie que l'image ne contient aucun
objet. Elle est entraînée comme arrière-plan plutôt que de déclencher une
erreur. Une ligne de plus de cinq champs est lue comme un polygone, dont la boîte
devient l'étendue. Une exportation de segmentation utilisée pour entraîner la
détection se charge donc sans erreur. Doctor rapporte le nombre de lignes qui
ont suivi ce parcours.

## Autres tâches

La segmentation conserve la même structure avec des lignes polygonales
`<class_id> <x1> <y1> ... <xN> <yN>` d'au moins trois points. Une ligne de
détection à cinq champs est acceptée et désigne une instance rectangulaire.

La pose ajoute `kpt_shape: [K, D]` et une permutation `flip_idx` facultative au
fichier YAML. Chaque ligne contient exactement `5 + K * D` champs : la boîte,
puis `K` points clés `x y` ou `x y v`, avec une visibilité `0`, `1` ou `2`.

Les boîtes orientées emploient exactement neuf champs : la classe suivie de
quatre coins en coordonnées normalisées. Aucun angle n'est enregistré.

La segmentation sémantique associe chaque image à un masque monocanal de même
résolution, résolu en substituant `masks_dir` (`masks` par défaut) à `images`.
La valeur de pixel `255` signifie ignorer. `label_mapping` remappe les
identifiants source vers ceux de l'entraînement au chargement.

La classification emploie une arborescence ImageFolder plutôt que des fichiers
d'étiquettes. `train/` et `val/` contiennent chacun un répertoire par classe.
L'association classe-indice suit l'ordre des noms de dossiers triés.

La restauration associe une entrée dégradée à une cible propre de résolution
identique par `input_dir` et `target_dir`. La profondeur, les normales de
surface et les contours associent chacun une image à une carte dense par leur
propre clé de répertoire.

Le contrat complet par tâche, notamment les conventions d'échelle de profondeur
et l'encodage PNG des identifiants de segments panoptiques, se trouve dans
`docs/dataset_schema.md` dans le dépôt de la bibliothèque.

## JSON COCO natif

Un fichier d'annotations JSON COCO peut être utilisé directement. Ajoutez une
association `annotations`. Le chemin de partition devient alors la racine des
images :

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Lorsque `names` est présent, les noms de catégories du JSON doivent lui
correspondre et `names` définit les identifiants prédits par le modèle. Sans
`names`, les identifiants de catégories COCO sont triés et remappés de façon
dense dans `0..N-1`.

Ce parcours attend un répertoire d'images par partition. Une liste de chemins ou
une liste d'images `.txt` déclenche une erreur au lieu de charger
silencieusement un autre ensemble.

## Téléchargement automatique

Un dataset est considéré comme présent lorsque son chemin `train` ou `val` se
résout vers un répertoire non vide ou un fichier existant. Dans le cas
contraire, si le fichier YAML contient une clé `download`, sa valeur détermine
la suite.

Une URL `http` ou `https` est récupérée et, s'il s'agit d'un zip, extraite dans
la racine du dataset. Toute autre valeur est traitée comme un script Python
intégré et s'exécute uniquement avec `allow_download_scripts=True`. Sans cette
option, le script est ignoré avec un avertissement et l'entraînement continue
avec les éléments présents sur le disque.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

L'option constitue une barrière d'exécution du code, pas une barrière réseau.
Les téléchargements par URL ont lieu dans tous les cas. Seuls les blocs
`download: |` la nécessitent. La CLI affiche un avertissement lorsque l'option
est active, et doctor ne l'active jamais.

## Contrôler le dataset avant l'entraînement

`libreyolo doctor` lit un dataset de détection et signale les problèmes avant
qu'un GPU n'intervienne. Il renvoie le code 1 lorsqu'il trouve des erreurs et
peut donc servir de contrôle de CI.

<code-tabs name="doctor" />

Les contrôles se répartissent en six familles :

| Famille | Recherche |
|---|---|
| `config` | `names` absent, `nc` incompatible avec `names`, partitions absentes ou vides, noms de classes en double |
| `files` | images sans fichier d'étiquettes, étiquettes sans image, images absentes citées dans une partition, collisions de noms de base |
| `labels` | lignes mal formées, identifiants de classes hors de `[0, nc)`, coordonnées hors de `[0, 1]`, boîtes d'aire nulle, minuscules ou énormes, boîtes en double, fichiers d'étiquettes identiques octet par octet |
| `balance` | classes sans ou avec peu d'instances, rapport de déséquilibre, classes présentes dans une seule partition, proportion d'images d'arrière-plan |
| `images` | fichiers illisibles, rotation EXIF, dispositions de canaux inhabituelles, images uniformes, doublons exacts et proches |
| `splits` | même image présente dans deux partitions, exactement ou presque |

`--only` et `--skip` acceptent un identifiant de contrôle ou un préfixe de
famille. `skip=images,labels.tiny_object` est donc valide. `--fast` supprime
tous les contrôles qui doivent décoder les pixels, c'est-à-dire les familles
`images` et `splits`.

Deux comportements sont à connaître. `--strict` fait échouer le code de sortie
pour les avertissements comme pour les erreurs. Doctor ne couvre en outre que
les datasets de détection. Un dataset de pose, de segmentation ou de boîtes
orientées est refusé avec un message qui nomme le type détecté, plutôt que
contrôlé selon un contrat incorrect.

## Voir aussi

- [Hyperparamètres](/docs/train/hyperparameters) pour les arguments acceptés par
  `train()` une fois les données en place.
- [Validation et métriques](/docs/train/validation) pour l'évaluation sur la
  partition `val` ou `test`.
