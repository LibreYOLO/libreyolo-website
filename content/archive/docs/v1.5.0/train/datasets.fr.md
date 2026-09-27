---
title: Datasets
seo_title: "Datasets d'entraînement dans LibreYOLO"
description: >-
  Le YAML de dataset lu par LibreYOLO, la disposition de dossiers attendue, le
  fonctionnement du téléchargement automatique et la commande doctor qui vérifie
  un dataset avant l'entraînement.
lead: >-
  Un dataset LibreYOLO est un fichier YAML qui nomme une racine, ses splits et
  ses noms de classes. Tout le reste, y compris l'emplacement des fichiers
  d'étiquettes, est dérivé de ce fichier par convention.
keywords:
  - format dataset yolo
  - data.yaml
  - entraînement dataset personnalisé
  - format étiquettes yolo
  - dataset json coco
  - téléchargement automatique dataset
  - libreyolo doctor
  - vérifier déséquilibre classes
  - fuite entre splits train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Un nom fourni, un chemin relatif ou un chemin absolu fonctionnent
        tous.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Vérifier un dataset
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
        # Lit seulement les étiquettes et le YAML. Les vérifications de
        corruption,

        # de doublons et de fuite entre splits exigent les pixels et sont
        ignorées.

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

## Pointer l'entraînement vers un dataset

`data=` accepte le chemin d'un fichier YAML ou le nom d'une configuration
fournie avec le package.

<code-tabs name="train" />

Le nom est résolu dans un ordre fixe : d'abord un chemin absolu existant, puis
le nom tel quel relativement au répertoire de travail, puis le même nom avec
`.yaml` ajouté, et enfin le répertoire des configurations fournies. Si rien ne
correspond, l'erreur nomme chaque répertoire recherché et liste les
configurations fournies.

## Configurations fournies

Treize configurations de dataset sont distribuées dans le package, sous
`libreyolo/config/datasets/`.

| Configuration | Tâche | Remarques |
|---|---|---|
| `coco8.yaml` | detect | 8 images, téléchargement depuis une URL simple |
| `coco128.yaml` | detect | 128 images |
| `coco1000.yaml` | detect | 800 pour train, 200 pour val |
| `coco5000.yaml` | detect | 4000 pour train, 1000 pour val |
| `coco.yaml` | detect | COCO 2017 complet |
| `coco-val-only.yaml` | detect | val2017 uniquement |
| `coco8-pose.yaml` | pose | 8 images, points clés COCO-17 |
| `coco-pose.yaml` | pose | points clés COCO 2017 |
| `ade20k.yaml` | semantic | 150 classes |
| `cityscapes.yaml` | semantic | 19 classes, téléchargement manuel |
| `cocostuff.yaml` | semantic | 182 classes, téléchargement manuel |
| `gopro.yaml` | restore | paires pour le défloutage |
| `sr8.yaml` | restore | paires pour la super-résolution |

Seuls `coco8.yaml` et `coco128.yaml` contiennent une URL de téléchargement
simple. Les autres contiennent soit un bloc de téléchargement Python, qui exige
l'activation décrite ci-dessous, soit supposent que les données sont déjà sur
le disque.

## Emplacement d'un dataset sur le disque

La clé YAML `path` nomme la racine du dataset. Un `path` absolu est utilisé tel
quel. Un chemin relatif est d'abord recherché sous le répertoire des datasets,
puis à côté du fichier YAML lui-même. Un dataset sur le point d'être téléchargé
est placé sous le répertoire des datasets.

Ce répertoire est `~/datasets`, sauf remplacement par la variable
d'environnement `LIBREYOLO_DATASETS_DIR`. Il n'existe aucun fichier de
paramètres pour le définir.

## Les clés YAML

```yaml
path: my-dataset        # racine du dataset
train: images/train     # requis pour l'entraînement
val: images/val         # requis pour la validation
test: images/test       # facultatif
nc: 3                   # facultatif ; doit correspondre à names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # facultatif
```

`train`, `val` et `test` acceptent chacun un répertoire d'images, un fichier
`.txt` qui liste un chemin d'image par ligne, ou une liste mélangeant les deux.
Les lignes d'une liste `.txt` peuvent être relatives, auquel cas elles sont
résolues par rapport au répertoire du fichier de liste, et les lignes commençant
par `#` sont ignorées.

`names` peut être une liste ou une correspondance à clés entières. `nc` est
facultatif ; lorsque les deux sont présents et ne concordent pas, doctor le
signale comme une erreur.

## Disposition des répertoires et fichiers d'étiquettes

La détection, la segmentation, la pose et les boîtes orientées partagent la
même disposition. Le chemin de l'étiquette est dérivé du chemin de l'image en
remplaçant un composant de répertoire `images` par `labels` et l'extension par
`.txt` :

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Seul un composant de chemin `images` complet est remplacé, si bien qu'un
répertoire nommé `images_old` reste inchangé.

Une ligne de détection contient cinq champs, tous normalisés dans `[0, 1]` par
rapport à la largeur et à la hauteur de l'image d'origine :

```text
<class_id> <cx> <cy> <w> <h>
```

Un fichier d'étiquette absent ou vide signifie que l'image ne contient aucun
objet. Elle est alors entraînée comme arrière-plan sans provoquer d'erreur. Une
ligne de plus de cinq champs est lue comme un polygone et sa boîte correspond à
l'étendue de ce polygone, si bien qu'un export de segmentation utilisé pour un
entraînement de détection se charge sans plainte. Doctor indique combien de
lignes ont suivi ce chemin.

## Autres tâches

La segmentation conserve la même disposition avec des lignes de polygones,
`<class_id> <x1> <y1> ... <xN> <yN>`, d'au moins trois points. Une ligne de
détection à cinq champs est acceptée et désigne une instance rectangulaire.

La pose ajoute `kpt_shape: [K, D]` et une permutation `flip_idx` facultative au
YAML. Chaque ligne contient exactement `5 + K * D` champs : la boîte, puis `K`
points clés sous la forme `x y` ou `x y v`, avec une visibilité de `0`, `1` ou
`2`.

Les boîtes orientées utilisent exactement neuf champs, soit la classe suivie de
quatre coins en coordonnées normalisées. Aucun angle n'est stocké dans le
fichier.

La segmentation sémantique associe chaque image à un masque monocanal de même
résolution, résolu en remplaçant `images` par `masks_dir` (`masks` par défaut).
La valeur de pixel `255` signifie ignorer. `label_mapping` remappe les
identifiants source vers les identifiants d'entraînement au chargement.

La classification utilise une arborescence ImageFolder au lieu de fichiers
d'étiquettes, avec un répertoire par classe dans chacun de `train/` et `val/`.
La correspondance entre classes et indices suit l'ordre trié des noms de
dossiers.

La restauration associe une entrée dégradée à une cible propre de résolution
identique par `input_dir` et `target_dir`. La profondeur, les normales de
surface et les contours associent chacun une image à une carte dense par leur
propre clé de répertoire.

Le contrat complet par tâche, notamment les conventions d'échelle de profondeur
et l'encodage PNG des identifiants de segments panoptiques, se trouve dans
`docs/dataset_schema.md` dans le dépôt de la bibliothèque.

## JSON COCO natif

Un fichier d'annotations JSON COCO peut être utilisé directement. Ajoutez une
correspondance `annotations`, et le chemin du split devient la racine des
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
correspondre, et `names` définit les identifiants d'étiquettes prédits par le
modèle. Sans `names`, les identifiants de catégories COCO sont triés et
remappés de façon dense vers `0..N-1`.

Ce chemin attend un répertoire d'images par split. Une liste de chemins ou une
liste d'images `.txt` provoque une erreur au lieu de charger silencieusement un
autre ensemble.

## Téléchargement automatique

Un dataset est considéré comme présent lorsque son chemin `train` ou `val` se
résout vers un répertoire non vide ou un fichier existant. Dans le cas
contraire, si le YAML possède une clé `download`, sa valeur détermine la suite.

Une URL `http` ou `https` est récupérée et, s'il s'agit d'un zip, extraite dans
la racine du dataset. Toute autre valeur est traitée comme un script Python
intégré et ne s'exécute que si `allow_download_scripts=True`. Sans cette option,
le script est ignoré avec un avertissement et l'entraînement continue sur ce
qui se trouve sur le disque.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

Le flag contrôle l'exécution de code, pas l'accès réseau. Les téléchargements
d'URL ont lieu dans les deux cas ; seuls les blocs `download: |` l'exigent. La
CLI affiche un avertissement lorsque le flag est actif, et doctor ne l'active
jamais.

## Vérifier le dataset avant l'entraînement

`libreyolo doctor` lit un dataset de détection et signale ce qui échouerait
avant de faire intervenir un GPU. Il renvoie le code 1 lorsqu'il trouve des
erreurs, ce qui en fait une barrière adaptée à la CI.

<code-tabs name="doctor" />

Les vérifications se répartissent en six familles :

| Famille | Recherche |
|---|---|
| `config` | `names` absent, `nc` différent de `names`, splits absents ou vides, noms de classes en double |
| `files` | images sans fichier d'étiquette, étiquettes sans image, images listées mais absentes, collisions de noms de base |
| `labels` | lignes mal formées, identifiants de classe hors de `[0, nc)`, coordonnées hors de `[0, 1]`, boîtes d'aire nulle, boîtes minuscules ou énormes, boîtes en double, fichiers d'étiquettes identiques octet par octet |
| `balance` | classes sans instance ou avec peu d'instances, ratio de déséquilibre des classes, classes présentes dans un seul split, proportion d'images d'arrière-plan |
| `images` | fichiers indécodables, rotation EXIF, dispositions de canaux inhabituelles, images uniformes, doublons exacts et proches |
| `splits` | même image présente dans deux splits, de façon exacte ou presque identique |

`--only` et `--skip` acceptent un identifiant de vérification ou un préfixe de
famille, si bien que `skip=images,labels.tiny_object` est valide. `--fast`
supprime toute vérification qui doit décoder des pixels, à savoir les familles
`images` et `splits`.

Deux comportements sont à connaître. `--strict` fait aussi échouer le code de
sortie pour les avertissements, pas seulement pour les erreurs. De plus, doctor
ne couvre que les datasets de détection : un dataset de pose, de segmentation
ou de boîtes orientées est rejeté avec un message indiquant le type détecté, au
lieu d'être vérifié selon le mauvais contrat.

## Pages connexes

- [Hyperparamètres](/docs/train/hyperparameters) pour les arguments acceptés par
  `train()` une fois les données en place.
- [Validation et mesures](/docs/train/validation) pour l'évaluation sur le split
  `val` ou `test`.
