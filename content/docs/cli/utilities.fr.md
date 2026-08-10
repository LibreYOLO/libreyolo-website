---
title: utilitaires libreyolo
seo_title: "Référence des commandes utilitaires du CLI libreyolo"
description: "Les petites commandes LibreYOLO : version, checks, models, formats, cfg, info, metadata, enroll et compare, chacune avec ses arguments et ses valeurs par défaut."
lead: "Neuf commandes qui rapportent ou inspectent au lieu de calculer. Elles affichent les informations sur l'environnement, l'inventaire des modèles et des formats, les valeurs par défaut résolues, les détails d'un checkpoint, et elles construisent et interrogent une galerie de visages."
keywords: ["commandes cli libreyolo", "libreyolo version", "vérifier installation libreyolo", "lister les modèles libreyolo", "formats export libreyolo", "configuration par défaut libreyolo", "métadonnées checkpoint pytorch", "reconnaissance faciale python cli", "comparer deux visages python"]
last_verified: "1.5.0"
meta:
  - label: Commandes
    value: version, checks, models, formats, cfg, info, metadata, enroll, compare
    mono: true
  - label: Sortie
    value: "stdout, en texte ou, avec json=true, sous forme d'un seul objet portant schema_version"
snippets:
  examples:
    - label: Environnement
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Ce qui est disponible
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Inspecter un checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
---

## Synopsis

```bash
libreyolo <command> [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi :
`model=x` et `--model x` sont le même argument. Toutes les commandes de cette
page écrivent leur résultat sur stdout et acceptent `json=true` et `quiet=true`.

La commande racine possède un flag qui lui est propre, `libreyolo --version`,
qui affiche la chaîne de version puis s'arrête. La sortie est plus réduite que
celle de la commande `version` ci-dessous.

## version

Affiche la version de LibreYOLO ainsi que les versions de Python, de torch et
de CUDA avec lesquelles elle s'exécute.

```bash
libreyolo version
```

| Argument | Défaut | Signification |
|---|---|---|
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

## checks

Affiche l'environnement plus en détail : Python, torch, CUDA, cuDNN, chaque
GPU détecté avec son nom et sa mémoire, et la version installée de chaque
paquet optionnel utilisé par les chemins d'export.

```bash
libreyolo checks
```

| Argument | Défaut | Signification |
|---|---|---|
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

La liste des paquets couvre `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` et `scipy`. Un paquet qui n'est pas installé est signalé comme
tel plutôt qu'omis : un export qui échoue se rattache ainsi à une dépendance
manquante depuis cette seule commande.

## models

Liste chaque famille de modèles avec ses tâches, ses tailles, les noms CLI qui
pointent vers ses checkpoints, et la résolution d'entrée de chaque taille.

```bash
libreyolo models
```

| Argument | Défaut | Signification |
|---|---|---|
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

Une famille dont la dépendance optionnelle n'est pas installée est indiquée
comme indisponible, accompagnée de la ligne `pip install` qui la rendrait
disponible. Les noms CLI sont ce que `model=` accepte comme raccourci :
`yolox-s` correspond à `LibreYOLOXs.pt`, et les tâches autres que la détection
portent leur suffixe de tâche.

## formats

Liste les formats d'export que l'environnement installé peut produire, avec
l'extension de fichier de chaque format et sa prise en charge de FP16 et INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argument | Défaut | Signification |
|---|---|---|
| `family` | | Affiche les niveaux de prise en charge pour une famille de modèles. `model=` est accepté comme la même option |
| `task` | | Tâche canonique du modèle. La tâche par défaut de la famille si non défini |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

Sans `family`, la sortie se limite à l'inventaire des formats. Avec, chaque
format reçoit le niveau de prise en charge pour cette famille et cette tâche,
la raison de ce niveau, et toute contrainte qui s'y rattache. Une famille
inconnue, ou une tâche que la famille ne prend pas en charge, est une erreur
d'utilisation.

Les alias de format apparaissent à côté de leur nom canonique : `engine` pour
`tensorrt`, `litert` pour `tflite`.

## cfg

Affiche la configuration par défaut résolue : les valeurs par défaut
d'entraînement, celles de validation, celles de prédiction, et les surcharges
par famille.

```bash
libreyolo cfg
```

| Argument | Défaut | Signification |
|---|---|---|
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

Les valeurs sont lues dans les dataclasses de configuration, pas dans une
copie : c'est donc la référence sur ce qu'un entraînement utilisera lorsque
vous ne passez pas d'argument. `family_overrides` est la section qui explique
pourquoi une famille s'est entraînée avec des réglages que vous n'avez pas
demandés. Voir [`libreyolo train`](/docs/cli/train) pour la façon dont ces
surcharges sont appliquées.

## info

Charge un modèle sur le CPU et rapporte sa famille, sa taille, son nombre de
paramètres, ses classes, et le niveau d'export pour chaque format.

```bash
libreyolo info model=<name|path>
```

| Argument | Défaut | Signification |
|---|---|---|
| `model` | | Nom du modèle ou chemin vers les poids. Obligatoire |
| `detailed` | `false` | Inclut le détail paramètre par paramètre |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

## metadata

Lit les métadonnées d'un checkpoint sans construire de modèle, et les valide
contre le schéma de checkpoint LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argument | Défaut | Signification |
|---|---|---|
| `path` | | Chemin vers un checkpoint `.pt`. Obligatoire |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

Les entrées volumineuses qui portent des tenseurs sont résumées plutôt
qu'affichées, pour que la sortie reste lisible sur un checkpoint
d'entraînement complet. Un checkpoint qui n'existe pas se termine sur
`checkpoint_not_found`, et un checkpoint dont les métadonnées échouent à la
validation affiche les erreurs et se termine sur `1`.

## enroll

Construit une galerie de visages à partir d'une arborescence d'un dossier par
personne, pour que les prédictions ultérieures puissent nommer les visages
qu'elles trouvent.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argument | Défaut | Signification |
|---|---|---|
| `model` | | Modèle d'embedding de visages, chemin ou nom. Obligatoire |
| `source` | | Arborescence d'un dossier par personne, `source/<identity>/*.jpg`. Obligatoire |
| `gallery` | | Fichier de galerie `.npz` en sortie. Complété sur place s'il existe déjà. Obligatoire |
| `face_detector` | | Détecteur de visages : un `.onnx` YuNet ou un détecteur LibreYOLO. Le détecteur par défaut de la famille si non défini |
| `device` | `auto` | Périphérique : `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

```bash
# people/ contient un dossier par identité, dont le nom devient l'identité.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Le nom du sous-dossier est l'identité. Une image de référence sans visage
détectable est ignorée avec une ligne sur stderr et les autres continuent ;
une source sans sous-dossier d'identité, ou dans laquelle aucun visage n'a été
trouvé, est une erreur.

Passez le fichier obtenu à
[`libreyolo predict`](/docs/cli/predict) sous la forme `gallery=people.npz`
pour que les détections portent une identité et un score de correspondance.

## compare

Rapporte la similarité cosinus entre deux images de visage et indique si elle
franchit le seuil de même identité.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argument | Défaut | Signification |
|---|---|---|
| `model` | | Modèle d'embedding de visages, chemin ou nom. Obligatoire |
| `source` | | Première image. Obligatoire |
| `source2` | | Seconde image, à comparer à la première. Obligatoire |
| `face_detector` | | Détecteur de visages : un `.onnx` YuNet ou un détecteur LibreYOLO |
| `threshold` | `0.4` | Seuil de similarité cosinus pour la décision de même identité |
| `device` | `auto` | Périphérique : `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Masque stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` est enregistré comme second nom de cette commande et prend
les mêmes arguments.

`compare` comme `enroll` ont besoin d'un modèle dont la tâche est l'embedding
de visages. Tout le reste se termine sur `config_unsupported`. Les chemins
d'images locaux et les URL `http` ou `https` sont acceptés comme sources.

## Exemples

<code-tabs name="examples" />

## Notes

stdout porte le résultat ; la progression et les avertissements vont sur
stderr. `json=true` affiche un seul objet avec `schema_version`, et c'est la
forme à lire depuis un script. La sortie texte est celle par défaut, destinée
à être lue par une personne.

Les codes de sortie suivent la même table que le reste du CLI : `0` en cas de
succès, `2` pour une erreur d'utilisation ou de configuration, `3` quand une
source est introuvable, `4` quand un modèle ou un checkpoint ne peut pas être
chargé, et `1` pour les autres échecs à l'exécution.

À voir aussi : [`libreyolo doctor`](/docs/cli/doctor), la commande
d'inspection côté dataset, et [`libreyolo profile`](/docs/cli/profile), celle
côté performance.
