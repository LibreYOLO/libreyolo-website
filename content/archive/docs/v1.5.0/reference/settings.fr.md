---
title: Paramètres
seo_title: Variables d'environnement et répertoires de LibreYOLO
description: >-
  Toutes les variables d'environnement lues par LibreYOLO, les répertoires
  écrits, les tokens nécessaires et les réglages qui modifient le chemin de code
  exécuté.
lead: >-
  LibreYOLO ne possède aucun fichier de configuration. Les comportements qui ne
  sont pas des arguments de fonction sont contrôlés par des variables
  d'environnement et quelques répertoires conventionnels, tous énumérés ici.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - dossier poids libreyolo
  - cache libreyolo
last_verified: 1.5.0
verification: "Variables trouvées en recherchant os.environ et os.getenv dans libreyolo/**/*.py en v1.5.0\_; sémantique lue à chaque site d'utilisation. Conventions de répertoires lues dans libreyolo/data/utils.py, libreyolo/utils/download.py, libreyolo/export/exporter.py, libreyolo/models/base/model.py et libreyolo/models/sam3dbody/mhr_body.py."
snippets:
  usage:
    - label: Déplacer la racine des datasets
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Lire la valeur résolue depuis Python
      language: python
      code: >
        from libreyolo.data import DATASETS_DIR


        # Valeur par défaut : ~/datasets ; LIBREYOLO_DATASETS_DIR la remplace au
        moment de l'import.

        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Variables d'environnement

| Variable | Valeur par défaut | Effet |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Racine des datasets. Lue une fois lors de l'import dans `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | non définie | Remplace le paramètre de validation `faster_coco_eval`. `1`, `true`, `yes` ou `on` impose le backend rapide, toute autre valeur le désactive et l'absence de valeur s'en remet au paramètre de configuration |
| `LIBREYOLO_KERNELS` | non définie | Sélection des kernels. `off` ou `reference` impose les implémentations de référence\u00a0; toute autre valeur ne sélectionne que les implémentations enregistrées sous ce nom |
| `LIBREYOLO_QUANT_KERNELS` | non définie | Ancien alias de `LIBREYOLO_KERNELS`, lu uniquement lorsque ce dernier n'est pas défini |
| `LIBREYOLO_HUB_KERNELS` | non définie | `0`, `false`, `off` ou `no` désactive le chargement de kernels depuis Hugging Face Hub. Toute autre valeur, y compris l'absence de valeur, le laisse activé |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Emplacement du modèle corporel MHR utilisé par la tâche `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | non définie | Doit valoir exactement `1`, `true`, `yes` ou `on` pour exposer l'assistant LocateAnything dans l'outil d'étiquetage. Toute autre valeur le laisse désactivé |
| `SAM_3D_BODY_PATH` | non définie | Chemin du package SAM 3D Body pour la famille de maillage, lorsqu'il n'est pas transmis au constructeur |
| `HF_TOKEN` | non définie | Token d'accès Hugging Face utilisé pour les dépôts protégés |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` est lue au moment de l'import. La définir après
l'import de `libreyolo.data` ne modifie donc pas `DATASETS_DIR`.

Les kernels du Hub nécessitent une double activation. La récupération à
l'exécution ne se produit que lorsque le package facultatif `kernels` est
installé. Installer `libreyolo[hub-kernels]` constitue donc l'activation et
`LIBREYOLO_HUB_KERNELS=0` la désactivation. Une installation sans cet extra
n'est affectée dans aucun des deux cas.

La sélection des kernels court-circuite aussi les imports. Lorsque
`LIBREYOLO_KERNELS` impose `off` ou `reference`, les fournisseurs accélérés
inclus dans l'arborescence ne sont jamais importés. Le registre contrôlé par
ces trois variables est documenté dans la page sur les
[kernels](/docs/reference/kernels).

## Variables définies par la bibliothèque

Ces variables sont écrites et non lues. Leur définition manuelle n'est donc
pas une méthode prise en charge.

| Variable | Définie par |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | L'assistant de spawn DDP, une valeur par processus worker |
| `CUDA_VISIBLE_DEVICES` | Temporairement restreinte pendant la configuration distribuée, puis restaurée |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Définie à `1` par les trainers EC avec `setdefault`, une valeur existante l'emporte donc |
| `MOMENTUM_ENABLED` | Définie avec `setdefault` par le chargeur de la famille de maillage |

`LOCAL_RANK` sert également de signal de mode distribué\u00a0: sa présence dans
l'environnement indique au code d'entraînement qu'il s'exécute sous DDP.

## Variables des loggers

Les loggers d'entraînement facultatifs utilisent des valeurs d'environnement
par défaut pour le nom du projet.

| Variable | Valeur par défaut | Utilisée par |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Logger Weights and Biases, lorsqu'aucun projet n'est transmis |
| `COMET_PROJECT_NAME` | `libreyolo` | Logger Comet, lorsqu'aucun projet n'est transmis |

L'authentification auprès de ces services suit leurs propres outils et non
LibreYOLO.

## Tokens

`HF_TOKEN` est le token d'accès Hugging Face. Lorsqu'il n'est pas défini, le
token est lu dans `~/.cache/huggingface/token`, où une connexion avec le CLI
Hugging Face l'écrit. Les deux méthodes fonctionnent.

Un token n'est requis que pour les dépôts protégés. SAM 3 est l'exemple
fourni\u00a0: ses poids sont téléchargés depuis un dépôt protégé par une licence
personnalisée, vous devez donc accepter les conditions sur la page du dépôt et
authentifier la session.

## Répertoires

| Chemin | Contenu |
|---|---|
| `weights/` | Checkpoints téléchargés, snapshots Hugging Face téléchargés et artefacts exportés |
| `~/datasets` | Racine des datasets, sauf indication contraire de `LIBREYOLO_DATASETS_DIR` |
| `~/.cache/huggingface/token` | Token Hugging Face lorsqu'il n'est pas dans `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Modèle corporel MHR, sauf indication contraire de `LIBREYOLO_MHR_PATH` |
| `runs/track/` | Sortie par défaut de `model.track(save=True)` |

`weights/` est relatif au répertoire de travail. Un simple nom de fichier est
résolu dans celui-ci. `LibreYOLO("LibreYOLO9t.pt")` recherche donc
`weights/LibreYOLO9t.pt` et le télécharge à cet emplacement s'il est absent.
`model.export()` écrit dans ce même répertoire lorsqu'aucun `output_path` n'est
fourni. Les niveaux frères téléchargent les snapshots multifichiers dans
`weights/<Prefix><size>/`.

## Comportement des téléchargements

Les téléchargements de poids sont retentés trois fois avec backoff, reprennent
depuis un fichier partiel et sont protégés par un fichier de verrouillage pour
éviter que deux processus récupèrent simultanément le même checkpoint. Une
famille qui télécharge depuis un hôte tiers peut épingler une somme de contrôle
et échouer de façon sécurisée en cas de différence.

Certains téléchargements affichent un avis de licence avant de démarrer. Ces
avis font partie du chemin de téléchargement et ne peuvent pas être désactivés
par configuration.

## Backend de validation

`model.val()` accepte `faster_coco_eval=True` par défaut et se rabat sur
pycocotools lorsque le package n'est pas installé, avec un avertissement
unique. Définir `LIBREYOLO_FASTER_COCO_EVAL` remplace le paramètre propre à
l'appel. Un banc de benchmark qui ne peut pas modifier les configurations par
exécution doit utiliser cette variable. Le backend réellement exécuté est
indiqué dans `model.last_eval_backend`.

## Scripts de téléchargement des datasets

Le champ `download` d'un fichier YAML de dataset peut contenir du Python. Il
n'est exécuté que si `allow_download_scripts=True` est transmis à l'appel qui
lit ce fichier. Il s'agit d'un argument de fonction de `val()` et `export()`,
et non d'une variable d'environnement.

