---
title: Loggers d'expériences
seo_title: Loggers d'expériences et callbacks dans LibreYOLO
description: >-
  Envoyez les mesures d'entraînement vers TensorBoard, MLflow, Weights & Biases,
  Comet, ClearML, Neptune ou DVCLive, et écrivez votre propre callback sur les
  quatre hooks d'entraînement.
lead: >-
  Chaque famille entraînable émet quatre événements d'entraînement. Les loggers
  intégrés sont des objets callback qui écoutent ces mêmes événements, si bien
  qu'une intégration de backend et un hook personnalisé utilisent une interface
  unique.
keywords:
  - entraînement tensorboard
  - suivi mlflow
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callbacks entraînement
  - métriques entraînement csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Par nom
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Instance configurée
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Une fonction simple
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Un objet avec plusieurs hooks
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Suivre une exécution dans le navigateur
      language: bash
      code: >
        libreyolo monitor                     # exécution la plus récente sous
        runs/

        libreyolo monitor runs/train/exp      # une exécution précise
source_hash: de035acbaed32804
---

## Activer un logger

`loggers=` accepte un nom enregistré, une instance configurée ou un itérable
qui mélange les deux.

<code-tabs name="logger" />

Les noms sont insensibles à la casse. L'ensemble enregistré est `tensorboard`,
`mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` et `dvc`, ce dernier
étant un alias de `dvclive`. Toute autre valeur provoque immédiatement une
erreur qui liste les noms valides. Aucune valeur ne les active tous et il
n'existe aucun flag CLI : `loggers=` est un argument Python.

## Données enregistrées par chaque backend

Tous écrivent les mêmes noms de mesures. Un tableau de bord se présente donc de
la même façon quel que soit votre choix :

| Clé | Valeur |
|---|---|
| `train/loss` | loss d'entraînement moyenne de l'époque |
| `train/loss/<component>` | chaque composant de loss rapporté par la famille |
| `lr/<group>` | learning rate de chaque groupe de paramètres de l'optimiseur |
| `val/<metric>` | chaque mesure de validation, sans son préfixe `metrics/` |
| `time/epoch_seconds` | temps réel de l'époque |

L'étape est l'époque indexée à partir de 1. La configuration d'entraînement
entièrement résolue est enregistrée comme paramètres au début de l'entraînement,
et le nom d'exécution vaut par défaut `<family><size>-<task>`, par exemple
`yolo9s-detect`.

À la fin de l'entraînement, les backends qui prennent en charge les artefacts
envoient `results.csv`, `train_config.yaml` et `summary.json` lorsqu'ils
existent, ainsi que `weights/best.pt` avec `log_checkpoints=True`. TensorBoard
n'envoie rien, car il n'a aucun concept d'artefact. Aucun logger n'envoie les
images de graphiques de validation.

## Comportement en cas d'échec

Un package de backend absent provoque une erreur à la construction, qui indique
la commande d'installation, car demander un logger et ne rien obtenir
silencieusement masque un bug.

Une défaillance du backend pendant l'exécution produit l'effet inverse. La
première exception d'un handler désactive ce logger pour le reste de
l'exécution, la journalise, termine l'exécution du backend comme un échec, puis
l'entraînement continue. La panne d'un serveur de suivi ne vous coûte pas
l'entraînement.

## Backends

Chacun nécessite son propre extra.

| Nom | Extra | Constructeur |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Importez les classes depuis `libreyolo.training`.

Quelques remarques propres aux backends sont à connaître avant la première
exécution :

Les fichiers d'événements TensorBoard vont par défaut dans
`<save_dir>/tensorboard`. Consultez-les avec
`tensorboard --logdir runs/train`.

MLflow 3.x a déprécié le stockage local `./mlruns` et provoque une erreur sauf
si `MLFLOW_ALLOW_FILE_STORE=true`. Pour un suivi local sans serveur, passez
plutôt une URI de base de données comme dans l'extrait ci-dessus, puis lisez-la
avec `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases se rabat sur la variable d'environnement `WANDB_PROJECT`, puis
sur `libreyolo`. Comet se rabat sur `COMET_PROJECT_NAME`, puis sur `libreyolo`,
et récupère les identifiants dans sa propre configuration ; `online=False`
produit une expérience hors ligne. ClearML crée une nouvelle tâche, rapporte la
configuration sous `TrainConfig` et désactive la capture automatique du
framework pour éviter de rapporter les mesures deux fois. Neptune utilise le
client `neptune-scale` actuel plutôt que l'ancien package, et `mode="offline"`
journalise localement.

DVCLive écrit dans `<save_dir>/dvclive`. Il construit son arborescence de résumé
à partir de `/` et ne peut pas contenir un nombre flottant à un chemin qui est
aussi un parent. `train/loss/box` est donc écrit comme `train/loss.box`, tandis
que `train/loss` conserve son nom. LibreYOLO désactive également les valeurs par
défaut habituelles de DVCLive qui enregistrent une expérience DVC et écrivent un
fichier `dvc.yaml` à la racine. Un logger activé explicitement ne crée donc
aucun état de contrôle de version hors du répertoire d'exécution ; passez
`save_dvc_exp=True` ou un `dvcyaml=` explicite pour les rétablir.

Neptune est délibérément exclu de `libreyolo[all]` : son client stable exige une
version de protobuf inférieure à 7, tandis que l'extra TFLite exige protobuf 7.
Installez `libreyolo[neptune]` dans un environnement sans l'extra TFLite.

## Écrire un callback

Les quatre mêmes événements pilotent tout.

<code-tabs name="callback" />

| Événement | Moment | Contenu |
|---|---|---|
| `TrainStartEvent` | après la configuration, avant l'époque 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | après chaque époque, entraînement et validation | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | à la fin de l'entraînement | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | si l'entraînement lève une erreur | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Un callable simple reçoit uniquement `TrainEpochEvent`. Un objet peut
implémenter n'importe quel sous-ensemble de `on_train_start`,
`on_train_epoch_end`, `on_train_end` et `on_train_exception` ; les méthodes
absentes sont ignorées.

`TrainStartEvent.config` est la configuration entièrement résolue, où les kwargs
de l'utilisateur sont fusionnés avec les valeurs par défaut de la famille, sous
forme de correspondance en lecture seule. Les événements sont des dataclasses
gelées et leurs correspondances sont en lecture seule. Un callback ne peut donc
pas modifier l'exécution en y écrivant.

Une exception issue de `on_train_start`, `on_train_epoch_end` ou `on_train_end`
se propage et termine l'exécution. Seul `on_train_exception` est protégé, afin
qu'il ne puisse pas masquer l'échec d'origine.

Pendant un entraînement multi-GPU, les callbacks se déclenchent uniquement sur
le rang 0. Avec le spawn DDP automatique, ils doivent aussi être sérialisables
par pickle, ce qui exige une classe ou une fonction au niveau du module plutôt
qu'une closure ou une lambda. Consultez l'
[entraînement multi-GPU](/docs/train/multi-gpu).

## Fichiers toujours écrits par chaque exécution

Trois fichiers sont placés dans le répertoire d'exécution sans aucune
configuration, pour chaque famille :

| Fichier | Écriture | Contenu |
|---|---|---|
| `status.json` | atomiquement, à chaque époque ainsi qu'au début, à la fin et en cas d'échec | `state` parmi `running`, `completed` ou `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, dernières `metrics`, `best_metric`, `best_epoch` et objet `error` en cas d'échec |
| `metrics.jsonl` | ajout d'une ligne par époque | une ligne JSON par époque, selon le même schéma que `results.csv` |
| `train.log` | en direct | sortie de console de l'exécution |

`status.json` constitue la lecture peu coûteuse pour un script ou un agent qui
interroge une exécution, et l'écriture atomique garantit qu'un lecteur ne voit
jamais un fichier partiellement écrit.

`results.csv` et `summary.json` sont séparés et dépendent de la famille. Ils
sont écrits pour YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC et
DINOv2, mais pas pour les autres familles. `results.csv` reçoit une ligne par
époque avec les composants de loss, les mesures de validation et les learning
rates sous forme de colonnes, et son en-tête s'élargit lorsqu'une nouvelle
colonne apparaît. Lors d'une reprise, il est ramené aux lignes antérieures à
l'époque reprise au lieu de les dupliquer.

En parallèle, le trainer écrit toujours `train_config.yaml` pendant la
configuration et les checkpoints sous `weights/`.

## Suivre une exécution en direct

<code-tabs name="monitor" />

`libreyolo monitor` fournit un tableau de bord dans le navigateur à partir des
fichiers ci-dessus en utilisant uniquement la bibliothèque standard :
graphiques de mesures, fin du journal et éventuelles images de validation, avec
actualisation tant que l'exécution est active. Il est en lecture seule et ne
touche jamais au processus d'entraînement. Il peut donc se rattacher à une
exécution active, rouvrir une exécution terminée ou inspecter une exécution qui
a planté.

## Pages connexes

- [Validation et mesures](/docs/train/validation) pour la signification des clés
  `val/` et la façon d'ajouter une loss de validation.
- [Performances d'entraînement](/docs/train/performance) pour le profileur, un
  outil différent qui répond à une autre question.
