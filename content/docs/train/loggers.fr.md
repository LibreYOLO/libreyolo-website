---
title: Systèmes de journalisation des expériences
seo_title: Systèmes de journalisation et callbacks dans LibreYOLO
description: >-
  Envoyer les métriques d'entraînement à TensorBoard, MLflow, Weights & Biases,
  Comet, ClearML, Neptune ou DVCLive, et écrire votre propre callback sur les
  quatre hooks d'entraînement.
lead: >-
  Chaque famille entraînable émet quatre événements d'entraînement. Les systèmes
  de journalisation intégrés sont des objets callback à l'écoute de ces mêmes
  événements. Une intégration de backend et un hook personnalisé partagent donc
  une interface.
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
  - surveillance libreyolo
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
    - label: Une fonction ordinaire
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

        libreyolo monitor runs/train/exp      # exécution précise
source_hash: de035acbaed32804
---

## Activer un système de journalisation

`loggers=` accepte un nom enregistré, une instance configurée ou un itérable
qui mélange les deux.

<code-tabs name="logger" />

Les noms sont insensibles à la casse. L'ensemble enregistré comprend
`tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` et
`dvc`, ce dernier étant un alias de `dvclive`. Toute autre valeur déclenche
immédiatement une erreur qui énumère les noms valides. Aucune valeur ne les
active tous et il n'existe aucune option CLI. `loggers=` est un argument Python.

## Données enregistrées par chaque backend

Tous écrivent les mêmes noms de métriques. Un tableau de bord garde donc la
même apparence quel que soit votre choix.

| Clé | Valeur |
|---|---|
| `train/loss` | Perte d'entraînement moyenne de l'époque |
| `train/loss/<component>` | Chaque composant de perte rapporté par la famille |
| `lr/<group>` | Taux d'apprentissage de chaque groupe de paramètres de l'optimiseur |
| `val/<metric>` | Chaque métrique de validation, sans son préfixe `metrics/` |
| `time/epoch_seconds` | Temps réel de l'époque |

L'étape est le numéro d'époque à partir de 1. La configuration d'entraînement
entièrement résolue est journalisée comme paramètres au début. Le nom
d'exécution vaut par défaut `<family><size>-<task>`, par exemple
`yolo9s-detect`.

À la fin, les backends compatibles avec les artefacts téléversent
`results.csv`, `train_config.yaml` et `summary.json` lorsqu'ils existent, ainsi
que `weights/best.pt` avec `log_checkpoints=True`. TensorBoard ne téléverse rien
car il ne possède aucun concept d'artefact. Aucun système de journalisation ne
téléverse les graphiques de validation.

## Comportement en cas d'échec

L'absence du paquet d'un backend déclenche une erreur à sa construction avec la
commande d'installation. Demander un système de journalisation sans rien
obtenir masquerait un bogue.

Un échec du backend pendant l'exécution produit le comportement inverse. La
première exception d'un gestionnaire désactive ce système pour le reste de
l'exécution, la journalise, termine l'exécution du backend comme échouée, puis
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

Quelques particularités des backends sont utiles avant la première exécution.

Les fichiers d'événements TensorBoard sont placés par défaut dans
`<save_dir>/tensorboard`. Affichez-les avec
`tensorboard --logdir runs/train`.

MLflow 3.x a déprécié le stockage local `./mlruns` et déclenche une erreur sauf
si `MLFLOW_ALLOW_FILE_STORE=true`. Pour un suivi local sans serveur, transmettez
plutôt une URI de base de données comme dans l'extrait, puis lisez-la avec
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases revient à la variable d'environnement `WANDB_PROJECT`, puis à
`libreyolo`. Comet revient à `COMET_PROJECT_NAME`, puis à `libreyolo`, et tire
ses identifiants de sa propre configuration. `online=False` produit une
expérience hors ligne. ClearML crée une nouvelle tâche, rapporte la
configuration sous `TrainConfig` et désactive la capture automatique des
frameworks afin de ne pas rapporter deux fois les métriques. Neptune emploie le
client `neptune-scale` actuel plutôt que l'ancien paquet, et `mode="offline"`
journalise localement.

DVCLive écrit dans `<save_dir>/dvclive`. Il construit son arbre de résumé à
partir de `/` et ne peut pas stocker un nombre flottant à un chemin qui sert
aussi de parent. `train/loss/box` devient donc `train/loss.box` tandis que
`train/loss` conserve son nom. LibreYOLO désactive également les comportements
habituels de DVCLive qui enregistrent une expérience DVC et écrivent un
`dvc.yaml` à la racine. Un système activé explicitement ne crée ainsi aucun
état de contrôle de version hors du répertoire d'exécution. Transmettez
`save_dvc_exp=True` ou un `dvcyaml=` explicite pour les réactiver.

Neptune est volontairement exclu de `libreyolo[all]`. Son client stable exige
une version de protobuf antérieure à 7, tandis que l'extra TFLite exige
protobuf 7. Installez `libreyolo[neptune]` dans un environnement dépourvu de
l'extra TFLite.

## Écrire un callback

Les quatre mêmes événements pilotent tout.

<code-tabs name="callback" />

| Événement | Moment | Contenu |
|---|---|---|
| `TrainStartEvent` | après la configuration, avant l'époque 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | après chaque époque, entraînement et validation | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | après la fin de l'entraînement | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | si l'entraînement déclenche une erreur | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Une fonction ordinaire reçoit uniquement `TrainEpochEvent`. Un objet peut
implémenter tout sous-ensemble de `on_train_start`, `on_train_epoch_end`,
`on_train_end` et `on_train_exception`. Les méthodes absentes sont ignorées.

`TrainStartEvent.config` est la configuration entièrement résolue, les
arguments utilisateur étant fusionnés avec les valeurs par défaut de la
famille, sous forme d'association en lecture seule. Les événements sont des
dataclasses figées et leurs associations sont en lecture seule. Un callback ne
peut donc pas modifier l'exécution en y écrivant.

Une exception de `on_train_start`, `on_train_epoch_end` ou `on_train_end` se
propage et termine l'exécution. Seul `on_train_exception` est protégé afin de
ne pas masquer l'échec d'origine.

En entraînement multi-GPU, les callbacks ne se déclenchent que sur le rang 0.
Avec la création DDP automatique, ils doivent aussi être sérialisables par
pickle, soit une classe ou une fonction au niveau du module plutôt qu'une
fermeture ou une lambda. Consultez
[l'entraînement multi-GPU](/docs/train/multi-gpu).

## Fichiers écrits par chaque exécution

Trois fichiers sont placés dans le répertoire d'exécution sans aucune
configuration, pour chaque famille.

| Fichier | Écriture | Contenu |
|---|---|---|
| `status.json` | atomiquement, à chaque époque et au démarrage, à la fin et en cas d'échec | `state` valant `running`, `completed` ou `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, dernières `metrics`, `best_metric`, `best_epoch` et objet `error` en cas d'échec |
| `metrics.jsonl` | une ligne ajoutée par époque | une ligne JSON par époque, avec le même schéma que `results.csv` |
| `train.log` | en direct | sortie console de l'exécution |

`status.json` est la lecture peu coûteuse pour un script ou un agent qui
interroge une exécution. Son écriture atomique empêche un lecteur de voir un
fichier partiellement écrit.

`results.csv` et `summary.json` sont distincts et dépendent de la famille. Ils
sont écrits pour YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC
et DINOv2, mais pas pour les autres familles. `results.csv` reçoit une ligne
par époque avec les composants de perte, les métriques de validation et les
taux d'apprentissage en colonnes. Son en-tête s'élargit lorsqu'une nouvelle
colonne apparaît. Lors d'une reprise, il est tronqué aux lignes antérieures à
l'époque reprise afin d'éviter les doublons.

Le programme d'entraînement écrit également toujours `train_config.yaml` à la
configuration et les checkpoints sous `weights/`.

## Suivre une exécution en direct

<code-tabs name="monitor" />

`libreyolo monitor` sert avec la seule bibliothèque standard un tableau de bord
dans le navigateur au-dessus de ces fichiers : graphiques des métriques, fin du
journal et éventuelles images de validation, avec actualisation pendant
l'exécution. Il est en lecture seule et ne touche jamais le processus
d'entraînement. Il peut donc se connecter à une exécution active, rouvrir une
exécution terminée ou inspecter une exécution interrompue.

## Voir aussi

- [Validation et métriques](/docs/train/validation) pour la signification des
  clés `val/` et l'ajout d'une perte de validation.
- [Performances d'entraînement](/docs/train/performance) pour le profileur, qui
  répond à une autre question.
