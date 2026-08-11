---
title: Logger degli esperimenti
seo_title: Logger degli esperimenti e callback in LibreYOLO
description: >-
  Invia le metriche di addestramento a TensorBoard, MLflow, Weights & Biases,
  Comet, ClearML, Neptune o DVCLive, e scrivi il tuo callback sui quattro hook
  di addestramento.
lead: >-
  Ogni famiglia addestrabile emette quattro eventi di addestramento. I logger
  integrati sono oggetti callback in ascolto su quegli stessi eventi, quindi
  l'integrazione con un backend e un hook scritto da te usano una sola
  interfaccia.
keywords:
  - tensorboard training
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callback addestramento
  - metriche training csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Per nome
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Istanza configurata
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
    - label: Una semplice funzione
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Un oggetto con più hook
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
    - label: Seguire un run nel browser
      language: bash
      code: |
        libreyolo monitor                     # il run più recente sotto runs/
        libreyolo monitor runs/train/exp      # un run specifico
source_hash: de035acbaed32804
---

## Attivare un logger

`loggers=` accetta un nome registrato, un'istanza configurata o un iterabile che
mescola le due cose.

<code-tabs name="logger" />

I nomi non distinguono maiuscole e minuscole. L'insieme registrato è
`tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` e
`dvc`, dove l'ultimo è un alias di `dvclive`. Qualsiasi altro valore solleva
subito un errore ed elenca i nomi validi. Non esiste un valore che li abiliti
tutti, e non c'è nessun flag da CLI: `loggers=` è un argomento Python.

## Cosa registra ogni backend

Scrivono tutti gli stessi nomi di metrica, quindi la dashboard ha lo stesso
aspetto qualunque backend tu scelga:

| Chiave | Valore |
|---|---|
| `train/loss` | la loss media di addestramento dell'epoca |
| `train/loss/<component>` | ogni componente della loss riportata dalla famiglia |
| `lr/<group>` | il learning rate di ogni gruppo di parametri dell'optimizer |
| `val/<metric>` | ogni metrica di validazione, con il prefisso `metrics/` rimosso |
| `time/epoch_seconds` | il tempo reale impiegato dall'epoca |

Lo step è l'epoca contata a partire da 1. La configurazione di addestramento
completamente risolta viene registrata come parametri all'avvio
dell'addestramento, e il nome del run è per impostazione predefinita
`<family><size>-<task>`, per esempio `yolo9s-detect`.

Alla fine dell'addestramento i backend che supportano gli artefatti caricano
`results.csv`, `train_config.yaml` e `summary.json` quando esistono, più
`weights/best.pt` con `log_checkpoints=True`. TensorBoard non carica niente,
perché non ha un concetto di artefatto. Nessun logger carica le immagini dei
grafici di validazione.

## Comportamento in caso di errore

Un pacchetto del backend mancante solleva un errore alla costruzione, indicando
il comando di installazione, perché chiedere un logger e non ottenere niente in
silenzio nasconde un bug.

Un errore del backend durante il run fa l'opposto. La prima eccezione sollevata
da un handler disabilita quel logger per il resto del run, la registra nel log,
chiude il run del backend segnandolo come fallito, e l'addestramento continua. Un
tracking server che va giù non ti costa l'addestramento.

## I backend

Ognuno richiede il proprio extra.

| Nome | Extra | Costruttore |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Importa le classi da `libreyolo.training`.

Note specifiche dei singoli backend che vale la pena conoscere prima del primo
run:

I file di evento di TensorBoard finiscono per impostazione predefinita in
`<save_dir>/tensorboard`. Li guardi con `tensorboard --logdir runs/train`.

MLflow 3.x ha deprecato lo store su file locale `./mlruns` e solleva un errore se
non è impostato `MLFLOW_ALLOW_FILE_STORE=true`. Per un tracking locale senza
server, passa invece un URI di database, come nello snippet qui sopra, e leggilo
con `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases ripiega sulla variabile d'ambiente `WANDB_PROJECT` e poi su
`libreyolo`. Comet ripiega su `COMET_PROJECT_NAME` e poi su `libreyolo`, e prende
le credenziali dalla propria configurazione; `online=False` dà un esperimento
offline. ClearML crea un task nuovo, riporta la configurazione sotto
`TrainConfig` e disabilita la cattura automatica del framework, così le metriche
non vengono riportate due volte. Neptune usa il client attuale `neptune-scale`
invece del pacchetto legacy, e `mode="offline"` registra in locale.

DVCLive scrive in `<save_dir>/dvclive`. Costruisce il suo albero di riepilogo a
partire da `/`, e non può tenere un float in un percorso che è anche un genitore,
quindi `train/loss/box` viene scritto come `train/loss.box` mentre `train/loss`
mantiene il suo nome. LibreYOLO disattiva anche i soliti default di DVCLive, che
salvano un esperimento DVC e scrivono un `dvc.yaml` nella radice, così un logger
attivato esplicitamente non crea nessuno stato di controllo di versione fuori
dalla directory del run; passa `save_dvc_exp=True` o un `dvcyaml=` esplicito per
riaverli.

Neptune è escluso di proposito da `libreyolo[all]`: il suo client stabile
richiede protobuf sotto la 7 mentre l'extra TFLite richiede protobuf 7. Installa
`libreyolo[neptune]` in un ambiente senza l'extra TFLite.

## Scrivere un callback

Gli stessi quattro eventi guidano tutto.

<code-tabs name="callback" />

| Evento | Quando | Cosa porta |
|---|---|---|
| `TrainStartEvent` | dopo il setup, prima dell'epoca 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | dopo ogni epoca, addestramento e validazione | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | dopo che l'addestramento è terminato | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | se l'addestramento solleva un'eccezione | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Un semplice callable riceve solo `TrainEpochEvent`. Un oggetto può implementare
un sottoinsieme qualsiasi di `on_train_start`, `on_train_epoch_end`,
`on_train_end` e `on_train_exception`; i metodi mancanti vengono saltati.

`TrainStartEvent.config` è la configurazione completamente risolta, i kwargs
dell'utente uniti ai default della famiglia, come mapping di sola lettura. Gli
eventi sono dataclass congelate e i loro mapping sono di sola lettura, quindi un
callback non può cambiare il run scrivendoci dentro.

Un'eccezione sollevata da `on_train_start`, `on_train_epoch_end` o
`on_train_end` si propaga e termina il run. Solo `on_train_exception` è protetto,
così non può mascherare il fallimento originale.

Nell'addestramento multi-GPU i callback vengono invocati solo sul rank 0. Con lo
spawn DDP automatico devono anche essere serializzabili con pickle, il che
significa una classe o una funzione a livello di modulo invece di una closure o
di una lambda. Vedi [Addestramento multi-GPU](/docs/train/multi-gpu).

## Cosa scrive comunque ogni run

Tre file finiscono nella directory del run senza nessuna configurazione, con ogni
famiglia:

| File | Scritto | Contenuto |
|---|---|---|
| `status.json` | in modo atomico, a ogni epoca e all'avvio, alla fine e in caso di fallimento | `state` con valore `running`, `completed` o `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, le `metrics` più recenti, `best_metric`, `best_epoch`, e un oggetto `error` in caso di fallimento |
| `metrics.jsonl` | in append una volta per epoca | una riga JSON per epoca, con lo stesso schema di `results.csv` |
| `train.log` | live | l'output su console del run |

`status.json` è la lettura economica per uno script o un agente che interroga un
run, e la scrittura atomica fa sì che un lettore non veda mai un file scritto a
metà.

`results.csv` e `summary.json` sono separati e dipendono dalla famiglia. Vengono
scritti per YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC e
DINOv2, e non per le altre famiglie. `results.csv` riceve una riga per epoca con
le componenti della loss, le metriche di validazione e i learning rate come
colonne, e la sua intestazione si allarga quando compare una nuova colonna. Alla
ripresa viene tagliato fino alle righe precedenti all'epoca da cui si riprende,
invece di duplicarle.

Accanto a questi, il trainer scrive sempre `train_config.yaml` al setup e i
checkpoint sotto `weights/`.

## Seguire un run in tempo reale

<code-tabs name="monitor" />

`libreyolo monitor` serve una dashboard nel browser costruita sui file qui sopra
usando solo la libreria standard: grafici delle metriche, la coda del log e le
eventuali immagini di validazione, che si aggiornano mentre il run è attivo. È di
sola lettura e non tocca mai il processo di addestramento, quindi si aggancia a
un run live, riapre uno finito o ispeziona uno andato in crash.

## Correlati

- [Validazione e metriche](/docs/train/validation) per capire cosa significano le
  chiavi `val/` e come aggiungere una loss di validazione.
- [Prestazioni dell'addestramento](/docs/train/performance) per il profiler, che
  è uno strumento diverso con una domanda diversa.
