---
title: Loggers de experimentos
seo_title: "Loggers de experimentos y callbacks en LibreYOLO"
description: "Envía las métricas de entrenamiento a TensorBoard, MLflow, Weights & Biases, Comet, ClearML, Neptune o DVCLive, y escribe tu propio callback sobre los cuatro hooks de entrenamiento."
lead: "Toda familia entrenable emite cuatro eventos de entrenamiento. Los loggers integrados son objetos callback que escuchan esos mismos eventos, así que la integración con un backend y un hook propio usan una única interfaz."
keywords:
  - tensorboard entrenamiento
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callbacks de entrenamiento
  - metricas de entrenamiento csv
  - libreyolo monitor
last_verified: "1.5.0"
snippets:
  logger:
    - label: Por nombre
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Instancia configurada
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
    - label: Una función simple
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Un objeto con varios hooks
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEndEvent, TrainEpochEvent, TrainStartEvent


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
    - label: Seguir una ejecución en el navegador
      language: bash
      code: |
        libreyolo monitor                     # la ejecución más reciente bajo runs/
        libreyolo monitor runs/train/exp      # una ejecución concreta
---

## Activar un logger

`loggers=` acepta un nombre registrado, una instancia configurada o un iterable
que mezcle ambas cosas.

<code-tabs name="logger" />

Los nombres no distinguen mayúsculas de minúsculas. El conjunto registrado es
`tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` y
`dvc`, siendo este último un alias de `dvclive`. Cualquier otra cosa lanza un
error de inmediato y lista los nombres válidos. No hay ningún valor que los
active todos, y no hay flag de CLI: `loggers=` es un argumento de Python.

## Qué registra cada backend

Todos escriben los mismos nombres de métricas, así que un dashboard se ve igual
elijas el que elijas:

| Clave | Valor |
|---|---|
| `train/loss` | la loss (función de pérdida) media de entrenamiento de la época |
| `train/loss/<component>` | cada componente de la loss que reporta la familia |
| `lr/<group>` | el learning rate de cada grupo de parámetros del optimizador |
| `val/<metric>` | cada métrica de validación, sin su prefijo `metrics/` |
| `time/epoch_seconds` | el tiempo de reloj de la época |

El step es la época, empezando en 1. La configuración de entrenamiento
totalmente resuelta se registra como parámetros al inicio del entrenamiento, y
el nombre de la ejecución es por defecto `<family><size>-<task>`, por ejemplo
`yolo9s-detect`.

Al terminar el entrenamiento, los backends que soportan artefactos suben
`results.csv`, `train_config.yaml` y `summary.json` cuando existen, más
`weights/best.pt` con `log_checkpoints=True`. TensorBoard no sube nada, porque
no tiene concepto de artefacto. Ningún logger sube las imágenes de las gráficas
de validación.

## Comportamiento ante fallos

Si falta el paquete de un backend, el error salta en la construcción y nombra el
comando de instalación, porque pedir un logger y no obtener nada en silencio
esconde un bug.

Un fallo del backend durante la ejecución hace lo contrario. La primera
excepción de un handler desactiva ese logger para el resto de la ejecución, la
registra, cierra la ejecución del backend marcándola como fallida, y el
entrenamiento continúa. Que se caiga un servidor de tracking no te cuesta el
entrenamiento.

## Los backends

Cada uno necesita su propio extra.

| Nombre | Extra | Constructor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Importa las clases desde `libreyolo.training`.

Notas específicas de cada backend que conviene conocer antes de la primera
ejecución:

Los archivos de eventos de TensorBoard van por defecto a
`<save_dir>/tensorboard`. Míralos con `tensorboard --logdir runs/train`.

MLflow 3.x dejó obsoleto el almacén local de archivos `./mlruns` y lanza un
error salvo que `MLFLOW_ALLOW_FILE_STORE=true`. Para tracking local sin
servidor, pasa en su lugar una URI de base de datos, como en el snippet de
arriba, y léela con `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases recurre a la variable de entorno `WANDB_PROJECT` y luego a
`libreyolo`. Comet recurre a `COMET_PROJECT_NAME` y luego a `libreyolo`, y toma
las credenciales de su propia configuración; `online=False` da un experimento
offline. ClearML crea una tarea nueva, reporta la configuración bajo
`TrainConfig` y desactiva la captura automática del framework para que las
métricas no se reporten dos veces. Neptune usa el cliente actual
`neptune-scale` en lugar del paquete antiguo, y `mode="offline"` registra en
local.

DVCLive escribe en `<save_dir>/dvclive`. Construye su árbol de resumen a partir
de `/`, y no puede guardar un float en una ruta que además es un padre, así que
`train/loss/box` se escribe como `train/loss.box` mientras que `train/loss`
conserva su nombre. LibreYOLO también desactiva los valores por defecto
habituales de DVCLive de guardar un experimento de DVC y escribir un `dvc.yaml`
en la raíz, de modo que un logger opcional no crea estado de control de
versiones fuera del directorio de la ejecución; pasa `save_dvc_exp=True` o un
`dvcyaml=` explícito para recuperarlos.

Neptune queda deliberadamente fuera de `libreyolo[all]`: su cliente estable
requiere protobuf por debajo de 7 mientras que el extra de TFLite requiere
protobuf 7. Instala `libreyolo[neptune]` en un entorno sin el extra de TFLite.

## Escribir un callback

Los mismos cuatro eventos lo mueven todo.

<code-tabs name="callback" />

| Evento | Cuándo | Lleva |
|---|---|---|
| `TrainStartEvent` | tras el setup, antes de la época 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | después de cada época, entrenamiento y validación | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | cuando el entrenamiento termina | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | si el entrenamiento lanza una excepción | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Un callable simple recibe solo `TrainEpochEvent`. Un objeto puede implementar
cualquier subconjunto de `on_train_start`, `on_train_epoch_end`, `on_train_end`
y `on_train_exception`; los métodos que falten se omiten.

`TrainStartEvent.config` es la configuración totalmente resuelta, los kwargs del
usuario fusionados con los valores por defecto de la familia, como un mapping de
solo lectura. Los eventos son dataclasses congeladas y sus mappings son de solo
lectura, así que un callback no puede cambiar la ejecución escribiendo en uno.

Una excepción lanzada desde `on_train_start`, `on_train_epoch_end` o
`on_train_end` se propaga y termina la ejecución. Solo `on_train_exception` está
protegido, para que no pueda enmascarar el fallo original.

En entrenamiento multi-GPU, los callbacks se disparan solo en el rank 0. Con el
spawn automático de DDP además tienen que ser picklables, lo que significa una
clase o una función a nivel de módulo en lugar de un closure o una lambda.
Consulta [Entrenamiento multi-GPU](/docs/train/multi-gpu).

## Lo que toda ejecución escribe de todos modos

Tres archivos aparecen en el directorio de la ejecución sin ninguna
configuración, en todas las familias:

| Archivo | Se escribe | Contenido |
|---|---|---|
| `status.json` | de forma atómica, cada época y al inicio, al final y en caso de fallo | `state` con valor `running`, `completed` o `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, las últimas `metrics`, `best_metric`, `best_epoch`, y un objeto `error` en caso de fallo |
| `metrics.jsonl` | se añade una línea por época | una fila JSON por época, con el mismo esquema que `results.csv` |
| `train.log` | en vivo | la salida por consola de la ejecución |

`status.json` es la lectura barata para un script o un agente que consulta
periódicamente una ejecución, y la escritura atómica hace que un lector nunca
vea un archivo a medio escribir.

`results.csv` y `summary.json` van aparte y dependen de la familia. Se escriben
para YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC y DINOv2, y no
para las demás familias. `results.csv` recibe una fila por época con los
componentes de la loss, las métricas de validación y los learning rates como
columnas, y su cabecera se ensancha cuando aparece una columna nueva. Al
reanudar, se recorta hasta las filas anteriores a la época reanudada en lugar de
duplicarlas.

Junto a esos, el trainer siempre escribe `train_config.yaml` en el setup y los
checkpoints bajo `weights/`.

## Seguir una ejecución en vivo

<code-tabs name="monitor" />

`libreyolo monitor` sirve un dashboard en el navegador sobre los archivos de
arriba usando solo la biblioteca estándar: gráficas de métricas, el final del
log y las imágenes de validación que haya, refrescándose mientras la ejecución
está activa. Es de solo lectura y nunca toca el proceso de entrenamiento, así
que se engancha a una ejecución en vivo, reabre una terminada o inspecciona una
que ha fallado.

## Relacionado

- [Validación y métricas](/docs/train/validation) para saber qué significan las
  claves `val/` y cómo añadir una loss de validación.
- [Rendimiento del entrenamiento](/docs/train/performance) para el profiler, que
  es una herramienta distinta con una pregunta distinta.
