---
title: Hiperparámetros
seo_title: Hiperparámetros de entrenamiento en LibreYOLO
description: >-
  Los argumentos de train() que importan: epochs, batch, lr0, optimizer, EMA,
  autobatch, acumulación de gradientes y resume, además de por qué los valores
  por defecto cambian según la familia.
lead: >-
  Cada argumento de entrenamiento es un campo de una dataclass TrainConfig. La
  clase base define el campo y su valor por defecto; cada familia de modelos
  hereda de ella y sobrescribe los valores por defecto que cambia su receta
  publicada.
keywords:
  - argumentos de entrenamiento yolo
  - learning rate
  - tamaño de batch
  - autobatch
  - media movil exponencial ema
  - acumulacion de gradientes
  - reanudar entrenamiento yolo
  - early stopping patience
  - amp bfloat16
  - train config yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Leer los valores por defecto resueltos de una familia
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Imprime los valores por defecto de train, val y predict, incluidas las
        sobrescrituras de cada familia.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sondea la memoria de la GPU y resuelve a una potencia de dos
        concreta.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-batches de 16 por paso del optimizador, batch efectivo 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Carga el checkpoint de la ejecución interrumpida y pide reanudarla.
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Las claves del yaml son nombres de campos de TrainConfig. Los kwargs
        explícitos ganan.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Cómo pasar argumentos

`train()` acepta argumentos con nombre y la CLI acepta esos mismos nombres en
forma `key=value`.

<code-tabs name="train" />

Ambos caminos terminan en el mismo sitio. Los kwargs se pasan a
`TrainConfig.from_kwargs()`, que construye la dataclass de configuración de la familia.

## Una errata no lanza ningún error

`from_kwargs()` descarta cualquier clave que no sea un campo de la configuración y
emite un `UserWarning` que la nombra. El entrenamiento arranca entonces con el
valor por defecto en su sitio:

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

No falla nada, la ejecución termina y el learning rate nunca fue el que pidió
quien llamó. Lee los warnings en la primera época de una receta nueva. La CLI es
más estricta, porque valida los nombres de los flags antes de construir la
configuración, así que un flag mal escrito en la CLI se rechaza de plano.

## Los valores por defecto son por familia

`TrainConfig` define el campo y un valor por defecto base. Cada familia hereda de
ella y sobrescribe lo que cambia su receta publicada, así que no hay una
única respuesta correcta a "cuál es el learning rate por defecto".

Los valores por defecto base son `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` y `amp=True`. Tres ejemplos de cuánto se aleja de ahí una familia:

| Campo | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE y DEIM vienen con `amp=False` porque el decoder de D-FINE limita las
activaciones a 65504, el mayor valor finito de float16. YOLO-NAS y FOMO también
lo desactivan por defecto. El flag `--amp` de la CLI toma `True` por defecto para
todas las familias, así que cuenta como proporcionado por el usuario y sobrescribe
el valor por defecto de la familia; no lo toques salvo que quieras cambiarlo.

Para leer los valores por defecto reales de una familia en lugar de adivinarlos:

<code-tabs name="defaults" />

## Tamaño de batch

`batch` es el batch global. En el entrenamiento multi-GPU cada rank carga
`batch // world_size`, así que el número que pasas es el número de imágenes por
paso del optimizador, independientemente de cuántas GPU intervengan. Consulta
[Entrenamiento multi-GPU](/docs/train/multi-gpu).

`batch=-1` activa el autobatch. El trainer sondea el modelo en modo entrenamiento
con una pasada backward real en potencias de dos, ajusta una recta a la curva
de memoria y elige la mayor potencia de dos estrictamente por debajo del valor
extrapolado que quepa dentro del 60 por ciento de la VRAM total.

<code-tabs name="autobatch" />

Sondear en modo entrenamiento con una pasada backward es justo la clave: un
sondeo en modo inferencia pasa por alto las activaciones retenidas y los tensores
de gradientes, que en una CNN profunda son varias veces la huella de la
inferencia. RF-DETR baja la fracción objetivo al 45 por ciento, porque la pasada
backward sintética del sondeo sigue subestimando lo que cuestan su criterio y sus
capas de decoder auxiliares.

El autobatch es una funcionalidad de CUDA. En CPU o MPS registra una línea y
mantiene el batch por defecto.

## Acumulación de gradientes

`nbs` fija el tamaño de batch nominal, o efectivo. El trainer acumula
`round(nbs / batch)` micro-batches por paso del optimizador.

<code-tabs name="accumulate" />

Si se deja en `None`, el valor por defecto, la acumulación está desactivada y el
entrenamiento no cambia.

## Learning rate y planificación

`lr0` es el learning rate inicial y `optimizer` acepta `sgd`, `adam` y `adamw`.
`momentum` es el momentum de SGD o el beta1 de Adam, `weight_decay` es el término
L2, y `nesterov` se aplica a SGD.

La planificación la determinan `scheduler`, `warmup_epochs`, `warmup_lr_start` y
`min_lr_ratio`. `no_aug_epochs` fija cuántas épocas finales se ejecutan sin
aumento de datos fuerte, y varios schedulers lo usan también para dar forma a su
tramo final, así que no es solo un parámetro de aumento de datos. Lo que hace cada
familia con su vertiente de aumento de datos está en
[Aumento de datos](/docs/train/augmentations).

Algunas familias añaden sus propios parámetros de learning rate. `backbone_lr_mult`
escala el grupo del backbone frente a la cabeza, `clip_max_norm` fija el recorte
de gradientes, y SegFormer usa `head_lr_mult` para ejecutar su cabeza de
decodificación a diez veces el ritmo del backbone. Estos viven en la subclase de
configuración de la familia, no en la base.

## EMA

`ema=True` mantiene una media móvil exponencial de los pesos junto a los
entrenados. Está activada por defecto en todas partes excepto en FOMO.

`ema_decay` es el decaimiento objetivo. El decaimiento entra de forma progresiva
en lugar de empezar en su objetivo: el valor efectivo en la actualización `n` es
`ema_decay * (1 - exp(-n / tau))` con `tau` a 2000 por defecto, así que las
actualizaciones tempranas siguen al modelo más de cerca y las tardías lo suavizan.
Los valores por defecto de cada familia van desde `0.997` en YOLO-NAS pose,
pasando por `0.9998` en YOLOX, hasta `0.9999` en YOLOv9 y la línea DETR.

Los pesos de la EMA son los que se validan y los que llevan `best.pt` y `last.pt`.
Los pesos entrenados en bruto también se guardan, bajo la clave `train_model`, de
modo que un resume continúa desde la trayectoria entrenada y no desde la media.

## Precisión

`amp=True` ejecuta la pasada forward bajo el autocast de CUDA. `amp_dtype`
selecciona `float16` (el valor por defecto) o `bfloat16`; `fp16` y `bf16` son
grafías aceptadas.

Float16 necesita escalado dinámico de la loss (la función de pérdida) y recibe un
`GradScaler` vivo. El rango de exponente más amplio de bfloat16 no lo necesita,
así que su scaler se construye pero queda desactivado, lo que mantiene idéntico el
camino del optimizador. Pedir bfloat16 en un dispositivo CUDA sin soporte de
bfloat16 lanza un error al preparar el entrenamiento en lugar de degradarse en
silencio.

## Salida, checkpoints y parada

Las ejecuciones se escriben en `project/name`. `project` es `runs/train` por
defecto en todas partes, pero `name` es una de las sobrescrituras por familia: el
valor por defecto base es `exp`, mientras que YOLOv9 usa `yolo9_exp` y D-FINE usa
`dfine_exp`. Con `exist_ok=False`, el valor por defecto, un directorio existente
recibe un sufijo incremental en lugar de ser sobrescrito.

`save_period` escribe un `weights/epoch_<N>.pt` extra cada N épocas, además de
`weights/last.pt` después de cada época y `weights/best.pt` cada vez que mejora la
métrica que se sigue. `eval_interval` fija cada cuánto se ejecuta la validación, y
`patience` detiene la ejecución tras ese número de épocas sin mejora, con `0`
desactivando el early stopping.

`cache` acelera las épocas repetidas manteniendo las imágenes decodificadas en RAM
(`True` o `"ram"`) o como archivos `.npy` junto a las fuentes (`"disk"`). Las
lecturas cacheadas son idénticas byte a byte a las lecturas sin caché. Con workers
de dataloader, `"disk"` es la opción más segura de las dos.

## Reanudar

`resume=True` continúa una ejecución interrumpida. El checkpoint hay que cargarlo
primero, porque resume lo lee del modelo, no de un argumento aparte.

<code-tabs name="resume" />

Resume restaura los pesos entrenados, el estado del optimizador, los pesos de la
EMA y su contador de actualizaciones, el seguimiento de la mejor métrica, la
escala del `GradScaler` y los estados aleatorios de PyTorch, CUDA y NumPy. Empieza
en la época del checkpoint más uno y adelanta la planificación hasta esa posición.

Dos cosas que no hará. `resume=True` no se puede combinar con `pretrained`, y
hacerlo lanza un error. Y cuando la clave de mejor métrica del checkpoint difiere
de la de la ejecución actual, el seguimiento de la mejor métrica se reinicia a cero
con un warning en lugar de comparar valores que no significan lo mismo.

## Recetas en un archivo

`cfg=` carga un mapeo YAML de nombres de campos de `TrainConfig` y lo fusiona por
debajo de los argumentos con nombre explícitos, así que un kwarg siempre gana
sobre el archivo.

<code-tabs name="cfg" />

`size` y `num_classes` se eliminan del archivo, porque ya los define la instancia
del modelo. No hay flag `--cfg` en la CLI; la ruta del archivo es un argumento de
Python.

## Relacionado

- [Datasets](/docs/train/datasets) para lo que acepta `data=`.
- [Aumento de datos](/docs/train/augmentations) para los parámetros de aumento de
  datos y qué familias los respetan.
- [Congelación de capas](/docs/train/layer-freezing) y [LoRA](/docs/train/lora)
  para entrenar un subconjunto de los pesos.
- [Validación y métricas](/docs/train/validation) para lo que informa la ejecución.
