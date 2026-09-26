---
title: Entrenamiento multi-GPU
seo_title: Entrenamiento multi-GPU en LibreYOLO
description: >-
  Entrena en varias GPU con device="0,1". Cómo la biblioteca lanza los workers
  de DDP, por qué batch es el batch global, cuándo activar sync_bn y la vía de
  torchrun.
lead: >-
  El entrenamiento multi-GPU en LibreYOLO es DistributedDataParallel de PyTorch:
  un proceso por GPU, cada uno con una réplica completa del modelo y una porción
  de cada batch, con los gradientes promediados entre los ranks en cada paso.
keywords:
  - entrenamiento ddp pytorch
  - entrenamiento multi gpu
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - tamaño de batch global
  - backend nccl gloo
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # El guard de __main__ es obligatorio: cada worker lanzado reimporta
        este

        # módulo, y sin el guard relanzaría el entrenamiento de forma recursiva.

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch global: 16 imágenes por GPU con dos GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Lanzamiento
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Se sondea una vez en la GPU 0 y se escala a un múltiplo del world size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Entrenar en dos GPU

Pasa una lista de dispositivos. No cambia nada más.

<code-tabs name="train" />

Cuando recibe más de un dispositivo y no hay entorno de torchrun, el `train()` del
modelo guarda los pesos en un archivo temporal, resuelve el autobatch si se ha
pedido y lanza un proceso worker por GPU con `torch.multiprocessing.spawn`. Cada
worker reimporta la clase del modelo, lo reconstruye a partir de los pesos
guardados y ejecuta la vía normal de un solo dispositivo, porque desde dentro de
un worker lanzado las variables de entorno de torchrun están definidas. Cuando la
ejecución termina, el mejor checkpoint del rank 0 se vuelve a cargar en la
instancia del modelo que hizo la llamada.

`device` acepta `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` y
`"auto"`. Solo una lista de más de un índice CUDA dispara el spawn.

## El guard de `__main__` es obligatorio

Los workers lanzados reimportan el módulo del que salieron. Sin un guard
`if __name__ == "__main__":`, esa importación vuelve a ejecutar la llamada de
entrenamiento y cada worker lanza sus propios workers. La biblioteca detecta el
caso y lanza un error en lugar de dejar que se repita en cascada:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Todo lo que cruza hacia un worker se serializa con pickle, así que `callbacks=`
tiene que ser picklable. Una clase a nivel de módulo funciona; un closure o una
lambda no, y el error lo dice y apunta a los loggers integrados como
alternativa.

## batch es el batch global

`batch` es el número de imágenes por paso del optimizador entre todas las GPU. El
dataloader de cada rank se construye con `batch // world_size` y un
`DistributedSampler`, así que `batch=32` con dos GPU significa 16 imágenes por
GPU, no 32.

Un batch que no es divisible entre el world size lanza un error en lugar de
entrenar en silencio con un tamaño distinto:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Los gradientes los promedia el propio DDP, así que la función de pérdida se pasa
sin escalar. Multiplicarla además por el world size inflaría el learning rate
efectivo en un factor cercano al número de GPU.

## Autobatch bajo DDP

`batch=-1` funciona y devuelve un batch global divisible por el world size.

<code-tabs name="autobatch" />

En la vía del spawn el sondeo se ejecuta en el proceso padre sobre el primer
dispositivo, antes de que exista ningún worker, así que cada worker recibe un
entero concreto y no hace falta ninguna coordinación entre procesos. Bajo
torchrun, el rank 0 sondea y difunde el resultado como un único tensor long.

El sondeo mide la capacidad de una GPU y la multiplica por el world size. Cuando
`nbs` está definido, el batch global se limita a `nbs` y se redondea hacia abajo
a un múltiplo del world size, de modo que añadir GPU reduce el número de pasos de
acumulación en vez de recortar el batch por GPU. La mecánica del sondeo en sí está
en [Hiperparámetros](/docs/train/hyperparameters).

## SyncBatchNorm

Bajo DDP, las capas BatchNorm de cada rank solo ven su propia porción. Con
`batch // world_size` esa porción puede ser lo bastante pequeña como para que las
estadísticas acumuladas degraden el modelo convergido frente a una ejecución en
una sola GPU.

`sync_bn=True` convierte cada BatchNorm en SyncBatchNorm para que las
estadísticas se calculen sobre el batch global. La conversión solo ocurre cuando
el modo distribuido está activo, así que una ejecución en una sola GPU no se ve
afectada por el flag en ningún caso.

Ya viene activado por defecto en las familias convolucionales con mucho
BatchNorm: YOLOX, YOLOv7, YOLOv9 y sus variantes, YOLO-NAS, PicoDet, RTMDet y
FOMO. Todas las demás familias lo dejan desactivado por defecto. Cuando un modelo
contiene BatchNorm, `sync_bn` está desactivado y el batch por rank es inferior a
16, el trainer avisa.

<code-tabs name="syncbn" />

No hay flag de CLI para `sync_bn`. Es un argumento de Python.

## Lanzar con torchrun

torchrun también funciona, y es la opción correcta cuando el planificador de un
clúster ya se encarga de lanzar los procesos. Escribe el script para un solo
dispositivo y deja que torchrun defina el entorno de rank.

<code-tabs name="torchrun" />

No combines ambas cosas. Con el entorno de torchrun presente, `device="0,1"` no
lanza procesos; el trainer toma `cuda:LOCAL_RANK` y torchrun controla el número
de procesos.

## Comportamiento de los ranks

El rank 0 es dueño de todos los efectos secundarios. Resuelve el directorio de la
ejecución y difunde el nombre resuelto para que todos los ranks coincidan,
escribe los checkpoints y los artefactos, y dispara los callbacks del usuario y
los loggers. Los demás ranks entrenan y aportan gradientes.

Cada rank inicializa de forma distinta la semilla de su dataloader y del RNG de
aumento de datos, derivándola del `seed` configurado, para que los ranks no
obtengan los mismos aumentos.

## Plataforma y backend

El backend se elige automáticamente: NCCL cuando están disponibles tanto CUDA
como NCCL, y Gloo en caso contrario. NCCL no se compila en Windows, así que las
ejecuciones en Windows usan Gloo sin ninguna configuración. El grupo de procesos
se inicializa con un timeout de tres horas.

## Lo que no funciona bajo DDP

- La captura de grafos CUDA. `cuda_graph=True` registra una línea y entrena en
  modo eager. Consulta [Rendimiento del entrenamiento](/docs/train/performance).
- El profiler de entrenamiento. `profile=True` se ignora con un aviso.

No todas las familias admiten el spawn automático. Veinticuatro sí, y cubren las
familias de detección, clasificación, semántica y restauración que se entrenan.
Una familia que no lo admite, si recibe un dispositivo multi-GPU, lanza un error
que nombra la API del modelo y el comando de torchrun en lugar de entrenar en
silencio en una sola GPU.

## Relacionado

- [Hiperparámetros](/docs/train/hyperparameters) para `batch`, `nbs` y reanudar.
- [Loggers de experimentos](/docs/train/loggers) para la restricción de
  picklabilidad en los callbacks.
- [GPU en la nube](/docs/train/cloud-gpus) para alquilar una máquina multi-GPU.
