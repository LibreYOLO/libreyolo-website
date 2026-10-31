---
title: Grafos CUDA
seo_title: Matriz de soporte de grafos CUDA en LibreYOLO
description: >-
  Qué familias capturan su forward en predicción y su forward y su backward en
  entrenamiento, qué se garantiza sobre los números, dónde se parte una captura
  y por qué una familia no soportada lanza un error.
lead: >-
  Un grafo CUDA registra una ejecución de una secuencia fija de kernels y la
  reproduce como un único lanzamiento. LibreYOLO captura la inferencia en 39
  familias verificadas y el entrenamiento en 24, siempre por familia, siempre
  tras una comprobación de paridad bit a bit y nunca como fallback silencioso.
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - grafos cuda pytorch
  - acelerar inferencia yolo cuda graph
  - entrenar yolo con cuda graph
  - capture_error_mode thread_local
last_verified: 1.5.0
verification: >-
  Lista de familias de inferencia derivada de la matriz CAPTURABLE de
  tests/e2e/test_cuda_graph_families.py en la v1.5.0. Lista de familias de
  entrenamiento, clases de paridad y tiempos tomados de
  docs/training_cuda_graphs.md. La API y el NotImplementedError, de
  BaseModel._require_cuda_graph_support, cuda_graph_scope y capture_graph en
  libreyolo/models/base/model.py, con la variable de clase SUPPORTS_CUDA_GRAPH.
  Las divisiones por costura, leídas de los overrides de _get_graph_runner en
  las familias depth_anything3, birefnet, ppocr, sam y sensenova y de
  libreyolo/models/base/detr_cuda_graph.py. capture_error_mode, de
  libreyolo/models/base/cuda_graph.py y libreyolo/training/cuda_graph.py. El
  fallback de entrenamiento, de libreyolo/training/trainer.py y el flag
  --cuda-graph, de libreyolo/cli/commands/train.py.
meta:
  - label: Familias de inferencia
    value: '39'
  - label: Familias de entrenamiento
    value: '24'
  - label: Flag de inferencia
    value: predict(cuda_graph=True)
    mono: true
  - label: Flag de entrenamiento
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predicción
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # True captura en el primer uso de cada forma de entrada.

        # "auto" espera a que una forma se repita antes de pagar el coste de la
        captura.

        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Entrenamiento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Entrenamiento desde la CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Qué se captura

Un grafo registra una secuencia fija de kernels y las direcciones de memoria que
leen y escriben. No registra valores, formas ni flujo de control. La reproducción
es un único lanzamiento en lugar de cientos, y por eso la ganancia es mayor en
redes pequeñas con tamaños de batch pequeños, donde un paso está dominado por el
coste de lanzamiento y no por la aritmética.

Los dos puntos de entrada capturan cantidades de trabajo distintas.

| | Dentro del grafo | Eager |
|---|---|---|
| Inferencia | El forward de la red, `model._forward(x)` | El preprocesado, la NMS, todo el postprocesado |
| Entrenamiento | El forward y el backward de la red | La loss (la función de pérdida), el paso del optimizador, el recorte de gradientes, la EMA, la planificación del learning rate |

Ni la NMS ni la loss de detección son candidatas. Ambas seleccionan con máscaras
booleanas, ejecutan matching húngaro o un assigner y ramifican según el
resultado, que es exactamente lo que un grafo no puede registrar. Dejarlas fuera
es lo que hace que la captura sea segura, en lugar de una limitación que haya que
sortear.

<code-tabs name="usage" />

`cuda_graph` acepta tres valores en predicción. `False` es el valor por defecto.
`True` captura la primera vez que se ve cada forma de entrada. `"auto"` espera a
que una forma se repita, de modo que el trabajo puntual o de formas variables
nunca paga una captura que no va a reutilizar.
`capture_graph(imgsz=None, batch=1, dtype=None)` saca ese coste de la primera
petición, `graph_info()` informa de los grafos capturados y del número de
reproducciones, y `release_graphs()` los libera.

En entrenamiento el flag es un booleano a secas, `--cuda-graph` en la CLI.
Consulta [rendimiento en predicción](/docs/predict/performance) y
[rendimiento en entrenamiento](/docs/train/performance) para ver los controles
que lo rodean.

## Soporte en inferencia

El soporte es por familia, se declara mediante la variable de clase
`SUPPORTS_CUDA_GRAPH`, y una familia solo se marca después de capturar y
reproducir de forma idéntica bit a bit frente a dos entradas de prueba tomadas de
distribuciones diferentes. Esa matriz de paridad compartida cubre 39 familias en
nueve tareas.

| Tarea | Familias |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Varias familias aparecen bajo más de una tarea, así que la matriz tiene más filas
que familias distintas. Otras tres familias capturan a través de rutas de código
específicas, con sus propios tests dedicados en lugar de la matriz compartida, y
no forman parte de las 39: PP-OCR, SAM y SenseNova.

La verificación es bit a bit, no aproximada. Una versión anterior del protocolo
juzgaba la paridad por magnitud relativa y degradó por error a tres familias
sanas, YOLOX, EfficientNetV2 y YOLOv7, cuya diferencia entre eager y grafo mide
alrededor de 1e-7 aun siendo idéntica bit a bit en la prueba que importa.

## Soporte en entrenamiento

La captura en entrenamiento pasó de dos familias a 24 en esta versión, repartidas
en cinco tareas.

| Tarea | Familias |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Todo lo demás entrena en eager: las otras tareas sobre esas mismas familias, las
familias que no están en la lista, las ejecuciones distribuidas y las de
destilación. La captura también se omite mientras una forma sigue siendo nueva,
porque la ruta de entrenamiento espera a que una forma de entrada se repita tres
veces antes de capturar, lo que significa que con `multi_scale=True` puede no
capturarse nunca.

## Dos respuestas distintas para una familia no soportada

La ruta de inferencia lanza un error. `predict(cuda_graph=True)` sobre una
familia que no se ha adherido lanza `NotImplementedError` nombrando la familia,
en lugar de ejecutar en eager y dejarte creer que has conseguido una aceleración
que no has conseguido. La razón es que una captura mala no falla de forma
ruidosa: reproducir un forward que hace algo no capturable devuelve números
incorrectos en silencio, así que el soporte tiene que ser una afirmación
explícita por familia y no un intento con fallback.

La ruta de entrenamiento lo escribe en el log. `train(cuda_graph=True)` siempre
se puede pasar sin riesgo, y una familia, tarea o configuración que no se puede
capturar escribe una línea y entrena en eager, sin cambios. Una captura que falla
a mitad de una ejecución también deja el resto de la ejecución en eager en lugar
de abortarla. La asimetría es deliberada: una predicción es una llamada que
puedes arreglar en el punto de llamada, mientras que una ejecución de
entrenamiento no debería morir en la hora seis por una optimización opcional.

## División por la costura

Algunas familias no se pueden capturar enteras porque una de sus etapas hace
realmente algo que un grafo no puede registrar. En lugar de descartar la familia,
la captura se parte por una costura verificada: la parte capturable se reproduce,
el resto se ejecuta en eager y la salida combinada es la misma que ejecutándolo
todo en eager.

| Familia | Capturado | En eager, y por qué |
|---|---|---|
| Depth Anything 3 | La red | El paso del cielo, que es trabajo visible desde el host posterior al forward |
| BiRefNet | El encoder, `forward_enc` | El decoder, cuyo `deform_conv2d` se reproduce con un resultado distinto bajo captura |
| PP-OCR | La etapa de detección, `forward_det` | El reconocimiento, porque el ancho de los recortes varía en cada línea |
| SAM | El encoder de imagen | La ruta de prompts, que se ejecuta muchas veces por cada codificación |
| SenseNova | La torre de visión | La generación autorregresiva, con una caché KV que crece en cada paso |
| Detectores encoder-decoder | El backbone y el encoder | El decoder y el criterio húngaro |

La división de BiRefNet merece una segunda lectura: el mal comportamiento de
`deform_conv2d` bajo captura se reproduce en una llamada suelta, fuera de
cualquier modelo. Sustituirlo por un equivalente en PyTorch puro se descartó
porque eso habría desplazado también las predicciones en eager, y los números de
eager son el contrato.

El caso encoder-decoder cubre D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 y EC. Su decoder construye queries de contrastive denoising a partir
del ground truth, y el número de esas queries sale del mayor recuento de ground
truth del batch, así que el número de tokens del decoder cambia de un batch a
otro. Eso es justo lo único que un grafo no tolera. El backbone más el encoder
suponen aproximadamente entre un quinto y un cuarto de un paso en estas familias,
y por eso están al final de la tabla de aceleraciones.

PP-OCR captura un grafo por cada forma de entrada de detección, con el límite del
tope de caché del runner, y devuelve el resultado en eager cuando no hay ningún
ámbito de captura activo.

## Comportamiento numérico

La mayoría de las familias son idénticas bit a bit y, donde no lo son, la razón
se nombra en lugar de despacharse con un gesto. En el paso cero del entrenamiento
la loss es idéntica bit a bit en las 24 familias y ningún buffer de BatchNorm
difiere; es la comparación de gradientes la que separa las categorías.

| Clase | Familias | Significado |
|---|---|---|
| Exacta | La mayoría de las 24 | Todos los gradientes idénticos bit a bit |
| 1 ULP | fomo, lingbotvision | El último bit de float32, alrededor de 1e-7 relativo, por un orden de suma distinto |
| Ruido de eager | El linaje DETR | Lo capturado difiere de eager no más de lo que dos ejecuciones eager difieren entre sí |
| Redondeo de coma flotante | rtmdet | 137 de 139 gradientes idénticos bit a bit, dos difieren en unos 3e-4 |
| Flujo de RNG propio | segformer | El stochastic depth queda dentro de la región capturada |

La clase de ruido de eager es la importante de leer bien. En esas familias, dos
ejecuciones eager con la misma semilla ya discrepan entre sí, así que ser
idéntico bit a bit no es un listón que la ejecución con grafo no haya superado;
es un listón que no supera nadie. Eso vale de forma más general con `amp=False`,
donde un no determinismo relativo medido de 3.2e-7 en el gradiente de un peso
fp32 se acumula: dos ejecuciones eager de YOLOv9-t con la misma semilla divergen
un 36 por ciento en 20 pasos, y desactivar TF32 no lo arregla.

## Pin memory

La captura se ejecuta con `capture_error_mode="thread_local"`. Con el modo
`"global"` que PyTorch trae por defecto, el hilo de pin memory del DataLoader que
va preparando el siguiente batch llama a `cudaHostAlloc`, lo que invalida la
captura en curso y a la vez queda envenenado por ella, de modo que la ejecución
muere al recoger el siguiente batch con un error lanzado desde dentro del hilo de
pin memory. Esa combinación se observó dos veces en una campaña de entrenamiento
real antes de diagnosticarla.

El modo thread-local restringe solo al hilo que captura. El hilo de pin nunca
toca el stream de captura, así que nada de lo que hace tendría por qué estar en
el grafo. El entrenamiento va más allá y sustituye temporalmente
`torch.cuda.CUDAGraph` por una subclase que fuerza ese modo, porque
`make_graphed_callables` no expone ningún argumento para ello, bajo un lock para
que dos capturas concurrentes no puedan dejar la sustitución instalada.

## Cuánto se gana

Medido en una RTX 5070 Ti con AMP, un proceso por rama, reproduciendo un batch
real para dejar al dataloader fuera de la ecuación, el más rápido de 24 pasos
tras el calentamiento. Detección a 640 px, clasificación a 224 px.

| Familia | Batch | Aceleración |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Todo lo demás | varía | de 1.04x a 1.26x |

Una ejecución completa gana menos, porque un grafo no puede acelerar el
dataloader ni la validación. Un fine-tuning de YOLOv9-t de 20 épocas sobre 406
imágenes pasó de 428,4 s a 367,7 s, una ganancia extremo a extremo de 1.16x, con
un mAP50-95 idéntico de 0.6394 en ambas ramas y losses por época idénticas.

El techo lo marca qué parte de un paso es red. En el mismo hardware, a 640 px y
batch 8, eso es el 84 por ciento en YOLOv9-t pero solo el 26 por ciento en
RTMDet-t, que pasa la mayor parte de un paso en su asignador de etiquetas. El
coste de lanzamiento es mayor en Windows, así que en Linux las ganancias quedan
aproximadamente entre un tercio y la mitad de esta tabla, y una ejecución
limitada por el dataloader no ve ningún cambio en tiempo de reloj. La memoria
pico se mueve entre un 5 por ciento menos y un 19 por ciento más.

## Advertencias

Un grafo registra direcciones, no valores, así que cualquier cosa que reubique
los parámetros lo tira. Cambiar de dispositivo con `predict(device=...)`,
cuantizar y descuantizar invalidan los grafos capturados.

El tamaño de batch importa más que la familia: RT-DETR-r18 gana 1.19x con batch 2
y 1.04x con batch 8, porque un batch grande está limitado por cómputo y tiene
menos coste de lanzamiento que eliminar.

La suite de paridad de inferencia se ejecutó sin el paquete opcional `kernels`
instalado, así que la seguridad de la captura con los kernels compilados del Hub
activos no queda cubierta por ella. Pon `LIBREYOLO_HUB_KERNELS=0` para dejarlos
fuera de la ecuación mientras aíslas un problema de captura. Consulta
[kernels](/docs/reference/kernels).
