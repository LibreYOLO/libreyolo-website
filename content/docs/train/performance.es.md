---
title: Rendimiento del entrenamiento
seo_title: "Entrenar más rápido: grafos CUDA, AMP, profiler"
description: "Haz que un entrenamiento vaya más rápido: captura el paso en grafos CUDA, elige un dtype para AMP y usa el profiler integrado para ver adónde se va realmente el tiempo."
lead: "Tres palancas cambian lo rápido que va un paso de entrenamiento: la precisión mixta, la captura del forward y el backward de la red en grafos CUDA, y lo que el profiler diga que está frenando el paso de verdad."
keywords:
  - grafos cuda entrenamiento
  - acelerar entrenamiento yolo
  - entrenamiento con precisión mixta
  - entrenar con bfloat16
  - profiler pytorch
  - entrenamiento limitado por el dataloader
  - overhead de lanzamiento de kernels
  - utilización de la gpu entrenamiento
last_verified: "1.5.0"
snippets:
  profile:
    - label: Perfilar y seguir entrenando
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Perfila una ventana corta de pasos reales, imprime un veredicto y
        # después sigue la ejecución con los hooks retirados.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: Solo medir y parar
      language: bash
      code: |
        # Fija no_aug_epochs=0 y ejecuta las épocas justas para llenar la ventana.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Profundizar en el resultado
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
---

## Medir antes de tocar nada

Las tres palancas de abajo arreglan problemas distintos, y aplicar la
equivocada no cambia nada. El profiler dice cuál de esos problemas tienes.

<code-tabs name="profile" />

`profile=True` mide una ventana de pasos reales de entrenamiento, cinco
descartados y luego veinte medidos por defecto, imprime un informe, escribe sus
artefactos y después sigue entrenando con los hooks retirados. No cuesta nada
cuando está apagado, y se ignora en entrenamiento distribuido.

El informe termina en uno de cuatro veredictos:

| Veredicto | Significado | Palancas |
|---|---|---|
| `dataloader` | la GPU espera a los datos de entrada | más `workers`, `cache="ram"` o `"disk"`, aumento de datos más ligero, batch más grande |
| `host / launch` | la GPU se alimenta demasiado despacio, muchos kernels diminutos | batch más grande, grafos CUDA, menos sincronizaciones con el host por paso |
| `compute` | la GPU está saturada | AMP o bfloat16, o aceptarlo |
| `memory-pressure` | thrashing del allocator, la VRAM al límite | baja el batch; aquí las cifras de utilización no son fiables |

El número de utilización es el tiempo ocupado por kernels dividido entre el
tiempo de paso sin sincronizar. La ventana se parte a propósito: la primera
mitad se ejecuta sin sincronización extra, para que el veredicto refleje el
solapamiento real, y solo la segunda mitad rodea cada fase con una
sincronización para atribuir el tiempo de GPU. Sincronizar en cada fase da
holgura a los workers del dataloader y esconde la inanición, así que las cifras
de composición no se usan nunca para elegir el veredicto.

En el directorio de la ejecución aparecen cuatro archivos: `timeline.html`, que
se abre por sí solo en un navegador, `profile_trace.json` para Perfetto o
Nsight, `profile_summary.json` y `profile.json`, el autocontenido, el que se
copia de un lado a otro y se vuelve a pasar a los subcomandos
`libreyolo profile`.

Hay dos cosas de `profile run` que conviene saber. Fija `no_aug_epochs=0`,
porque el profiler mide la época 0 y una ejecución corta con el
`no_aug_epochs` por defecto perfilaría el dataloader más ligero, el que va sin
aumento de datos, en lugar del que el entrenamiento usa de verdad. Y
`--repeat N` informa de la media y la desviación estándar, lo que importa
porque un paso limitado por los lanzamientos tiene ruido suficiente como para
que una sola ejecución induzca a error; escribe un directorio por prueba,
`prof_1`, `prof_2` y así sucesivamente, más un `profile_repeat.json` agregado.

## Precisión mixta

`amp=True` es el valor por defecto en la mayoría de las familias y ejecuta el
forward bajo el autocast de CUDA. `amp_dtype` elige entre `float16` y
`bfloat16`.

<code-tabs name="amp" />

Float16 necesita escalado dinámico de la función de pérdida (loss) y recibe un
escalador de gradientes activo; el rango de exponente más ancho de bfloat16 no
lo necesita, así que su escalador queda desactivado. Cuatro familias vienen con
`amp=False`, D-FINE, DEIM, YOLO-NAS y FOMO, y el ajuste de DEIM llega a
RT-DETRv4 por herencia. D-FINE indica el motivo: su decoder limita las
activaciones a 65504, el mayor valor finito de float16.

La semántica de los argumentos, incluido qué hace una petición de bfloat16 en
hardware sin soporte de bfloat16, está en
[Hiperparámetros](/docs/train/hyperparameters).

## Grafos CUDA

`cuda_graph=True` captura el forward y el backward de entrenamiento de la red en
un grafo CUDA, y elimina el overhead de lanzamiento de kernels por paso.

<code-tabs name="graph" />

Pasar el flag siempre es seguro. Una familia, tarea o configuración que no se
puede capturar escribe una línea en el log y entrena en eager, sin cambios.

Solo se captura la red. La loss se queda en eager por diseño, porque las losses
de detección seleccionan con máscaras booleanas, ejecutan matching húngaro y
ramifican según los resultados de la asignación, y un grafo no puede registrar
nada de eso. El paso del optimizador, el recorte de gradientes, la actualización
del EMA y el schedule del learning rate también se quedan en eager.

Eso acota la ganancia a la parte del paso que es red, y esa proporción varía
mucho. Medido en una RTX 5070 Ti a 640 px, batch 8: el 84 por ciento de un paso
de YOLOv9-t es red, el 44 por ciento de uno de YOLOv7-b, el 31 por ciento de uno
de YOLOX-t y el 26 por ciento de uno de RTMDet-t. Los dos últimos pasan la mayor
parte del paso dentro de sus asignadores de etiquetas, así que capturar la red
es lo que menos les ayuda.

### Cuánto se gana

Condiciones de todas las cifras de abajo: RTX 5070 Ti, Windows, AMP, un proceso
por rama a partir de un estado guardado compartido, reproduciendo un batch real
para dejar al dataloader fuera del bucle, el más rápido de 24 pasos tras el
calentamiento. Detección a 640 px, clasificación a 224 px. El tamaño de batch va
por fila.

| Familia | Tamaño | Batch | Eager | Con grafo | Aceleración |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7,0 ms | 1,9 ms | 3,63x |
| MobileNetV4 | s | 16 | 14,5 ms | 5,3 ms | 2,74x |
| EfficientNetV2 | b0 | 16 | 29,0 ms | 11,9 ms | 2,44x |
| YOLOv9 | t | 8 | 93,6 ms | 47,0 ms | 1,99x |
| NAFNet | s | 8 | 132,5 ms | 105,5 ms | 1,26x |
| PicoDet | s | 8 | 145,0 ms | 118,7 ms | 1,22x |
| D-FINE | n | 4 | 185,3 ms | 159,2 ms | 1,16x |
| RF-DETR | n | 4 | 276,3 ms | 239,8 ms | 1,15x |
| YOLOX | t | 8 | 102,2 ms | 90,5 ms | 1,13x |
| RTMDet | t | 8 | 149,7 ms | 136,2 ms | 1,10x |
| YOLOv7 | b | 4 | 102,5 ms | 98,0 ms | 1,05x |

Esas cifras aíslan el paso de GPU. Un fine-tuning completo paga además el
dataloader y la validación. YOLOv9-t sobre un dataset de detección de 406
imágenes, 20 épocas, batch 8, 640 px, 4 workers de dataloader, en la misma
máquina: 428,4 s de tiempo de reloj en eager frente a 367,7 s con grafo, una
ganancia de 1,16x, con un mAP50-95 de 0,6394 en las dos ramas.

Tres cosas mueven estos números. Los batches pequeños están limitados por los
lanzamientos y los grandes por el cómputo, así que RT-DETR-r18 gana 1,19x con
batch 2 y 1,04x con batch 8. El overhead de lanzamiento es máximo en Windows, y
en Linux las ganancias son más o menos de un tercio a la mitad de las de la
tabla. Y una ejecución limitada por el dataloader no ve ningún cambio en el
tiempo de reloj, y por eso el profiler va primero.

La captura se activa igual con `amp=False`, pero los kernels fp32 tardan más,
así que el paso está menos limitado por los lanzamientos y la mayoría de
familias ganan menos. En el mismo hardware, MobileNetV4-s con batch 16 pasa de
2,74x bajo AMP a 3,61x en fp32, mientras que YOLOv9-t con batch 8 pasa de 1,99x
a 1,69x y RT-DETR-r18 con batch 4 de 1,12x a 0,99x.

### Dónde se aplica la captura

| Tarea | Familias |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Todo lo demás cae a eager con una línea de log: las otras tareas sobre esas
familias, las familias que no están en la lista, las ejecuciones distribuidas y
las de destilación. Un fallo de captura en tiempo de ejecución también deja el
resto de la ejecución en eager en lugar de fallar.

En los detectores encoder-decoder, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 y v4, y
EC, solo se capturan el backbone y el encoder. Su decoder lee el ground truth
para construir las queries de contrastive denoising, y el número de esas queries
sigue al mayor recuento de ground truth del batch, así que su número de tokens
cambia de un batch a otro.

### Formas

Un grafo vale exactamente para la forma de entrada con la que se capturó. El
entrenador cuenta las formas de los batches y captura en cuanto una forma se ha
repetido tres veces. Los batches con cualquier otra forma se ejecutan en eager:
los batches multiescala y el último batch parcial de una época.

Esta es la trampa de las familias DETR, que redimensionan cada batch por
defecto. Con `multi_scale=True`, una ejecución corta puede no ver ninguna forma
las veces suficientes como para llegar a capturar. Pasa `multi_scale=False`
cuando lo que buscas es la aceleración.

YOLOX cambia lo que calcula la región capturada a mitad de una ejecución, y
activa su rama de regresión L1 cuando el mosaic se cierra en `no_aug_epochs`.
El entrenador invalida ahí la captura y vuelve a capturar en cuanto la nueva
forma se asienta.

### Comportamiento numérico y memoria

La mayoría de familias reproducen su trayectoria de loss en eager bit a bit bajo
AMP. FOMO y LingBot-Vision difieren en el último bit de float32 por un orden de
suma distinto. Los detectores de atención deformable, D-FINE, DEIM, DEIMv2,
RT-DETR, RF-DETR y EC, tampoco reproducen sus propias ejecuciones en eager,
porque ese backward acumula con atómicas y las convoluciones TF32 eligen un
orden de reducción en cada lanzamiento; la ejecución con grafo se queda dentro
de esa dispersión. RTMDet difiere en torno a 3e-4 relativo en dos de 139
gradientes, porque comparte convoluciones de la cabeza entre niveles de la
pirámide y los dos caminos de backward suman tres contribuciones en distinto
orden. SegFormer tiene stochastic depth dentro de la región capturada, así que
un grafo reproducido consume su propio flujo aleatorio y es estadísticamente
equivalente a eager en lugar de idéntico; el gestor lo registra una vez en el
momento de la captura.

Con `amp=False` no hay nada bit a bit idéntico en este hardware, con captura o
sin ella. Dos ejecuciones en eager de YOLOv9-t con la misma semilla divergen un
36 por ciento relativo en 20 pasos, y YOLOX-t un 2,6 por ciento, porque cuDNN
elige un algoritmo no determinista de gradiente de pesos para algunas formas de
convolución en fp32.

Un grafo capturado fija buffers estáticos de entrada, de salida y de workspace,
así que el pico de VRAM sube más o menos en un juego extra de activaciones. En
las familias de arriba, el pico de asignación se movió entre el -5 y el +19 por
ciento. El coste relativo es mayor en los modelos de clasificación pequeños,
cuyas activaciones ya son pequeñas de partida: ResNet-18 a 224 px, batch 16,
pasó de 0,48 GB en eager a 0,57 GB con grafo. Si eso empuja una ejecución por
encima del límite, baja el batch o deja el flag apagado.

## Relacionado

- [Hiperparámetros](/docs/train/hyperparameters) para `batch`, `nbs`, `cache` y
  `workers`.
- [Entrenamiento multi-GPU](/docs/train/multi-gpu), donde ni los grafos CUDA ni
  el profiler están disponibles.
- [Grafos CUDA](/docs/reference/cuda-graphs) para la matriz de soporte combinada
  de inferencia y entrenamiento, las divisiones por la costura y el contrato
  numérico.
