---
title: FAQ
seo_title: "Preguntas frecuentes sobre LibreYOLO"
description: "Respuestas breves a las preguntas que atraviesan todos los modelos de LibreYOLO: hardware, licencias, pesos, dispositivos, entrenamiento, cobertura de exportación y la CLI."
lead: "Respuestas a preguntas que no son específicas de una familia de modelos. Todo lo específico de una familia vive en la página de esa familia."
keywords: [libreyolo faq, libreyolo preguntas frecuentes, libreyolo necesita gpu, libreyolo licencia comercial, donde se guardan los pesos libreyolo, libreyolo cli, libreyolo sin internet]
last_verified: "1.5.0"
---

## ¿Con qué modelo debería empezar?

YOLOv9 para un detector CNN y RF-DETR para uno basado en transformers. Ambos
están en el nivel insignia, lo que significa que las funcionalidades se diseñan
y se validan en GPU contra ellos antes que contra cualquier otro. Consulta
[YOLOv9](/docs/models/yolov9) y [RF-DETR](/docs/models/rf-detr), o
[todos los modelos](/docs/models) para el resto.

## ¿Necesito una GPU?

No. Todos los modelos funcionan en CPU, y todo lo que aparece en el
[quickstart](/docs/quickstart) está escrito para ejecutarse ahí. Una GPU cambia
cuánto tardan el entrenamiento y la inferencia sobre vídeo, no si funcionan.

## ¿Cómo elige LibreYOLO el dispositivo?

El valor por defecto es `device="auto"`, que usa CUDA cuando PyTorch lo reporta
disponible, después Metal Performance Shaders cuando está disponible, y CPU en
caso contrario. Para fijarlo, pasa `device` al modelo o a `predict`, `train`,
`val` y `export`. Acepta `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, un entero sin
más como `0`, o una cadena de dígitos; los dos últimos se expanden a `cuda:<n>`.

`libreyolo checks` imprime la build de Torch, sus versiones de CUDA y cuDNN, y
todas las GPU que puede ver. Si ese comando no muestra CUDA, el wheel de PyTorch
es una build de CPU; [instalación](/docs/install) explica cómo reemplazarlo.

## ¿Dónde van los pesos descargados?

A `weights/`, relativo al directorio de trabajo. Una referencia de modelo sin
componente de directorio se resuelve ahí y se descarga en el primer uso; una
referencia que incluye un directorio se usa exactamente como está escrita y
nunca se descarga. Consulta [checkpoints y pesos](/docs/weights).

## ¿Puedo trabajar sin acceso a la red?

Sí. Descarga los checkpoints una vez en una máquina conectada, copia el
directorio `weights/`, y nada volverá a acceder a la red. Una ruta compartida de
solo lectura también funciona, ya que una referencia que contiene un directorio
se toma literalmente. Los datasets se resuelven bajo `~/datasets`, o bajo
`LIBREYOLO_DATASETS_DIR`.

## ¿Puedo usar LibreYOLO comercialmente?

El código tiene licencia MIT. Los pesos preentrenados son una cuestión aparte:
pueden heredar términos del proyecto o dataset del que proceden, y esos términos
no son uniformes ni siquiera dentro de una misma familia. La licencia del
repositorio concreto de Hugging Face es la autoritativa, y cada página de modelo
incluye una sección de licencias que la reproduce. Cuando los pesos están
restringidos, LibreYOLO imprime la restricción antes de que empiece la descarga.

## ¿Puedo cargar un checkpoint de otro proyecto?

Normalmente sí, pasando su ruta a `LibreYOLO()`. Los formatos upstream
reconocidos se convierten al cargar, conservando su número de clases y sus
nombres, y se escribe un checkpoint de LibreYOLO junto al original.
[Importar pesos existentes](/docs/migrate) explica qué se reconoce y qué
necesita un script de conversión.

## ¿Por qué train lanza NotImplementedError?

Porque esa familia solo incluye inferencia, y la excepción indica el motivo.
Predecir, validar y, donde está soportado, exportar funcionan; no hay bucle de
entrenamiento para esa arquitectura en LibreYOLO. El nivel de soporte en la
cabecera de la página de cada modelo te lo dice antes de intentarlo. Consulta
[conceptos básicos](/docs/concepts).

## ¿Qué devuelve val?

Un diccionario plano, no un objeto. Las claves de detección incluyen
`metrics/precision`, `metrics/recall`, `metrics/mAP50` y
`metrics/mAP50-95`. Las demás tareas devuelven las claves que tienen sentido
para ellas, como `metrics/accuracy_top1` para clasificación o `metrics/PQ`,
`metrics/SQ` y `metrics/RQ` para segmentación panóptica.

## ¿Cómo ejecuto sobre una carpeta, un vídeo o una webcam?

Pásalo como source. Una ruta de archivo es una imagen, un directorio es cada
imagen que contiene, la ruta de un vídeo es un vídeo, un entero es el índice de
una webcam, y una URL RTSP, RTMP, TCP, UDP o HLS es un stream en directo. Un
archivo `.streams` lista varias fuentes a la vez. Las fuentes en directo
requieren `stream=True`, que produce un `Results` por frame en lugar de
construir una lista; el mismo flag merece la pena para vídeos largos y
directorios grandes. Solo las URL de páginas de YouTube necesitan un extra,
`libreyolo[stream]`.

## ¿Cómo me quedo solo con algunas clases?

Pasa `classes` a `predict` con los índices de clase que quieras, por ejemplo
`classes=[0, 2]`. `conf` fija el umbral de confianza, por defecto `0.25`, y
`max_det` limita las detecciones por imagen, por defecto `300`.

## ¿La CLI usa flags o pares clave=valor?

Clave y valor unidos por un signo igual, en todos los comandos:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` acepta una ruta o un nombre corto de la forma `family-size`,
opcionalmente con un sufijo de tarea, y `libreyolo models` lista todos los
válidos. Los comandos de diagnóstico e inventario también aceptan `--json`, que
imprime los mismos datos como un objeto legible por máquina en stdout.

## ¿Todos los modelos pueden exportar a todos los formatos?

No. La cobertura es por familia y por tarea, no uniforme, y cada formato tiene
su propio extra que instalar. Cada página de modelo incluye la matriz de
exportación de su familia; la [sección de exportación](/docs/export) cubre los
formatos en sí.

## ¿Cuál es la diferencia entre segment, semantic y panoptic?

Tres tareas distintas. `segment` produce una máscara por objeto detectado.
`semantic` etiqueta cada píxel con una clase y no separa nada en instancias.
`panoptic` da a cada píxel exactamente una etiqueta, combinando las cosas
contables con la materia amorfa. Tienen distinto ground truth, distintos campos
de resultado y distintas métricas, y una familia soporta las que aparezcan en su
lista de tareas.

## ¿Cómo entreno con mis propias clases?

Escribe un YAML de dataset con `train`, `val` y `names`. Las etiquetas van junto
a las imágenes en un árbol `labels/` paralelo, un `.txt` por imagen, con
coordenadas normalizadas. `nc` es opcional y debe coincidir con `names` cuando
está presente. Ejecuta primero `libreyolo doctor <data.yaml>`: comprueba el
dataset en busca de problemas y sale con código distinto de cero cuando
encuentra errores, lo que lo hace utilizable como puerta de CI.

## ¿Por qué al cargar se imprime un aviso de metadatos?

Porque el checkpoint no lleva los metadatos completos de v1.0. La carga continúa
por una ruta de compatibilidad, y el aviso indica exactamente qué claves faltan.
Ejecuta `libreyolo metadata path=<file>` para ver qué contiene, y consulta
[checkpoints y pesos](/docs/weights) para saber qué exige el esquema.

## Un import dejó de funcionar tras una actualización. ¿Qué cambió?

Se renombraron dos clases por consistencia: `LibreYOLORTDETR` pasó a ser
`LibreRTDETR` y `LibreYOLORFDETR` pasó a ser `LibreRFDETR`. Los nombres antiguos
siguen resolviéndose y emiten un `DeprecationWarning` que apunta al nuevo, así
que el código existente sigue funcionando mientras lo actualizas.
