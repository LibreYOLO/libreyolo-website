---
title: Dome-DETR
families: [domedetr]
seo_title: "Dome-DETR: detección de objetos diminutos en LibreYOLO"
description: "Usa Dome-DETR en LibreYOLO para detección de objetos diminutos en imágenes aéreas y de dron. Convierte los pesos de upstream, predice, haz fine-tuning y valida con código bajo licencia MIT."
lead: "Un especialista en objetos diminutos construido sobre D-FINE: una cabeza de densidad decide dónde hay objetos, la atención del encoder se restringe a las ventanas que los contienen, y el número de queries se dimensiona a partir de esa densidad en lugar de quedar fijo. LibreYOLO lo soporta para detección."
keywords: [Dome-DETR, "detección de objetos diminutos", "detección de objetos pequeños", "imágenes aéreas", "detección con drones", "teledetección", VisDrone, AI-TOD, DETR, "queries adaptativas a la densidad"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Convertir y luego predecir
      language: bash
      code: |
        # LibreYOLO no aloja pesos de Dome-DETR, así que el checkpoint se
        # descarga del repositorio upstream y se convierte una sola vez.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Una ruta local, no un nombre a secas: en esta familia no se descarga nada.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt source=drone-frame.jpg save=True
    - label: Nombres de clase
      language: python
      code: |
        from libreyolo import LibreYOLO

        # No hay checkpoint de COCO, así que las clases vienen del dataset con el
        # que se entrenaron los pesos y se leen de los metadatos del checkpoint.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 clases de AI-TOD-V2

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 clases de VisDrone
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
---

## Instalación

Dome-DETR no necesita ningún extra opcional. Todo lo que necesita importar está
en la instalación base.

```bash
pip install libreyolo
```

## Predicción

No hay nada que se descargue automáticamente. LibreYOLO no aloja estos pesos,
así que el flujo es: descargar el checkpoint de upstream, convertirlo una vez y
luego cargar por ruta el archivo convertido. [Licencia](#licensing) explica por
qué.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran la selección de queries; `iou` se acepta por paridad de API pero no
tiene efecto, porque el decoder es un predictor de conjuntos sin paso de NMS.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

Dos capacidades están desactivadas en esta familia. La captura de grafos CUDA
está deshabilitada, porque el número de queries de PAQI depende de los datos y,
por tanto, el forward cambia de forma de una imagen a otra, que es justo lo que
la captura de grafos no puede absorber. El aumento de datos en test (TTA) se
ejecuta a un único tamaño cuadrado fijo, así que pedir un TTA multiescala no
hace nada.

## Variantes

Tres tamaños, s, m y l, todos a 800 por 800. El tamaño selecciona el backbone, y
el dataset del que vienen los pesos selecciona la profundidad del decoder y el
presupuesto de queries, así que un código de tamaño por sí solo no identifica un
grafo. Los pesos de AI-TOD-V2 seleccionan entre 300 y 1500 queries por imagen,
los de VisDrone entre 250 y 500, y el modelo grande ejecuta cuatro capas de
decoder en AI-TOD-V2 frente a seis en VisDrone.

Dome-DETR es D-FINE con tres añadidos. DeFE predice un mapa de densidad. MWAS
usa ese mapa para restringir la atención del encoder a las ventanas que de
verdad contienen objetos, en lugar de atender a todas partes. PAQI dimensiona el
conjunto de queries a partir de esa misma densidad en lugar de decodificar unas
300 fijas. La ganancia se concentra donde los objetos son más pequeños, y se
estrecha a medida que crecen: la ablación del propio upstream mueve el AP en
objetos muy diminutos de 14.0 a 17.8, mientras que el AP en objetos medianos
solo pasa de 45.4 a 46.4. Trátalo como un compañero de
[D-FINE](/docs/models/d-fine) para imágenes aéreas, de dron y de teledetección,
no como un reemplazo.

LibreYOLO no publica filas de benchmark para esta familia, porque no publica
checkpoints que medir.

## Entrenamiento

Dome-DETR es entrenable. El entrenamiento ejecuta el objetivo completo de
upstream: las losses de D-FINE más la supervisión de densidad y de conteo de
DeFE, con las queries de relleno enmascaradas fuera de los términos de
clasificación y con máscaras de atención de denoising por imagen, para que el
padding de una imagen no pueda filtrarse en el de otra.

<code-tabs name="train" />

La configuración hereda la receta de D-FINE y cambia lo que MWAS exige. `imgsz`
es 800, `lr0` es `2e-4`, el grupo de parámetros del backbone se escala con
`backbone_lr_mult=0.1`, y `multi_scale` se fuerza a desactivado, porque las
ventanas de MWAS necesitan que la entrada siga siendo divisible por el stride 8.
`batch` vale 4 por defecto en lugar de los 16 de D-FINE: PAQI rellena cada batch
hasta su miembro más ancho, así que la memoria sigue a la imagen más cargada del
batch en vez de a la media.

Una advertencia honesta sobre la precisión. Upstream entrena 160 epochs con
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, mientras que estos valores por
defecto ejecutan el schedule flat-cosine de D-FINE durante esos mismos 160
epochs. Ese schedule no se ha reproducido aquí, y las cifras de AP del paper
tampoco, así que léelas como resultados de los autores de upstream y no como una
promesa de que esta receta los alcanza. Proporciona el schedule de upstream si
el objetivo es igualar el paper.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario indexado por nombre de métrica, e imprime
resultados por clase si dejas `verbose` activado.

<code-tabs name="val" />

La validación se ejecuta contra tu propio dataset, en el formato con el que
entrenaste. El gate de validación COCO de la biblioteca no se aplica aquí, ya
que no existe ningún checkpoint de COCO de esta familia contra el que medir.

## Exportación

La exportación no está soportada, en ningún formato, y pedir una lanza un error
en vez de producir un archivo.

El motivo es PAQI. Decide el número de queries por imagen a partir de propuestas
filtradas por densidad y de un bucle voraz de supresión adaptativa a la
densidad, así que la longitud de salida del decoder es una propiedad de la
entrada y no del grafo. El trazado fija el número que la imagen de trazado
produjera en ese momento, lo que da un artefacto que devuelve resultados
incorrectos de forma silenciosa para cualquier otra imagen. Una formulación
estática tendría que desenrollar esa supresión sobre los 250 a 1500 candidatos,
y reducirla a un top-k fijo eliminaría exactamente el recall en objetos
diminutos que justifica la existencia de esta familia. Si necesitas un
transformer de detección exportable, [D-FINE](/docs/models/d-fine) es al que
recurrir.

## Checkpoints

No hay ninguno que listar. LibreYOLO no publica pesos de Dome-DETR, y ningún
nombre de la forma `LibreDOMEDETR<size>-<dataset>.pt` resuelve a una descarga.

Upstream publica seis checkpoints, s, m y l para cada uno de dos datasets:
AI-TOD-V2 con 9 clases y VisDrone con 12. No hay checkpoint de COCO, así que un
nombre de archivo canónico siempre lleva el sufijo del dataset, y los nombres de
clase viajan en los metadatos del checkpoint en vez de venir de una constante de
la familia. Pedir un `LibreDOMEDETRs.pt` a secas lanza un error de inmediato,
con un mensaje que nombra los dos archivos reales y el comando de conversión, en
vez de intentar una descarga que daría 404.

`weights/convert_domedetr_weights.py` se encarga de la conversión. Reconstruye
el grafo de LibreYOLO, carga en él los tensores de upstream y se niega a
escribir nada si una sola clave falta, sobra o tiene la forma equivocada, así
que un archivo convertido o coincide exactamente o no existe. Apúntalo a un
`.pth` de upstream y pasa el tamaño y la variante:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Sobre la fidelidad numérica, `weights/parity_domedetr.py` compara este port con
la implementación de upstream en los seis checkpoints y reporta
`max_abs_diff == 0.0` tanto en `pred_logits` como en `pred_boxes`, después de
comprobar antes la máscara de ventanas de MWAS bit a bit, y aparte compara cada
término de la loss con el criterion de upstream. Que quede claro qué es eso: un
script manual que necesita el checkout de upstream y los checkpoints publicados
en disco, ejecutado a mano. No forma parte de la integración continua, y ningún
job de CI lo reproduce.

## Licencia

<provenance-box>

Los pesos son la razón por la que esta familia no está replicada. La model card
de upstream no lleva campo de licencia en sus metadatos, y su texto afirma que
el proyecto es Apache-2.0 mientras a la vez restringe el material a fines
exclusivamente de investigación académica. Esas dos lecturas no concuerdan, y la
más estricta no es una autorización de redistribución, así que LibreYOLO enlaza
el repositorio upstream en vez de copiar los archivos, a la espera de una
aclaración. Ese mismo razonamiento es el que rige aquí para
[YOLO-NAS](/docs/models/yolo-nas).

El código es una cuestión aparte, y más clara. El repositorio upstream es
Apache-2.0, el port de LibreYOLO es MIT, y los pesos que entrenes tú con tus
propios datos son tuyos.

</provenance-box>

## Cita

Dome-DETR se publicó en ACM Multimedia 2025 con el título «Dome-DETR: DETR with
Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection». El preprint está en
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Los autores no
publican ningún bloque BibTeX en su repositorio, así que aquí no se reproduce
ninguno en vez de montarlo a mano.

<citation-block />
