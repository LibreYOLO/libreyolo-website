---
title: Segmentación con prompts
seo_title: Segmentación con prompts en LibreYOLO
description: >-
  Convierte un punto, un box o un concepto de texto en la máscara de un objeto
  con LibreYOLO. Carga SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM o PicoSAM3 a través
  de LibreSAM.
lead: >-
  La segmentación con prompts convierte un clic en una máscara: señalas un
  objeto, o dibujas un box a su alrededor, y el modelo devuelve su contorno. En
  LibreYOLO no es una clave de tarea aparte, sino un nivel de modelos que se
  carga mediante la factoría LibreSAM y cuyos resultados son Results de
  segmentación normales.
keywords:
  - segmentación con prompts
  - segmentación interactiva
  - segment anything python
  - prompt de punto
  - prompt de box
  - SAM python
  - máscara a partir de un clic
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de punto y de box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Un punto es [x, y] en píxeles; las labels son 1 positivo, 0 negativo.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polígonos
        print(result.boxes.xyxy)    # boxes ajustados derivados de las máscaras

        # Un prompt de box da una máscara por box.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Codificar una vez, lanzar muchos prompts'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # set_image ejecuta una sola vez el pesado encoder de imagen y lo
        cachea.

        model.set_image(SAMPLE_IMAGE)

        first = model.predict(points=[640, 420], labels=[1])

        second = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
    - label: Segmentarlo todo
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Sin prompt se usa una rejilla de puntos sobre toda la imagen. La
        # rejilla por defecto de 32 por lado son ~1024 pasadas del decoder,
        # lo que resulta lento en CPU.
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: Máscaras de ambigüedad
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Un punto puede significar una manga, una camisa o una persona.
        # multimask=True devuelve las tres máscaras de todo-frente-a-parte
        # en lugar de solo la mejor.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definición

La segmentación con prompts recibe una imagen más un prompt espacial y devuelve
la máscara de aquello a lo que apunta el prompt. No se clasifica nada: no hay
lista de clases, y `result.boxes` contiene boxes ajustados derivados de las
máscaras, no detecciones por derecho propio. `result.masks` lleva los datos de
las máscaras y `result.masks.xy` sus polígonos.

El prompt es la interfaz. `points` son coordenadas de píxel `[x, y]`, un conjunto
por objeto, con `labels` marcando cada punto como positivo (1, inclúyelo) o
negativo (0, exclúyelo). `bboxes` es `[x1, y1, x2, y2]`, una máscara por box.
Puntos y boxes se pueden combinar, en cuyo caso se emparejan por objeto y deben
tener la misma longitud. Si omites todos los prompts se ejecuta la ruta de
segmentarlo todo, una rejilla de puntos sobre la imagen.

Un solo punto es ambiguo por construcción. Hacer clic en una manga podría
significar la manga, la camisa o la persona, así que `multimask=True` devuelve
esas tres máscaras de todo-frente-a-parte por prompt en lugar de la única mejor.
`conf` filtra por el IoU predicho por el modelo, una puntuación de calidad de la
máscara, no una confianza de detección.

LibreYOLO no tiene una clave de tarea `promptable`. El nivel se registra como
`segment`, la misma clave que usa la segmentación de instancias. Lo que lo
diferencia es la forma de la llamada, y por eso tiene su propia factoría,
`LibreSAM()`, hermana de `LibreYOLO()`, `LibreOpenVocab()` y `LibreVLM()`. Una
sola firma `predict(image)` no puede expresar el bucle para el que están hechos
estos modelos: `set_image()` ejecuta el encoder de imagen una vez y cachea los
embeddings, cada llamada posterior a `predict()` con `source=None` solo paga el
decodificado del prompt, y `reset_image()` limpia la caché. El encoder de imagen
es el coste dominante y se ejecuta una vez por imagen, así que un segundo prompt
sobre la misma imagen se lo salta por completo.

## Modelos

Seis familias se cargan a través de `LibreSAM` por alias.

[SAM](/docs/models/sam) es el modelo por defecto, en tamaños `base`, `large` y
`huge`, también escritos `b`, `l` y `h`.

[SAM 2](/docs/models/sam-2), como `sam2-tiny`, `sam2-small`, `sam2-base-plus` y
`sam2-large`. LibreYOLO soporta su ruta de imagen.

[SAM 3](/docs/models/sam-3), como `sam3`, es la única familia que acepta un
prompt de concepto en texto: `text="yellow school bus"` devuelve todas las
instancias que coincidan. Pasar `text=` a cualquier otra familia falla con un
mensaje que menciona SAM 3. Sus pesos vienen de Meta bajo la SAM License
personalizada en lugar de la licencia MIT de LibreYOLO, y el repositorio está
restringido: acepta los términos en la página del modelo y autentícate con
`hf auth login` antes de la primera descarga. Lee
[SAM 3](/docs/models/sam-3) antes de desplegarlo.

[EdgeTAM](/docs/models/edgetam), como `edgetam`, es una variante on-device de
SAM 2. LibreYOLO soporta su ruta de imagen.

[MobileSAM](/docs/models/mobilesam), como `mobilesam`, sustituye el encoder
ViT-H de SAM por uno TinyViT destilado.

[PicoSAM3](/docs/models/picosam3), como `picosam3`, es una CNN compacta para
regiones indicadas con box en sensores edge. Aquí los prompts de box son todo el
contrato: puntos, texto, máscara, multimask y segmentarlo todo fallan con un
mensaje que apunta a SAM 2 o SAM 3.

El extra del nivel cubre las cuatro familias que se cargan mediante
`transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM y PicoSAM3 son ports nativos de LibreYOLO y no necesitan instalar
`transformers` para funcionar.

## Predicción

<code-tabs name="predict" />

`source` y `set_image()` son alternativas, no una secuencia: pasa una imagen a
`predict()` para una llamada de un solo paso, o llama antes a `set_image()` y
luego a `predict(source=None)` para cada prompt. Pasar `device=` a `predict()`
mueve el modelo para esa llamada y todas las posteriores, e invalida cualquier
embedding cacheado.

Segmentarlo todo es el modo caro. `points_per_side` vale 32 por defecto, lo que
son aproximadamente 1024 pasadas del decoder sobre la imagen; bájalo para
cualquier cosa interactiva en CPU. En ese modo `conf` aplica el umbral de rejilla
de la familia cuando lo dejas sin definir, mientras que en la ruta con prompt un
`conf` sin definir conserva todas las máscaras. Pasa `conf=0.0` para desactivar
el filtrado en cualquiera de los dos modos, y `max_det` para limitar cuántas
máscaras vuelven.

Los prompts de máscara no están soportados en esta versión, y `masks=` falla en
lugar de ignorarse. `track()` también falla en todo el nivel: son segmentadores
de imagen, así que ejecuta `predict()` por frame. Consulta
[predicción](/docs/predict) para fuentes y manejo de resultados.

## Entrenamiento

Ninguna familia de este nivel se entrena dentro de LibreYOLO. `train()` falla:
haz fine-tuning aguas arriba y carga los pesos resultantes.

## Validación

No hay validador para este nivel, y `val()` falla. Una máscara con prompt no
tiene un conjunto fijo de clases contra el que puntuar, así que las métricas
habituales de detección y segmentación no tienen a qué agarrarse. Puntuar una
máscara con prompt significa compararla con una máscara de referencia que
aportes tú, frente a los prompts que te importen.

## Exportación

La exportación queda fuera del alcance del nivel en su conjunto y `export()`
falla, con una excepción. [PicoSAM3](/docs/models/picosam3) exporta a ONNX su CNN
de región 96x96 en bruto como `roi_image -> mask_logits`; el recorte del box y el
redimensionado de la máscara de vuelta a coordenadas de imagen se quedan en
Python. Todas las demás familias se ejecutan a través de `predict()` en PyTorch.
Consulta [exportación](/docs/export) para los formatos disponibles en el resto de
la biblioteca.
