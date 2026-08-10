---
title: Detección de vocabulario abierto
seo_title: "Detección de vocabulario abierto en LibreYOLO"
description: "Detecta objetos a partir de un vocabulario de texto en LibreYOLO. Carga Grounding DINO, OWLv2, OMDet-Turbo u OV-DEIM a través de LibreOpenVocab y define las clases en tiempo de ejecución."
lead: "La detección de vocabulario abierto sustituye la lista fija de clases de un checkpoint por las palabras que elijas en el momento de la llamada. En LibreYOLO no es una tarea aparte: es la tarea detect servida por un nivel de modelos aparte, que se carga mediante la factoría LibreOpenVocab en lugar de LibreYOLO."
keywords: [detección de vocabulario abierto, detección de objetos zero shot, open vocabulary detection python, grounding dino python, owlv2, omdet turbo, detectar objetos con un prompt de texto]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Cambiar el vocabulario
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes es persistente: se mantiene hasta la siguiente llamada.
        # Las etiquetas deben ser únicas en minúsculas y sin artículos.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Umbral de texto de Grounding DINO
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtra por la puntuación del box, y text_threshold por la
        # puntuación de tokens de la frase decodificada. Ambos valen 0.25 por
        # defecto. Solo Grounding DINO acepta text_threshold; los demás fallan.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
---

## Definición

La detección de vocabulario abierto devuelve `Results` de detección normales:
bounding boxes, confianzas e índices de clase, con `result.names` mapeando esos
índices de vuelta a las cadenas que pediste. Lo que cambia es de dónde sale la
lista de clases. Un detector convencional se entrena contra un conjunto fijo de
categorías y nunca puede emitir una categoría fuera de él. Estos modelos reciben
el vocabulario como texto en tiempo de inferencia, así que
`set_classes(["forklift", "safety cone"])` basta para que esas sean las clases.

LibreYOLO no tiene una clave de tarea `open-vocabulary`. Estos modelos declaran
`SUPPORTED_TASKS = ("detect",)` como cualquier otro detector. Lo que los separa
es la ruta de carga: son snapshots de Hugging Face en lugar de checkpoints de
state-dict de LibreYOLO, así que quedan fuera de la factoría `LibreYOLO()` y se
construyen mediante `LibreOpenVocab()`. Esa factoría es hermana de `LibreSAM()`
y `LibreVLM()`, no un reemplazo de `LibreYOLO()`.

Las puntuaciones son puntuaciones de detección reales, no un caption generado y
parseado a posteriori. Cada familia puntúa regiones de la imagen contra el
embedding de texto de cada prompt.

## Modelos

Cuatro familias componen este nivel, y todas son solo de predicción. Carga
cualquiera de ellas por alias mediante `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino), de IDEA Research, en tamaños `t`
y `b`. Es la opción por defecto del nivel, y la única familia que acepta
`text_threshold`, un segundo corte sobre la puntuación de tokens de la frase
decodificada.

[OWLv2](/docs/models/owlv2), de Google Research, en tamaños `b16` y `l14`.
Puntúa regiones de la imagen contra embeddings de texto de un encoder tipo CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo), de Om AI Lab, en un único tamaño `t`.
Desacopla los embeddings de clase de un prompt de tarea en lenguaje natural, y
es la única familia aquí que suprime bounding boxes solapados dentro de su
propio post-procesado, así que `iou=` sí se respeta.

[OV-DEIM](/docs/models/ov-deim), en tamaños `s`, `m` y `l`, un detector de tipo
DETR que empareja las queries del decoder con embeddings de texto de una torre
de texto MobileCLIP incluida. Usa emparejamiento uno a uno con selección top-K,
así que no se ejecuta NMS en ninguna parte.

Los pesos de OV-DEIM son el caso restringido de este nivel. Los pesos del
detector son CC BY-NC 4.0, no comercial. La torre de texto incluida está bajo la
Machine Learning Research Model license de Apple, solo para uso de
investigación. El checkpoint `l` añade un fine-tune de backbone DINOv3-S bajo la
DINOv3 License de Meta. Los tres textos de licencia van dentro del repositorio
de pesos, y la biblioteca registra el mismo resumen cuando resuelve los pesos,
antes de construir el modelo. Lee [OV-DEIM](/docs/models/ov-deim) antes de
desplegarlo.

Este nivel necesita un extra:

```bash
pip install "libreyolo[openvocab]"
```

Eso cubre `transformers` y `timm` para las tres familias envueltas, y los
paquetes `huggingface_hub`, `safetensors`, `regex` y `ftfy` que OV-DEIM necesita
al ser un port nativo.

Hay un segundo nivel que también recibe un vocabulario de texto: `LibreVLM()`
carga modelos generativos de visión y lenguaje, como
[Qwen3-VL](/docs/models/qwen3-vl) y [Florence-2](/docs/models/florence-2), y
convierte su salida en los mismos `Results`. Comparte la superficie de
`set_classes()`. La diferencia está en qué produce los bounding boxes: las
familias de esta página son detectores discriminativos que emiten puntuaciones
directamente, mientras que el nivel de VLM los genera.

## Predicción

<code-tabs name="predict" />

`set_classes()` recibe una lista no vacía de cadenas de etiqueta y se mantiene
hasta que se vuelve a llamar. Las etiquetas deben ser únicas una vez pasadas a
minúsculas y sin los artículos iniciales, así que `"a bus"` y `"bus"` no pueden
coexistir en un mismo vocabulario. Las frases de varias palabras son etiquetas
como cualquier otra, y cada familia convierte la lista en su propia entrada de
texto antes de tokenizar, así que `"traffic cone"` es una query distinta de
`"cone"`.

Tres argumentos de predicción se comportan aquí de forma distinta que en un
detector nativo. `imgsz=` se rechaza, porque el procesador se encarga del
redimensionado en estas familias. `augment=True` se rechaza, ya que el aumento
de datos en test queda fuera del alcance del nivel. `iou=` solo aplica a la
familia cuyo procesador ejecuta su propia supresión; donde no se suprime nada,
pasarlo genera un aviso y se ignora.

Si no se define, `conf` toma el valor por defecto de la familia cargada en lugar
del 0.25 habitual de `predict()`, y ese valor por defecto no es el mismo en todo
el nivel. Defínelo explícitamente cuando compares dos familias sobre la misma
imagen.

`track()` falla en todo el nivel. Ejecuta `predict()` por frame en su lugar.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Entrenamiento

Ninguna familia de este nivel se entrena dentro de LibreYOLO. `train()` falla:
haz el fine-tuning upstream y carga los pesos resultantes. El vocabulario que
pasas a `set_classes()` es el único ajuste que cambia lo que detecta un modelo
cargado.

## Validación

No hay validador para este nivel, y `val()` falla. La validación de vocabulario
abierto necesita uno dedicado, porque el validador de detección estándar pasa
tensores de imagen directamente al modelo, mientras que estas familias requieren
entradas condicionadas por texto construidas junto a ellos.

## Exportación

La exportación queda fuera del alcance del nivel y `export()` falla. Estos
modelos se ejecutan mediante `predict()` en PyTorch.
