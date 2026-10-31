---
title: Fine-tuning con LoRA
seo_title: Fine-tuning con LoRA en LibreYOLO
description: >-
  Haz fine-tuning de un detector transformer con poca VRAM usando lora=True. Qué
  nueve familias lo soportan, la receta de adaptadores de cada una y cómo se
  comportan los checkpoints.
lead: >-
  LoRA congela las partes pesadas preentrenadas de un modelo y entrena junto a
  ellas pequeños adaptadores de bajo rango, más las capas que deben seguir
  siendo densas. En LibreYOLO toda la interfaz pública es un booleano.
keywords:
  - fine tuning lora
  - fine tuning eficiente en parametros
  - peft
  - dora
  - entrenar con poca vram
  - rf-detr lora
  - d-fine lora
  - fusionar adaptadores lora
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: La exportación fusiona los adaptadores
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Fusionar en memoria
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Instalación

LoRA se apoya en la dependencia opcional `peft`.

<code-tabs name="install" />

Sin ella, `lora=True` lanza un `ImportError` que menciona ese comando, en lugar de
entrenar un fine-tuning completo por accidente.

## Cómo se usa

<code-tabs name="train" />

`lora=True` es toda la interfaz. El rango, el alpha, el dropout y los módulos
objetivo están fijados por familia para coincidir con cada referencia original, y
no son parámetros expuestos al usuario.

Una familia que no soporta LoRA lanza un error en la configuración en lugar de
ignorar el flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

La CLI lo rechaza antes, cuando el modelo aún no se ha construido, usando su
propia lista de permitidos con esas mismas nueve familias.

## Qué familias

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 y v4, EC y ConvNeXt. La condición es
el atributo `supports_lora` de la clase trainer de cada familia, y la CLI lleva una
lista de permitidos equivalente.

La cobertura por tarea es más limitada que la cobertura por familia. D-FINE y EC
soportan solo detección, y sus rutas de segmentación y pose lanzan error. La ruta
semántica de RF-DETR lanza error. ConvNeXt es clasificación.

Todo lo demás lanza error. No hay modo parcial ni silencioso.

## Qué hace cada receta

Las recetas difieren porque las arquitecturas difieren, y una receta que funciona
sobre un backbone ViT no tiene a qué engancharse en uno convolucional.

RF-DETR usa DoRA, LoRA con descomposición de pesos, con rango 16 y alpha 16 sobre
las proyecciones de atención `query`, `key` y `value` del backbone DINOv2,
igual que la referencia de RF-DETR. El backbone ViT se congela; el proyector, el
decoder y la cabeza de detección siguen entrenándose con normalidad.

D-FINE, DEIM y RT-DETR v1, v2 y v4 combinan un backbone convolucional con un
encoder híbrido transformer y un decoder deformable, así que el reparto cambia.
El backbone convolucional se congela por completo, lo que además evita su paso
backward. Los bloques transformer congelan sus pesos base y entrenan adaptadores
LoRA estándar con el mismo rango 16 y alpha 16 sobre sus capas lineales: las capas
feed-forward `linear1` y `linear2`, el gate y las proyecciones de atención
deformable. Todo lo demás, la fusión convolucional del encoder, las proyecciones de
entrada, las cabezas de predicción y los embeddings de queries, sigue entrenándose
de forma densa.

Dos detalles de esa receta son deliberados. La self-attention del decoder se queda
congelada y sin adaptadores, porque `nn.MultiheadAttention` de PyTorch lee
`out_proj.weight` directamente y se saltaría en silencio un adaptador inyectado. Y
es LoRA estándar en lugar de DoRA, porque varias capas lineales del decoder se
inicializan a cero por diseño y la normalización de magnitud de DoRA divide por la
norma del peso.

DEIMv2 toma la misma receta con sus capas feed-forward SwiGLU `w12` y `w3` como
objetivos. Sus tamaños S, M, L y X llevan además un backbone ViT DINOv3, donde la
base ViT se congela y sus capas de atención fusionada `qkv` reciben adaptadores,
mientras que la pirámide de convoluciones del Spatial Tuning Adapter sigue
entrenándose como equivalente del proyector. Esos adaptadores `qkv` se ponen
incluso cuando la configuración ya venía con el ViT congelado, porque adaptar un
backbone congelado es justo el objetivo. Los tamaños por debajo de S usan un
backbone convolucional y toman la receta estándar.

EC es un DETR cuyo backbone es un ViT rodeado de una pirámide entrenable de
proyección convolucional. La base ViT se congela y sus capas `qkv` reciben
adaptadores, los bloques transformer toman la receta compartida, y el proyector y
las cabezas se quedan densos.

Los bloques de ConvNeXt llevan MLP lineales en formato channels-last, `fc1` y
`fc2`, y esos toman adaptadores estándar. Las convoluciones depthwise, las normas y
los parámetros de layer-scale se congelan. La cabeza de clasificación se queda
densa para que los recuentos de clases personalizados sigan funcionando.

Las cabezas de detección y clasificación se quedan siempre entrenables en todas las
recetas, porque un número de clases personalizado necesita una cabeza entrenada
desde cero.

## Checkpoints y exportación

`best.pt` y `last.pt` conservan los tensores de los adaptadores, así que un
entrenamiento con LoRA se reanuda o se inspecciona como cualquier otro. Cargar uno
de esos checkpoints requiere tener instalado el extra `lora`, porque el cargador
repite la inyección de adaptadores para que las claves encajen.

`export()` fusiona los adaptadores en pesos densos, así que el artefacto exportado
no arrastra ninguna dependencia de `peft`. La misma fusión está disponible
directamente para un modelo en memoria.

<code-tabs name="merge" />

Después de una fusión el árbol de módulos es completamente denso y una segunda
fusión no hace nada.

## Qué ahorra y qué no

LoRA recorta la memoria del optimizador y de los gradientes, y en las familias que
congelan el backbone entero también evita el paso backward de ese backbone.

La memoria de activaciones no cambia. Las activaciones del forward siguen teniendo
que conservarse para todo lo que quede entrenable, y eso suele ser lo que marca el
pico. Si vas muy justo de VRAM, baja también `batch` o `imgsz`.

## Relacionado

- [Congelación de capas](/docs/train/layer-freezing) para la otra forma de entrenar
  un subconjunto de los pesos, que funciona en todas las familias y no necesita
  ninguna dependencia extra. `freeze` y `lora=True` se combinan: los parámetros de
  los adaptadores siguen siendo entrenables incluso cuando su grupo del backbone
  está congelado.
- [Hiperparámetros](/docs/train/hyperparameters) para `batch`, `imgsz` y el resto
  de `train()`.
