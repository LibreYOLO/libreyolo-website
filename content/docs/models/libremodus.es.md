---
title: LibreMODUS
families: [libremodus]
seo_title: "LibreMODUS en LibreYOLO: análisis de imagen any-to-any"
description: "Usa LibreMODUS en LibreYOLO para profundidad, normales, bordes y detección, y para componerlos con any2any(). Solo inferencia; los pesos se cargan desde EPFL-VILAB."
lead: "LibreMODUS es una integración solo de inferencia del checkpoint MODUS 14B-A7B, un modelo any-to-any que convierte una entrada derivada de imagen en otra: entra RGB, sale profundidad; entra profundidad, salen normales; cualquiera de ellas más una frase, salen cajas. LibreYOLO soporta cuatro tareas a través de la API estándar de predict y un conjunto más amplio a través de any2any()."
keywords: [LibreMODUS, MODUS, any-to-any, "estimación de profundidad", "normales de superficie", "detección de bordes", "detectar objetos con una frase", EPFL VILAB]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Sin vocabulario personalizado, detect decodifica los tokens de
        # etiqueta COCO del checkpoint en ids de clase COCO-80 contiguos.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding por frase
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() cambia la detección a grounding por frase: cada frase
        # se ejecuta por separado y vuelve por el mismo contrato Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # De una a tres entradas derivadas de imagen (rgb, depth, normal,
        # canny/edge), más texto auxiliar opcional, compuestas hacia un objetivo.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # El grounding a través de any2any() necesita una entrada de texto
        # que nombre la frase.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
---

## Instalación

LibreMODUS necesita su propio extra, que arrastra `accelerate` para el dispatch de modelo grande que requiere este checkpoint.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO no redistribuye ni replica los pesos de MODUS. Por defecto, cargar un modelo `LibreMODUS` descarga los archivos necesarios directamente desde `EPFL-VILAB/MODUS` en una revisión fijada de Hugging Face, y una descarga nueva siempre necesita la cuenta autenticada de Hugging Face del propio usuario, incluso si la puerta de acceso del alojamiento upstream está temporalmente abierta. Revisa y acepta los términos upstream y después autentícate:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Para evitar cualquier petición de red, apunta a un snapshot que ya tengas:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Ese directorio debe contener `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` y `merges.txt`. Consulta Licencia más abajo para saber qué permiten los términos del checkpoint.

## Predicción

<code-tabs name="predict" />

La API estándar de tareas cubre cuatro tareas, cada una mapeada a un objetivo de MODUS: `depth` a profundidad relativa (`result.depth_map`), `normal` a normales de superficie (`result.normal_map`), `edge` a bordes al estilo Canny (`result.edges`) y `detect` a cajas COCO-80 (`result.boxes`), salvo que `set_classes()` lo cambie a grounding por frase. `set_task()` alterna entre ellas sobre el mismo modelo ya cargado. La receta publicada usa diez pasos de flow sampling con guidance de texto 4.0 y guidance de imagen 2.0; puedes sobrescribirlos con `inference_steps=`, `inference_cfg=` e `inference_image_cfg=` en la construcción.

`any2any()` da acceso a la superficie pública de análisis más amplia: de una a tres entradas derivadas de imagen (`rgb`, `depth`, `normal`, `canny`/`edge`), más texto auxiliar opcional, compuestas hacia cualquiera de estos objetivos: profundidad, normales, bordes, bordes derivados de SAM, detección COCO o grounding por frase. Todas las entradas derivadas de imagen deben describir el mismo lienzo alineado; LibreMODUS rechaza anchuras y alturas que no coincidan en lugar de redimensionarlas por separado. `chain=(...)` genera objetivos intermedios y los realimenta en el mismo contexto, dentro del presupuesto de tres condiciones con el que se entrenó el checkpoint. `verify=N` (N >= 2) genera N candidatos y se queda con el que mejor puntúa en una comprobación restringida de autoconsistencia, expuesta como `result.verification_score`.

`dtype="bf16"` (el valor por defecto) coincide con la precisión del checkpoint publicado; `dtype="fp8"` almacena como E4M3 los pesos lineales elegibles del tronco del decoder, con una escala por canal de salida, hace la conversión una sola vez en una caché local bajo `~/.cache/libreyolo/modus/fp8` y descuantiza al dtype de entrada en cada multiplicación de matrices, de modo que lo que sacrifica es memoria y no precisión a nivel de activaciones.

`train()`, `val()` y `export()` lanzan excepción: LibreMODUS es solo de inferencia, no se ofrece validación sobre dataset y no hay ninguna vía de exportación a ONNX, TensorRT ni TFLite. Tampoco se soportan el `predict()` por batches ni el test-time augmentation; cada llamada procesa una imagen.

## Licencia

<provenance-box>

LibreYOLO no aloja ni replica el checkpoint de MODUS en ningún sitio, tampoco en su propia organización de Hugging Face: cargarlo siempre trae la revisión fijada directamente desde EPFL-VILAB/MODUS, o lee un snapshot que ya esté en disco en `checkpoint_path`.

</provenance-box>

## Cita

<citation-block />
