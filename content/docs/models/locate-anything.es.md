---
title: LocateAnything
families: [locateanything]
seo_title: "LocateAnything: detección de vocabulario abierto y puntos"
description: "Usa LocateAnything en LibreYOLO para detección de vocabulario abierto y localización por puntos. Predice con cualquier etiqueta de texto; el entrenamiento, la validación y la exportación no están soportados."
lead: "LocateAnything es un modelo de grounding de visión y lenguaje publicado por NVIDIA que decodifica bounding boxes y puntos en paralelo, en lugar de ir token de coordenada en token de coordenada. LibreYOLO lo envuelve como detector y localizador por puntos de vocabulario abierto: cualquier lista de etiquetas de texto se convierte en el conjunto de clases, sin cabeza fija y sin necesidad de hacer fine-tuning."
keywords: [LocateAnything, NVIDIA, "modelo de visión y lenguaje", "detección de vocabulario abierto", "detección de puntos", VLM, grounding, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Vocabulario abierto: vale cualquier palabra, no una cabeza de clases
        # fija. Persiste en cada predict()/track() posterior hasta volver a fijarlo.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt de punto
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" devuelve un punto por objeto encontrado en vez de una caja.
        # Cambia de tarea en un modelo ya cargado con model.set_task("point").
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Chat directo
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # La vía de escape bajo la comodidad de la detección: preguntas libres,
        # recuentos o cualquier prompt que el envoltorio de cajas no cubra.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## Instalación

LocateAnything necesita el extra `vlm`, que arrastra `transformers` junto con
los paquetes `decord`, `lmdb` y `peft` que importa al cargarse su código
remoto de Hugging Face.

```bash
pip install "libreyolo[vlm]"
```

## Predicción

`LibreLocateAnything` es una clase de Python, no un checkpoint `.pt`: no se
carga a través de la factoría `LibreYOLO()` y la CLI de `libreyolo` no lo
resuelve. La factoría `LibreVLM(...)` (`from libreyolo import LibreVLM`)
también llega a esta familia por alias, por ejemplo
`LibreVLM("locate-anything")`; la clase que se usa abajo es lo que construye.
Cargarla descarga y ejecuta el código remoto del modelo publicado por NVIDIA
en Hugging Face, así que LibreYOLO fija la descarga a una revisión de commit
concreta en lugar de a la rama mutable `main`, y registra un aviso de licencia
una única vez antes de la primera descarga.

<code-tabs name="predict" />

`result.boxes` (tarea `detect`) y `result.points` (tarea `point`) llevan la
salida parseada igual que en cualquier otra familia. La confianza es un
marcador de posición: LocateAnything no emite ninguna puntuación por caja, así
que todas las detecciones reciben la misma confianza constante, y `conf=` solo
descarta las filas por debajo de esa constante, no las ordena. Si te saltas
`set_classes()`, el vocabulario cae por defecto en los nombres de COCO-80.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Un único tamaño publicado, 3b. Dos tareas comparten los mismos pesos: `detect`
(la predeterminada) devuelve cajas y `task="point"` devuelve en su lugar un
único punto por objeto encontrado, en `result.points`; se cambia de una a otra
sobre un modelo ya cargado con `model.set_task("point")`. El harness de
benchmarks de LibreYOLO no ha medido esta familia, así que no hay cifras de
precisión publicadas con las que compararlo.

LibreYOLO expone esta familia solo para predicción. `train()`, `val()` y
`export()` lanzan todos `NotImplementedError`: haz el fine-tuning upstream y
carga el resultado, la validación sobre dataset se omite porque una confianza
de marcador de posición haría engañoso el mAP de COCO, y la exportación queda
fuera del alcance para un modelo generativo sin state dict que trazar.

## Licencia

<provenance-box>

La NVIDIA License permite el uso, la reproducción y la modificación, pero
restringe el modelo y cualquier derivado a uso no comercial, de investigación
o de evaluación únicamente, para cualquiera que no sea NVIDIA y sus filiales:
no hay ningún umbral de ingresos ni excepción de pago. LocateAnything-3B
además combina otros dos componentes con licencia propia: un backbone de
lenguaje Qwen2.5-3B-Instruct bajo la Qwen Research License y un codificador de
visión MoonViT-SO-400M bajo MIT. LibreYOLO no aloja, replica ni redistribuye
nada de ello: `LibreLocateAnything` descarga los pesos y el código remoto
necesario directamente de `nvidia/LocateAnything-3B` en Hugging Face, fijados a
un commit concreto, la primera vez que se ejecuta.

</provenance-box>

## Cita

<citation-block />
