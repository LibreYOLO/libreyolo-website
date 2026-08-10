---
title: Reconocimiento facial
seo_title: "Reconocimiento facial en LibreYOLO"
description: "Detecta, genera embeddings e identifica caras en LibreYOLO. Registra una galería, compara dos imágenes y empareja por similitud del coseno, desde Python o la CLI."
lead: "El reconocimiento facial es la tarea embed aplicada a caras. Un detector localiza y alinea cada cara, una cabeza de reconocimiento devuelve un vector normalizado con L2 por cara, y la identidad se decide por similitud del coseno frente a referencias registradas, no por una lista fija de clases."
keywords: [reconocimiento facial python, embeddings de caras, verificación facial python, comparar dos caras, identificar personas en fotos, arcface onnx, similitud del coseno caras]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Los nombres librefacerec-* apuntan a la familia de embeddings faciales
        # sea cual sea el sufijo del archivo, y se descargan de la org de LibreYOLO
        # en Hugging Face en el primer uso junto con el detector de caras por defecto.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) boxes de caras
        print(result.embeddings.data.shape)  # (N, D), una fila por cara
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Comparar dos imágenes
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Ejecuta detección y embedding en ambas imágenes y compara la cara
        # con más confianza de cada una. La similitud del coseno está en [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Registrar una galería e identificar
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name es None por debajo del umbral
    - label: Registrar e identificar desde la CLI
      language: bash
      code: |
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=faces.npz
        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg gallery=faces.npz
    - label: Usa tus propios boxes de caras
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes se salta la detección por completo; face_detector acepta un
        # callable, un modelo de detección de LibreYOLO o una instancia de FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
---

## Definición

El reconocimiento facial devuelve un vector por cara, no una etiqueta. La
predicción se ejecuta en dos etapas: un detector de caras localiza cada cara y
sus cinco landmarks, el recorte se deforma hasta una alineación canónica de
112x112, y una cabeza de reconocimiento emite un embedding normalizado con L2.

`result.embeddings` es un payload `Embeddings` de forma `(N, D)`, alineado fila a
fila con `result.boxes`, de modo que la fila `i` describe la cara del box `i`.
Como las filas son vectores unitarios, la similitud del coseno es un producto
escalar, y `embeddings.similarity()` la calcula contra otro `Embeddings` o contra
una matriz entera en una sola llamada.

Ponerle nombre a una cara es un paso aparte. Una `Gallery` guarda vectores de
referencia con nombre; pasar `gallery=` a `predict()` añade `result.identities`,
alineado fila a fila con los embeddings, que lleva un nombre y su mejor
puntuación de coseno por cara. Una cara por debajo del umbral de coincidencia
conserva `None` como nombre, y nunca se sustituye por el nombre más cercano que
quedó por debajo del umbral.

La clave canónica de la tarea en la biblioteca es `embed`. `face-recognition`,
`facial-recognition`, `reid` y `face` se normalizan todas a ella, así que
`task="face-recognition"` y `task="embed"` seleccionan lo mismo. Las caras son la
variante por regiones de esa tarea más amplia;
[embeddings](/docs/tasks/embeddings) cubre las variantes de imagen completa y de
texto, la API compartida de `Embeddings`, `Identities` y `Gallery`, y los modelos
que producen vectores sin detectar nada.

## Modelos

[LibreFaceRec](/docs/models/librefacerec) es la familia de esta tarea. Son dos
artefactos ONNX detrás de una sola llamada: `librefacerec-l.onnx`, una cabeza de
reconocimiento iResNet100 que produce embeddings de 512 dimensiones, y
`librefacerec-det.onnx`, el detector de caras por defecto con cinco landmarks,
tomado del zoo de OpenCV. Ambos se descargan de la org de LibreYOLO en Hugging
Face en el primer uso. Cualquier otro archivo ONNX con la convención de ArcFace
(entrada alineada de 112x112, salida `(N, D)`) puede sustituir a la cabeza de
reconocimiento pasando su ruta en lugar de un nombre `librefacerec-*`.

La clave de tarea `embed` es más amplia que las caras. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) y [DINOv2](/docs/models/dinov2) también admiten
`task="embed"` y devuelven un único vector de la imagen completa, lo que es
recuperación de imágenes y no identidad facial. Comparten la API de `Gallery` y
`Embeddings`, así que el flujo de registrar y emparejar que se ve más abajo es
trasladable, pero no detectan ni alinean caras.

La cabeza de reconocimiento funciona a través de `onnxruntime`, que la
instalación base no incluye:

```bash
pip install "libreyolo[onnx]"
```

## Predicción

<code-tabs name="predict" />

Si no se le indica nada, `predict()` descarga y empareja el detector por defecto.
`face_detector` lo reemplaza por un callable, un modelo de detección de LibreYOLO
o una instancia de `FaceDetector`, y se puede fijar en el constructor o en cada
llamada. `face_boxes` se salta la detección usando boxes que ya tengas. En la
CLI, `face_detector=` acepta la ruta a un `.onnx` de detector de caras o el
nombre de un detector de LibreYOLO.

`model.verify(image_a, image_b)` es el atajo para dos imágenes: genera el
embedding de la cara con más confianza de cada una y devuelve
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` devuelve
todas las filas de caras de una o varias imágenes apiladas en un único tensor
`(N_total, D)`. Consulta [predicción](/docs/predict) para fuentes, streaming y
manejo de resultados.

## Formato del dataset

El registro lee una carpeta por identidad. El nombre de la carpeta pasa a ser la
identidad, y cada imagen que contiene aporta referencias para ese nombre:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` recorre ese árbol y escribe una galería `.npz`. Un archivo de
galería existente se amplía en el sitio en vez de reemplazarse, así que se pueden
ir añadiendo identidades con el tiempo. Las galerías quedan ligadas a los pesos
que las produjeron por la dimensión del embedding y una huella del archivo;
emparejar con un modelo distinto lanza un error en lugar de comparar espacios
vectoriales incompatibles.

Por defecto cada imagen de origen aporta una fila de referencia, la cara con más
confianza, así que un retrato con gente de fondo solo registra a su sujeto. Pasa
`select="all"` a `Gallery.enroll` para guardar todas las filas devueltas.

## Entrenamiento

Ninguna familia de esta tarea se entrena dentro de LibreYOLO.
`LibreFaceEmbedder.train()` lanza un error: entrena una cabeza de reconocimiento
fuera de la biblioteca, expórtala a ONNX con la convención de ArcFace y carga el
archivo por ruta.

## Validación

No hay validador de datasets para esta tarea, y `val()` lanza un error en vez de
fingir lo contrario. La precisión de verificación se mide sobre pares de imágenes
etiquetados con `model.verify()`, barriendo `threshold` para elegir el punto de
operación que quieras. La precisión de identificación se mide registrando una
galería y leyendo `result.identities.name` y `result.identities.score` sobre
imágenes reservadas, contando un nombre `None` como rechazo.

## Exportación

La cabeza de reconocimiento ya es un grafo ONNX, así que no hay nada que
convertir: `LibreFaceEmbedder.export()` lanza un error. Despliega el archivo
`.onnx` directamente, o apunta LibreYOLO hacia él y deja que la familia se
encargue de la detección, la alineación y la normalización.
