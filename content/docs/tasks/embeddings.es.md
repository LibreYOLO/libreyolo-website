---
title: Embeddings
seo_title: "Embeddings de imagen y de región en LibreYOLO"
description: "La tarea embed devuelve vectores float32 normalizados con L2 para una imagen entera, para cada región detectada o para texto. Registra una galería, empareja por similitud del coseno y busca desde Python o la CLI."
lead: "Una sola tarea cubre todos los vectores que produce LibreYOLO. embed devuelve filas float32 de longitud unitaria cuyo producto escalar es una puntuación de similitud, tanto si la fila describe una imagen entera como una sola cara detectada o una línea de texto, y la misma Gallery las empareja todas."
keywords: [embeddings de imagen python, embedding normalizado l2, búsqueda por similitud del coseno, tarea embed libreyolo, buscar imágenes similares, registrar galería de caras, embeddings clip, embeddings dinov2, embeddings reid]
last_verified: "1.5.0"
verification: "Clave de tarea y alias leídos de libreyolo/tasks.py. Payloads de resultado de las clases Embeddings e Identities en libreyolo/utils/results.py. API de Gallery de libreyolo/utils/gallery.py. embed y _postprocess_embeddings de libreyolo/models/base/model.py. Familias soportadas localizadas buscando embed en SUPPORTED_TASKS dentro de libreyolo/models/**/model.py. Superficie de CLI de libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py y libreyolo/cli/commands/predict.py. Intención de diseño de docs/adr/0015-embed-generalization.md."
meta:
  - label: Clave de tarea
    value: embed
    mono: true
  - label: Alias
    value: face-recognition, reid, face
    mono: true
  - label: Payloads de resultado
    value: Embeddings, Identities
    mono: true
  - label: Dtype de fila
    value: float32, longitud unitaria
snippets:
  predict:
    - label: Imagen entera
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP usa classify por defecto, así que pide el vector explícitamente.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), una fila por imagen
        print(result.boxes)                  # None: no se localizó nada
    - label: Por región
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # La fila i describe la región del box i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Varias imágenes a la vez
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Todas las filas de todos los resultados, en un solo tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Texto
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # El texto es un método, nunca una fuente de predicción. Una cadena
        # pasada a model(...) sigue siendo una ruta o una URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Comparar dos conjuntos de filas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Las filas son unitarias: la similitud del coseno es un producto escalar.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Imagen contra texto
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Registrar e identificar
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name es None por debajo del umbral
    - label: Búsqueda top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] de la primera fila
    - label: Registrar un vector que ya tienes
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # se normaliza al entrar
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Registrar un árbol de carpetas
      language: bash
      code: |
        # source/<identity>/*.jpg. Una galería existente se amplía in situ.
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=refs.npz
    - label: Identificar durante la predicción
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Comparar dos imágenes
      language: bash
      code: |
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify es el mismo comando bajo un segundo nombre.
        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg --json
---

## Definición

`embed` convierte una imagen, una región de una imagen o una cadena en una fila
float32 de ancho fijo cuya longitud es uno. Como cada fila es un vector
unitario, comparar dos de ellas es un producto escalar, y comparar dos conjuntos
es una sola multiplicación de matrices. Nada más en la tarea depende del modelo:
la recuperación, la detección de duplicados, la reidentificación y el
reconocimiento facial son todos la misma aritmética sobre filas distintas.

El vector es la salida. No hay lista de clases, así que el nombre se asigna
después comparando contra las referencias que tú proporcionas, no por nada que
la red se haya entrenado para predecir.

### Tres formas

| Forma | `Results.embeddings` | `Results.boxes` | Producido por |
|---|---|---|---|
| Imagen entera | `(1, D)` | `None` | Pasar una imagen a una familia de imagen entera |
| Región | `(N, D)` | `(N, 4)`, alineado fila a fila | Familias que localizan primero, como el reconocimiento facial |
| Texto | ni siquiera es un `Results` | | `model.embed_text(texts)`, que devuelve `(M, D)` |

Un resultado de imagen entera sigue siendo bidimensional incluso con una sola
imagen. `(D,)` no es una forma de retorno permitida, así que quien lo consume
nunca tiene que tratar la fila única como un caso especial. El texto devuelve un
tensor normal en lugar de un `Results`, porque una cadena no es una fuente de
imagen: pasar una a `model(...)` sigue significando una ruta o una URL, y la
biblioteca nunca supone que una cadena es prosa.

La clave canónica de la tarea es `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` y
`reid` se normalizan todas a ella, así que `task="reid"` y `task="embed"`
seleccionan exactamente lo mismo.

## Modelos

Cuatro familias cubren la tarea, y se dividen con claridad según si localizan
algo primero o no.

| Familia | Forma | Dimensión | También soporta |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Región, una fila por cara detectada | 512 | Nada; `embed` es su única tarea |
| [CLIP](/docs/models/clip) | Imagen entera, con torre de texto emparejada | 512 para `b32` y `b16`, 768 para `l14` | `classify`, que sigue siendo su tarea por defecto |
| [SigLIP 2](/docs/models/siglip2) | Imagen entera, con torre de texto emparejada | 768 para `b16`, 1152 para `so400m` | `classify`, que sigue siendo su tarea por defecto |
| [DINOv2](/docs/models/dinov2) | Imagen entera, solo imagen | 384 | `semantic`, `classify` |

CLIP y SigLIP 2 mantienen `classify` como tarea por defecto, así que hay que
pedir `task="embed"` de forma explícita. Su checkpoint `-cls` existente es el
artefacto compartido de dos torres; no se publica un checkpoint `-embed`
duplicado para los mismos pesos.

`embed_text` solo existe en CLIP y SigLIP 2, las dos familias con torre de
texto. DINOv2 no tiene ninguna. El embedding de DINOv2 se salta las cabezas
semántica y de clasificación y lee el token CLS final normalizado, a 224 píxeles;
las variantes `n`, `s`, `m` y `l` comparten todas el encoder DINOv2-S, así que
las cuatro devuelven `D = 384`.

Los backbones solo de clasificación añadidos en esta versión,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) y [DeiT](/docs/models/deit),
declaran únicamente `classify` y no cubren esta tarea.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` es el atajo por lotes: ejecuta `predict` y
concatena todas las filas de todos los resultados en un único tensor float32 de
CPU `(N_total, D)`, lanzando una excepción si las filas tienen dimensiones
distintas. Una familia que no tenga `embed` entre sus tareas soportadas lanza
`NotImplementedError`.

## Payloads de resultado

`result.embeddings` es un payload `Embeddings`. Su `data` es siempre `(N, D)`
float32, ya normalizado con L2 por la ruta de inferencia, y una entrada que no
sea bidimensional lanza una excepción en lugar de cambiar de forma en silencio.

| Miembro | Significado |
|---|---|
| `.data` | La matriz `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Las mismas filas, renormalizadas por precaución |
| `.similarity(other)` | `(N, M)` contra otro conjunto, o `(N,)` contra un único vector `(D,)` |
| `.verify(i, j, threshold=0.4)` | Si las filas `i` y `j` son el mismo sujeto |

`result.identities` es un payload `Identities`, presente solo cuando se pasó una
galería. Es un contenedor normal, no un tensor, así que mover un `Results` entre
dispositivos lo deja intacto.

| Miembro | Significado |
|---|---|
| `.name` | Lista de nombres, `None` donde nada superó el umbral |
| `.score` | Mejor puntuación de coseno float32 `(N,)`, conservada incluso cuando el nombre es `None` |
| `.data` | Lista de tuplas `(name, score)` |

<code-tabs name="similarity" />

Los vectores se dejan fuera de `summary()` y `to_json()` por defecto, ya que una
fila de 512 floats ocupa unos dos kilobytes por sujeto. Cada fila indica
`embedding_dim` en su lugar, más `identity` e `identity_score` cuando se usó una
galería. Pasa `summary(embeddings=True)` para incluir los números.

## Galerías

Una `Gallery` es un conjunto de filas de referencia con nombre. Guarda cada
referencia por separado en lugar de promediarlas, así que un nombre se puntúa
por su mejor referencia coincidente, y añadir una foto mala no puede arrastrar
el centroide de una identidad.

<code-tabs name="gallery" />

`Gallery(model)` se vincula a los pesos que producirán sus vectores.
`enroll(name, sources, select="best")` ejecuta la predicción sobre cada fuente y
se queda con la fila de mayor confianza por resultado; `select="all"` conserva
en cambio todas las filas, que es lo que quieres cuando una imagen de referencia
contiene legítimamente varios sujetos. `enroll_embedding(name, vector)` se salta
la inferencia y toma un vector directamente, normalizándolo y rechazando una
fila de ceros.

`FaceGallery` es un alias permanente de la misma clase, y los archivos escritos
por versiones anteriores centradas solo en caras siguen cargándose.

### Emparejamiento y umbrales

El emparejamiento es una multiplicación de matrices densa contra todas las
referencias almacenadas, reducida a una puntuación por nombre tomando el máximo.
No hay índice aproximado, lo que mantiene los números exactos y pone un techo
práctico al tamaño de la galería.

Dos puntos de entrada difieren en lo que hacen por debajo del umbral. `match()`
devuelve `[(name, score), ...]` por fila descartando todo lo que quede por
debajo del umbral, así que una fila sin coincidencias es una lista vacía.
`identify()` devuelve un payload `Identities` que siempre conserva la mejor
puntuación y pone el nombre a `None` cuando está por debajo del umbral. Ninguno
sustituye nunca por el nombre más cercano que no llegue al umbral.

El umbral por defecto es `0.4` en todas partes. Es un valor de coseno, no una
probabilidad, y el punto de operación correcto es una propiedad de tus datos y
de tu tolerancia a coincidencias falsas, así que haz un barrido sobre pares
etiquetados en lugar de aceptar el valor por defecto. `libreyolo enroll` y el argumento de
predicción `gallery=` usan el mismo número.

### Persistencia

`save(path)` escribe un `.npz` comprimido que contiene los vectores, los nombres
y un bloque de metadatos con la versión del formato, la dimensión del embedding
y una huella de los pesos que produjeron las filas. `Gallery.load(path,
model=...)` comprueba ambos antes de comparar nada, así que apuntar una galería
a un modelo distinto lanza una excepción en lugar de puntuar en silencio
vectores de dos espacios sin relación entre sí. Guardar una galería vacía se
rechaza.

## Línea de comandos

| Comando | Propósito |
|---|---|
| `libreyolo enroll` | Recorre un árbol de una carpeta por identidad y escribe o amplía una galería `.npz` |
| `libreyolo compare` | Genera el embedding del sujeto principal en dos imágenes e informa de la similitud del coseno |
| `libreyolo verify` | El mismo comando bajo un segundo nombre |
| `libreyolo predict gallery=...` | Adjunta identidades a una ejecución de predicción normal |

<code-tabs name="cli" />

Todos los comandos de LibreYOLO aceptan tanto `key=value` como `--key value`,
así que `gallery=refs.npz` y `--gallery refs.npz` son el mismo argumento.

`enroll` toma `model`, `source` y `gallery`, más los opcionales
`face-detector`, `device`, `--json` y `--quiet`. Lee una carpeta por identidad,
donde el nombre de la carpeta es la identidad y todas las imágenes de dentro
aportan referencias:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Una imagen que no produce nada se omite con una línea en stderr en lugar de
abortar la ejecución, y el resumen informa de cuántas referencias se guardaron
para cada nombre. Un archivo de galería existente se amplía in situ, así que se
pueden ir añadiendo identidades con el tiempo.

`compare` y `verify` son una misma función registrada dos veces. Toman `model`,
`source`, `source2` y un `threshold` opcional, e imprimen la similitud del
coseno, el veredicto de igual-o-distinto y el umbral que lo produjo. `--json`
imprime los mismos tres campos como un objeto.

En `predict`, `gallery` apunta a un `.npz` guardado y `gallery_threshold`
anula el valor por defecto de `0.4`. Pasar una galería a un modelo cuya
tarea no es `embed` es un error en lugar de una operación silenciosa sin efecto,
y un archivo de galería que falta sugiere el comando `libreyolo enroll` que lo
crearía.

## Caras

El reconocimiento facial es la forma de región de esta tarea, y es la única
implementación de esa forma que se distribuye. Añade una etapa de detección y
alineación delante de la cabeza de embedding, más un método `verify()`, un
argumento para traer tus propios boxes, cifras de precisión publicadas y guía de
calibración del umbral. Todo eso vive en
[reconocimiento facial](/docs/tasks/face-recognition), que es el recorrido a
seguir cuando el sujeto son caras. Todo lo de esta página se le aplica sin
cambios.

## Entrenar, validar y exportar

Nada en esta tarea se entrena dentro de LibreYOLO. La cabeza de embedding facial
es un artefacto ONNX cuyos `train()`, `val()` y `export()` lanzan excepción;
entrena una cabeza aguas arriba y carga el archivo por ruta. CLIP, SigLIP 2 y
DINOv2 entrenan y exportan a través de sus tareas de clasificación y
segmentación, no a través de `embed`.

No hay validador de recuperación. Mide la precisión de verificación sobre pares
etiquetados barriendo `threshold`, y la precisión de identificación registrando
una galería y leyendo `identities.name` e `identities.score` sobre imágenes
reservadas, contando un nombre `None` como un rechazo.
