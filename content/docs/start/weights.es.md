---
title: Checkpoints y pesos
seo_title: "Checkpoints y pesos de LibreYOLO"
description: "Cómo LibreYOLO encuentra, descarga y verifica los pesos de los modelos, dónde están alojados, cómo ejecutar sin red y qué hace que un checkpoint se cargue de forma segura."
lead: "Un checkpoint de LibreYOLO es un diccionario de torch.save que contiene un state dict más los metadatos necesarios para identificarlo. Esta página cubre de dónde vienen esos archivos, dónde acaban y cómo se cargan."
keywords: [pesos libreyolo, checkpoints libreyolo, descargar pesos libreyolo, libreyolo sin conexión, libreyolo hugging face, metadatos de un checkpoint]
last_verified: "1.5.0"
meta:
  - label: Alojados en
    value: "Un repositorio de Hugging Face por checkpoint:"
    links:
      - label: huggingface.co/LibreYOLO
        href: https://huggingface.co/LibreYOLO
  - label: Caché local
    value: weights/ bajo el directorio de trabajo
    mono: true
  - label: Esquema de metadatos
    value: v1.0
snippets:
  load:
    - label: Descarga automática
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Un nombre de archivo a secas se resuelve a weights/LibreYOLO9t.pt y
        # se descarga ahí si no está ya presente.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Ruta explícita
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Una ruta con componente de directorio se usa exactamente tal cual
        # está escrita y nunca se descarga de la red.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Lee los metadatos sin construir un modelo, e informa de si
        # cumplen el esquema.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: |
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )

        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")

        # Devuelve una lista de problemas. Vacía significa que el archivo cumple la v1.0.
        print(validate_checkpoint_metadata(loaded))
        print(loaded["model_family"], loaded["size"], loaded["task"], loaded["nc"])
---

## Dónde se busca un checkpoint

Una referencia de modelo sin componente de directorio, como `LibreYOLO9t.pt`, se
resuelve contra `weights/` relativo al directorio de trabajo actual. Si
`weights/LibreYOLO9t.pt` existe, se usa; si existe un archivo con ese nombre en
el propio directorio de trabajo, se usa ese en su lugar; en caso contrario,
`weights/LibreYOLO9t.pt` pasa a ser el destino de la descarga.

Una referencia que sí contiene un directorio, absoluto o relativo, se toma
literalmente. Esa es la forma que hay que usar cuando los pesos viven en un
lugar centralizado y no se debe descargar nada.

<code-tabs name="load" />

## Descarga automática

Cuando la ruta resuelta no existe, LibreYOLO analiza el nombre del archivo para
recuperar la familia, el tamaño y la tarea, y le pide a la familia
correspondiente una URL de descarga. La mayoría de familias la construyen a
partir de la organización LibreYOLO en Hugging Face, donde cada checkpoint tiene
su propio repositorio con el nombre del archivo:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Un sufijo de variante de dataset sigue formando parte del nombre del
repositorio, así que un checkpoint entrenado con algo distinto del dataset por
defecto de la familia se resuelve a su propio repositorio en lugar de
sobrescribir el de por defecto.

La transferencia en sí es defensiva, porque un archivo de pesos truncado falla
más tarde con un error poco útil. Las descargas se transmiten a un archivo
`.part` y se mueven a su sitio de forma atómica solo cuando están completas, de
modo que un proceso interrumpido nunca puede dejar un checkpoint a medio escribir
en la ruta final. Una transferencia interrumpida se reanuda desde su offset de
bytes usando un validador HTTP, y reinicia desde cero si el servidor indica que
el objeto ha cambiado. Los fallos se reintentan tres veces con backoff
exponencial. Los procesos concurrentes que apuntan a la misma ruta toman un
archivo de bloqueo, así que dos entrenamientos que arrancan a la vez descargan
una sola vez. Cuando una familia descarga de un host de terceros en lugar de la
organización LibreYOLO, puede fijar un checksum y rechazar el archivo si no
coincide.

Si `HF_TOKEN` está definida, o hay un token cacheado en
`~/.cache/huggingface/token`, se adjunta como bearer token. Solo se adjunta a
URLs de `huggingface.co`, así que una familia que descargue de otro host nunca
lo recibe.

No todas las familias descargan automáticamente. Algunas devuelven
deliberadamente ninguna URL porque los pesos publicados no se pueden
redistribuir, y el error explica entonces qué hay que proporcionar en su lugar.
Otras imprimen un aviso de licencia antes de que empiece la transferencia. Ese
aviso es la señal en tiempo de ejecución de que los términos de un checkpoint
son más restrictivos que los del código, y merece la pena leerlo en vez de
pasarlo de largo.

## La organización de Hugging Face

Los pesos publicados viven en
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), un repositorio por
checkpoint. Cada repositorio lleva una licencia, y la licencia no es uniforme
dentro de una familia: una familia cuyo código es MIT puede tener algunos pesos
que no lo son. El repositorio es la fuente autoritativa. Cada página de modelo
lista los checkpoints publicados de esa familia y sus licencias en sus secciones
de Checkpoints y Licencias.

## Trabajar sin conexión

Nada en la biblioteca requiere acceso a la red una vez que los archivos están en
local. Funcionan dos enfoques:

Prepoblar un directorio `weights/` junto a donde se ejecute el trabajo.
Descargar los checkpoints una vez en una máquina conectada y luego copiar el
directorio es suficiente; el paso de resolución de arriba los encuentra y nunca
llega a la red.

O pasar una ruta absoluta a una ubicación compartida. Una referencia con
componente de directorio se usa tal cual, así que un montaje de solo lectura con
pesos curados es una configuración válida. Si el proceso no puede escribir junto
a un checkpoint que necesita convertir, la conversión recurre a un directorio
temporal privado en lugar de fallar.

Los datasets siguen una regla aparte: se resuelven bajo `~/datasets`, o bajo el
directorio indicado por `LIBREYOLO_DATASETS_DIR` cuando esa variable está
definida.

## Seguridad en la carga

Los checkpoints son pickles, y un pickle puede ejecutar código arbitrario al
abrirse. LibreYOLO trata cada archivo de pesos como no confiable y lo carga por
la vía `weights_only=True` de PyTorch, que restringe el unpickler a tensores y a
un pequeño conjunto de tipos seguros. Esto se aplica al archivo que tú pasas, no
solo a los archivos que LibreYOLO descargó. En una build de PyTorch demasiado
antigua para soportar ese argumento, la carga se rechaza en vez de realizarse de
forma insegura.

Algunos checkpoints de entrenamiento de terceros incrustan objetos que el
unpickler restringido rechaza, como un objeto de configuración del framework con
el que se entrenaron. Esos objetos son metadatos que LibreYOLO no necesita, así
que durante la conversión cada clase bloqueada se sustituye por un sustituto
inerte que satisface al unpickler sin ejecutar nada, y solo los tensores
sobreviven al archivo convertido. Los nombres de módulo sensibles se rechazan
directamente en lugar de sustituirse, y el bucle de reintentos está acotado para
que un archivo diseñado para introducir una serie interminable de clases
bloqueadas falle en cerrado. Consulta
[importar pesos existentes](/docs/migrate) para el resto de esa vía.

## Metadatos del checkpoint

Un checkpoint de LibreYOLO es un diccionario cuya clave `model` contiene el state
dict de PyTorch. El esquema v1.0 exige nueve claves, y juntas permiten que la
factoría identifique un archivo sin analizar su nombre ni adivinar a partir de
las formas de los tensores.

| Clave | Significado |
|---|---|
| `model` | El state dict de PyTorch |
| `schema_version` | La versión del contrato de metadatos. La v1.0 usa la cadena `1.0` |
| `libreyolo_version` | La versión de LibreYOLO que produjo el archivo |
| `model_family` | Un identificador de familia registrado, como `yolo9` |
| `size` | La variante dentro de esa familia, como `t` o `r18` |
| `task` | Un nombre de tarea canónico |
| `nc` | Un número de clases positivo |
| `names` | Un mapeo de índice de clase a etiqueta, que cubre de `0` a `nc - 1` |
| `imgsz` | Una resolución de entrada positiva |

Las tareas con estructura adicional la registran junto a esas claves. Los
checkpoints de pose añaden `num_keypoints` y `keypoint_dim`, y pueden añadir
sigmas OKS por keypoint. Los checkpoints de OCR incrustan el charset CTC completo
para que el archivo sea autocontenido. Los checkpoints de restauración pueden
registrar el tipo de degradación y un factor de escalado. Los checkpoints del
entrenador añaden estado de reanudación como `epoch`, el estado del optimizador y
los pesos EMA; los pesos de inferencia publicados no deberían llevar eso.

Un archivo que cumple las nueve claves se carga por la vía de metadatos. Uno que
no las cumple o bien se convierte, si alguna familia reconoce su estructura, o
bien se carga por la vía de compatibilidad con un aviso que indica qué falta.

## Inspeccionar un checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` nunca construye un modelo, así que funciona con un archivo
cuya familia no está instalada y con un archivo del que no estás seguro.
