---
title: Importar pesos existentes
seo_title: Cargar pesos upstream en LibreYOLO
description: >-
  Apunta LibreYOLO a un checkpoint de un proyecto upstream. La autoconversión lo
  reempaqueta al cargarlo y conserva su número de clases y sus nombres.
lead: >-
  LibreYOLO porta sus familias de modelos desde proyectos upstream, así que los
  checkpoints que estos publican ya son casi cargables. Lo que les falta son los
  metadatos. La autoconversión los aporta en el momento de la carga.
keywords:
  - convertir pesos libreyolo
  - cargar checkpoint yolo preentrenado
  - migrar pesos a libreyolo
  - convertir pth a libreyolo
  - autoconversion checkpoint
last_verified: 1.5.0
meta:
  - label: Punto de entrada
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Se escribe junto al origen como
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Conversores en script
    value: weights/ en el repositorio
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Sustituye la ruta por la de un checkpoint que ya tengas. Un layout
        # upstream reconocido se convierte sobre la marcha, se escribe junto
        # al origen y luego se carga.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # El número de clases y los nombres salen de los tensores y de los
        # metadatos del propio archivo, así que un fine-tune conserva su
        # conjunto de etiquetas en lugar del de COCO.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Comprobar el resultado
      language: bash
      code: |
        # El archivo convertido cumple el mismo esquema que uno publicado.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Esta página trata sobre checkpoints de otros proyectos. Si lo que estás moviendo
es tu propio código desde un LibreYOLO más antiguo, consulta
[actualizar a 1.5.0](/docs/upgrade).

## Qué ocurre al cargar un archivo ajeno

`LibreYOLO()` carga cualquier archivo de pesos pasando primero por la ruta
restringida de solo pesos. Si el resultado lleva metadatos completos de
LibreYOLO, se usa directamente. Si no los lleva, el archivo pasa al
autoconversor antes de intentar nada más. Si la carga restringida falla del
todo, algo que ocurre cuando un checkpoint tiene un objeto de terceros
serializado dentro, se prueba el autoconversor con un cargador que neutraliza
esos objetos.

La autoconversión hace cuatro cosas. Desempaqueta el diccionario de tensores del
layout que haya usado el proyecto upstream. Pregunta a cada familia registrada si
reconoce las claves resultantes, remapeando nombres allí donde la nomenclatura
upstream difiere de la del port de LibreYOLO. Envuelve a la ganadora en un
checkpoint que cumple la versión 1.0 del esquema de metadatos, leyendo el tamaño,
la tarea y el número de clases de los propios tensores. Después escribe el
resultado junto al archivo de origen y carga ese.

<code-tabs name="convert" />

La conversión no es silenciosa. Un archivo convertido se registra en el log con
la familia, el nombre de origen, el nombre de salida y el número de clases
resultante, de modo que el log de una ejecución deja constancia exacta de qué se
cargó.

## Los layouts que desempaqueta

Los checkpoints upstream anidan sus pesos en un puñado de sitios convencionales,
y el conversor los prueba en orden hasta que uno contiene tensores: un bloque EMA
bajo `ema.module` o un `ema` plano, un `ema_state_dict` con su prefijo `module.`
eliminado, después `params_ema`, `params`, `ema_net`, `net`, `model`,
`state_dict` y, por último, el objeto en sí. Probar varios en lugar de solo el
primero implica que un bloque `ema` que solo contenga contadores no oculte los
pesos reales que hay debajo.

Los prefijos de envoltura también se quitan: `module.` del entrenamiento
distribuido, `_orig_mod.` de un modelo compilado y un anidamiento `model.model.`
que añaden algunas redistribuciones.

## Qué lee, y de dónde

El tamaño, la tarea y el número de clases salen de los tensores, no del nombre
del archivo, que es la razón por la que un checkpoint con fine-tuning se
convierte con su propio número de clases en lugar de con el valor por defecto de
la arquitectura. Los nombres de las clases se toman de los metadatos del propio
checkpoint cuando están presentes, de un bloque `args` o `hyper_parameters` si
los nombres están ahí, y se recortan al número de clases detectado para que un
fine-tune que haya conservado su conjunto de etiquetas base no arrastre índices
que su cabeza ya no tiene.

Las tareas densas se tratan de forma explícita en lugar de asignarles etiquetas
inventadas. Un checkpoint de profundidad recibe una clase llamada `depth`; uno de
restauración, una clase llamada `image`. Un checkpoint de pose tiene que aportar
un número de keypoints, ya sea desde los tensores o desde la familia; si ninguno
de los dos lo produce, se rechaza la conversión en lugar de escribir un archivo
incompleto.

RF-DETR tiene su propio reconocedor, porque la detección del tamaño necesita el
checkpoint entero y porque su cabeza tiene 91 salidas donde LibreYOLO usa la
convención COCO de 80 clases. Un checkpoint se normaliza a 80 clases cuando lleva
exactamente 80 nombres, o declara un número de clases de 80, o menciona COCO como
su dataset, o no lleva ningún metadato de clases ni de dataset. Un modelo de 90
clases auténtico, identificado por sus nombres, por un número explícito distinto
de 80 o por una pista de dataset que no sea COCO, se conserva tal cual.

## Dónde va el archivo convertido

La salida se escribe junto al origen, con el nombre tomado de él:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Un detector YOLOv9 tiny guardado como `upstream-checkpoint.pth` pasa a ser, por
tanto, `upstream-checkpoint-LibreYOLO9t.pt`. Nombrarlo según el origen en lugar
de según la familia implica que dos fine-tunes de la misma familia y el mismo
tamaño en un mismo directorio no se sobrescriben entre sí, y que ninguno choca
con un checkpoint oficial. El archivo se reescribe en cada carga, así que nunca
queda desfasado respecto a su origen. Si el directorio es de solo lectura, el
archivo convertido va a un directorio temporal privado y recién creado, y el log
dice dónde.

A partir de ahí es un checkpoint de LibreYOLO corriente: se carga por la ruta de
metadatos, y `libreyolo metadata` lo da por válido.

## Casos que necesitan ayuda

Dos familias quedan fuera del reconocedor genérico. La familia de mirada (gaze)
está excluida por completo: es solo de inferencia y los pesos que publica llevan
restricciones de redistribución. RF-DETR está excluida porque tiene el
reconocedor dedicado descrito más arriba, que es lo que se encarga de ella en su
lugar.

Los checkpoints PIDNet upstream en crudo se rechazan, con un error que apunta a
`weights/convert_pidnet_weights.py`. Ese script escribe los metadatos semánticos
de Cityscapes que el checkpoint necesita.

D-FINE y DEIM comparten las mismas claves de arquitectura, así que los tensores
por sí solos no permiten distinguirlos. Cuando ambas reclaman un archivo y no hay
en juego ninguna familia hermana con un marcador diferenciador, decide el nombre
del archivo: un nombre con la forma de `dfine_hgnetv2_n_coco.pth` o
`deim_hgnetv2_n_coco.pth` lo zanja, y un nombre que no dice nada se rechaza con
esa explicación en lugar de adivinarlo. Instanciar `LibreDFINE` o `LibreDEIM`
directamente también lo resuelve.

Cuando varias familias reclaman legítimamente un mismo archivo, una subclase gana
a la clase base que refina, y el orden del registro decide el resto, ya que ese
orden codifica cuán específica es la comprobación de cada familia. El nombre del
archivo solo se consulta para el empate entre D-FINE y DEIM, de modo que el
nombre de un archivo nunca puede anteponer una coincidencia amplia a una precisa.

## Los conversores en script

El repositorio incluye scripts de conversión por familia bajo `weights/`, además
de helpers compartidos para la fontanería repetida. Son la vía para un archivo
que la ruta en tiempo de ejecución rechaza, para producir un checkpoint por
adelantado en lugar de en el momento de la carga, y para las familias cuyos
metadatos hay que aportar en lugar de inferirlos de los tensores.

Esos scripts forman parte del repositorio, no del paquete instalado, así que usar
uno implica clonarlo:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Todos los scripts escriben un checkpoint que cumple la versión 1.0 del esquema,
que es el mismo listón que alcanza la autoconversión y el mismo que alcanzan los
pesos publicados. Consulta [checkpoints y pesos](/docs/weights) para saber qué
contiene ese esquema.
