---
title: Checkpoints upstream
seo_title: "Cargar checkpoints upstream en LibreYOLO"
description: "Cómo la conversión automática convierte un checkpoint upstream publicado en uno de LibreYOLO v1.0: los formatos que desenvuelve, qué reconoce cada familia y dónde se detiene."
lead: "Las familias de LibreYOLO están portadas desde proyectos upstream cuyos checkpoints publicados son casi cargables, pero no llevan metadatos de LibreYOLO. La conversión automática reconoce esos archivos, los envuelve en el esquema v1.0 y escribe el resultado junto al original."
keywords:
  - libreyolo autoconvert
  - cargar checkpoint upstream
  - convert_upstream_state_dict
  - pesos upstream libreyolo
  - convertir checkpoint yolo
  - cargar pesos .pt en libreyolo
last_verified: "1.5.0"
verification: "Comportamiento leído de libreyolo/models/autoconvert.py y BaseModel.convert_upstream_state_dict; los reconocedores por familia se comprobaron leyendo el override de convert_upstream_state_dict de cada familia, todo en la v1.5.0. Reglas COCO de RF-DETR según docs/checkpoint_schema.md."
snippets:
  usage:
    - label: Basta con pasar el archivo a la factoría
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un archivo upstream reconocido se convierte al cargarlo, y el
        # checkpoint convertido se escribe junto a él.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Cualquier checkpoint de LibreYOLO se carga sin cambios.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
---

## Qué ocurre al cargar

Cuando `LibreYOLO()` se encuentra con un archivo `.pt` que todavía no es un
checkpoint v1.0 completo, llama al conversor automático, que:

1. desenvuelve el diccionario de tensores de los formatos upstream habituales;
2. pregunta a cada familia registrada si reconoce el formato, remapeando las
   claves cuando la nomenclatura upstream difiere de la del port nativo;
3. envuelve a la ganadora en un checkpoint de metadatos estricto v1.0, leyendo
   el tamaño, la tarea y el número de clases de los propios tensores, de modo
   que los checkpoints con fine-tuning se convierten correctamente;
4. lo escribe junto al original como `<source>-<Prefix><size>[-task].pt` y
   devuelve esa ruta, para que la factoría lo cargue con normalidad.

No se le pide nada a quien llama. Un archivo que ninguna familia reclama no
devuelve nada y la factoría informa de que no ha podido cargarlo.

<code-tabs name="usage" />

## Formatos que desenvuelve

El diccionario de tensores se busca en este orden de preferencia, primero el
EMA, y cada candidato se prueba hasta que uno contiene tensores de verdad. Un
bloque EMA vacío o que solo tenga metadatos, por tanto, no oculta los pesos
válidos que hay debajo.

| Clave | Nota |
|---|---|
| `ema.module` | El envoltorio EMA habitual |
| `ema` | Envoltorios EMA planos antiguos que guardan los tensores directamente |
| `ema_state_dict` | Se elimina el prefijo `module.` de las entradas que lo llevan |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| El propio archivo | Un state dict plano |

Cada candidato se reduce después a sus entradas con valor de tensor y se
normaliza: se elimina un prefijo inicial `module.` o `_orig_mod.`, y a un
diccionario cuyas claves empiecen todas por `model.model.` se le quita ese
prefijo.

## Qué reconoce cada familia

El reconocimiento es un classmethod por familia. La implementación por defecto
reclama un formato cuyas claves ya coinciden con las del port nativo. Una
familia cuya nomenclatura upstream sea distinta lo sobrescribe con un remapeo,
y no devuelve nada para los formatos que no reconoce.

Familias que incluyen un reconocedor con remapeo: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Familias que rechazan la conversión automática de plano: `efficientdet`, `eomt`
y `pidnet` no devuelven nada desde el reconocedor, así que sus archivos
upstream pasan por un script de conversión. `l2cs` queda excluida del
reconocedor genérico porque es solo de inferencia y sus pesos tienen
restricciones de redistribución.

RF-DETR mantiene su propio reconocedor, porque necesita el checkpoint entero y
no solo el diccionario de tensores para detectar el tamaño y remapear las
clases de COCO. Solo se registra cuando sus dependencias opcionales están
instaladas.

Todas las demás familias registradas usan el comportamiento por defecto:
reclaman el archivo cuando su propio cargador ya reconoce esas claves.

## Qué familia gana

Varias familias pueden reclamar el mismo archivo, así que la resolución
reproduce las reglas de despacho de la factoría.

La reclamación de una subclase gana a la de su clase base. El orden de registro
sigue al de creación de las clases, así que una familia derivada se registra
después de la base que refina, y sus marcadores positivos no deben perder
frente al passthrough más amplio de la base.

Después decide el orden del registro, porque codifica la especificidad: la
reclamación más temprana es la coincidencia más específica.

El único empate que el orden del registro no puede romper es DEIM contra
D-FINE, cuyas claves de arquitectura son idénticas. Ahí, y solo ahí, el nombre
del archivo es la señal decisiva, y un archivo cuyo nombre no dé ninguna pista
se rechaza en lugar de adivinarlo. El nombre del archivo no se consulta en
ningún otro sitio, deliberadamente, de modo que una reclamación amplia y falsa
positiva nunca pueda imponerse a otra más específica solo por cómo se llame el
archivo.

## Carga segura

Los archivos upstream se cargan con el unpickler de solo pesos. Algunos
checkpoints de entrenamiento upstream incluyen objetos de bibliotecas que ese
unpickler rechaza. Esos objetos son metadatos de entrenamiento y no pesos, así
que cada global bloqueado se reintenta con una clase sustituta inerte que
satisface al unpickler sin ejecutar nada. El nombre capturado se usa solo como
etiqueta de texto: nunca se importa, se evalúa ni se llama.

Los nombres de módulo sensibles se rechazan de plano y nunca se sustituyen:
`builtins`, `os`, `sys`, `posix`, `nt` y `subprocess`. El bucle de reintentos
está limitado a 32 intentos, así que un archivo diseñado para introducir una
serie ilimitada de globals distintos falla de forma segura en lugar de quedarse
girando. Al checkpoint convertido solo llegan los tensores.

## Dónde va el archivo convertido

La salida se escribe junto al original, con el nombre
`<source>-<Prefix><size>[-task].pt`. Siempre se reescribe en lugar de
reutilizarse, lo que mantiene frescas las cargas repetidas del mismo origen y
evita colisiones con los pesos oficiales o con otro fine-tuning de la misma
familia, tamaño y tarea en el mismo directorio.

Cuando el directorio de origen es de solo lectura, la conversión recurre a un
directorio temporal privado y nuevo creado en cada llamada, y la línea de log
indica la ruta que ha usado. Solo si eso también falla se descarta la
conversión, con un aviso.

## Checkpoints de LibreYOLO existentes

Un archivo que lleve un marcador propio de LibreYOLO, `libreyolo_version` o
`model_family`, pertenece a la ruta de carga normal y no se vuelve a convertir.
El salto solo se aplica a una reclamación de tipo passthrough, es decir, a una
en la que el conjunto de claves no cambió. Una reclamación cuya conversión sí
cambió el conjunto de claves es prueba de un formato upstream ajeno y se acepta
incluso en un archivo marcado.

`schema_version` no se trata como marcador, deliberadamente, porque otras
herramientas de entrenamiento y exportación usan ese nombre genérico, y tampoco
lo son `names`, `nc`, `size`, `task` ni `imgsz`, porque un fine-tuning upstream
también puede llevarlos. Un fine-tuning ajeno que solo lleve una clave `names`
genérica no queda marcado, por tanto, así que su reclamación con claves nativas
se convierte con normalidad y deriva el número de clases de la cabeza de
tensores en lugar de cargarse mal como 80 clases.

## Metadatos que lee el conversor

Los nombres de clase se toman de una clave `names` de nivel superior, o de
`class_names` dentro de un bloque `args` o `hyper_parameters`. Un mapa de
nombres indexado por etiquetas en lugar de por índice de clase es inservible y
se sustituye por valores por defecto generados. Una lista de nombres más larga
que el número de clases detectado se recorta, porque los índices fuera de rango
harían fallar al validador estricto y abortarían la conversión en silencio.

Los `args` upstream se arrastran como metadatos simples, descartando cualquier
valor que no sea una cadena, un número, un booleano, una lista o un
diccionario, de modo que nada inseguro llegue al archivo guardado.

## Normalización COCO de RF-DETR

Los checkpoints upstream de RF-DETR exponen una cabeza de clasificación de 91
salidas, que son las 90 clases de COCO más el fondo. La conversión automática
normaliza un RF-DETR de COCO al convenio COCO-80, con el remapeo aplicado en el
postprocesado.

Un checkpoint se trata como COCO cuando lleva exactamente 80 nombres, o declara
un número de clases de 80, o tiene una pista de dataset `coco`, o no tiene
ningún metadato de clases ni de dataset. Este último caso importa: un state
dict upstream pelado es el checkpoint canónico preentrenado en COCO, y es el
único RF-DETR de 91 salidas sin metadatos que hay en distribución.

Un RF-DETR personalizado y genuino de 90 clases se conserva con 90 clases. Se
identifica por una lista de nombres, un número de clases explícito distinto de
80 o una pista de dataset que no sea COCO, de modo que el respaldo del
checkpoint pelado no se dispara con él. Los marcadores de posición vacíos se
ignoran al decidir si hay una pista de dataset.

## Límites

La conversión automática reconoce los formatos upstream publicados. No reescribe
una arquitectura, y no hace cargable un modelo no portado. Cuando ninguna
familia reclama un archivo, la respuesta es un script de conversión y no un
argumento de la factoría: el repositorio incluye `weights/convert_*.py` para
las familias que lo necesitan, entre ellas EoMT, PIDNet y EfficientDet.

La conversión tampoco inventa metadatos que no puede leer. El tamaño, la tarea
y el número de clases salen de los tensores; los nombres salen del archivo
cuando están presentes, y se generan como `class_i` cuando no.
