---
title: Niveles de estabilidad
seo_title: "Qué significa cada nivel de soporte de LibreYOLO"
description: "El vocabulario de niveles que usa LibreYOLO: los tres niveles de soporte de exportación, los cuatro niveles de API, los seis grupos de cobertura y lo que ninguno de ellos promete."
lead: "LibreYOLO usa la palabra nivel para tres cosas distintas: la evidencia que hay detrás de una ruta de exportación, el contrato de llamada al que responde una familia de modelos, y el grupo de cobertura en el que está inscrita esa familia. Esta página define cada uno y dice qué no implica."
keywords:
  - nivel de soporte libreyolo
  - validated available blocked
  - niveles de exportación libreyolo
  - grupos de cobertura libreyolo
  - g0 g1 g2 g3 g4
  - niveles de modelos libreyolo
last_verified: "1.5.0"
verification: "Niveles de exportación de docs/adr/0011-export-support-tiers.md y libreyolo/export/support.py; grupos de cobertura y recuentos por familia de MODEL_GROUPS en libreyolo/models/registry.py; la comprobación de entrenamiento desde cero de libreyolo/models/base/model.py y libreyolo/cli/commands/train.py; el inventario de la CLI leído de libreyolo/models/inventory.py; niveles de API de los docstrings de los paquetes libreyolo/models/sam/, openvocab/ y vlm/ y de los contratos de base.py, todo en la v1.5.0. Las etiquetas de grupo que ve el lector (Flagship, Core, Supported, Inference only, Museum, Sibling tier) son el vocabulario propio del sitio para esos mismos grupos, tomado de src/data/docs/registry.json."
snippets:
  usage:
    - label: Leer las dos clasificaciones de una familia
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
---

## Niveles de soporte de exportación

El nivel que decide si una llamada tiene éxito. Se aplica a la terna
`(family, task, format)`, y cada combinación tiene exactamente uno.

| Nivel | Significado | Qué ocurre al llamar a `export()` |
|---|---|---|
| `validated` | La paridad numérica está cubierta en CI o en una ejecución nocturna documentada | Se ejecuta |
| `available` | La conversión está implementada, pero no se ha registrado evidencia de paridad numérica en runtime | Se ejecuta |
| `blocked` | No hay ninguna ruta soportada | Lanza `NotImplementedError` en el preflight, con el motivo |

Validated y available continúan las dos sin pedir una confirmación ni emitir un
aviso genérico. La diferencia es la evidencia, no el permiso: una entrada
validated tiene detrás un test de paridad y una versión `since`, y una entrada
available todavía no. Una conversión a CoreML sin una ejecución de predicción en
macOS, por ejemplo, es available y no validated.

Una combinación blocked falla antes de las comprobaciones de dependencias, de la
carga de calibración, del trazado o de la creación de artefactos, así que no se
escribe nada parcial.

Cada celda validated lleva una restricción que describe la configuración de la
que salió el número de paridad, normalmente un lienzo de entrada fijo, batch 1,
FP32 y una versión concreta del runtime. Léela como una afirmación sobre esa
configuración y no sobre el formato en general. Las reglas que rellenan las
celdas sin entrada explícita están en la página de la
[matriz de exportación](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Niveles de API

El nivel que decide qué aspecto tiene una llamada. Una familia está en
exactamente uno, elegido por contrato de llamada y no por arquitectura.

| Nivel | Factoría | Contrato |
|---|---|---|
| Factoría de detectores | `LibreYOLO` | Un único forward sin prompt devuelve todos los objetos que ha encontrado, con puntuaciones calibradas. Los miembros se registran solos reconociendo un checkpoint |
| Segmentación con prompts | `LibreSAM` | Un forward no significa nada sin un prompt espacial o de concepto por imagen, suministrado en el momento de la llamada. Interactivo y con estado: codifica una vez, haz prompts muchas veces |
| Detección de vocabulario abierto | `LibreOpenVocab` | Detectores discriminativos condicionados por texto. La lista de clases es un prompt, que se fija con `set_classes` |
| Visión-lenguaje | `LibreVLM` | Un modelo generativo manejado como un detector. La lista de clases es un prompt y la confianza es un valor de relleno |

Los tres niveles hermanos deliberadamente no se registran en la factoría de
detectores, y por eso `LibreYOLO("some-alias")` no llega a ellos. Se cargan por
alias de tamaño y descarga automática, no por inspección del checkpoint.

Los cuatro devuelven el mismo `Results`, así que el código que va después no
cambia entre unos y otros. Lo que difiere es qué métodos funcionan: los niveles
hermanos lanzan `NotImplementedError` en `train()`, `val()` y `export()`, y los
niveles de SAM y de vocabulario abierto lo lanzan también en `track()`. La
página de cada nivel enumera sus propias exclusiones.

## Grupos de cobertura

La clasificación que decide qué familias entran en una ejecución de tests entre
familias, y la que un lector se encontrará con más probabilidad en una página de
modelo. Cada familia registrada está inscrita en exactamente un grupo, y un test
falla cuando una familia registrada no aparece en esa inscripción. `GROUPS`, en
`libreyolo/models/registry.py`, es la fuente de la columna Significado de abajo;
`MODEL_GROUPS`, en el mismo archivo, asigna cada familia, y la columna Familias
cuenta esa asignación directamente. La columna Etiqueta es el nombre corto que
el sitio usa para ese mismo grupo en la cabecera de una página de modelo.

| Grupo | Etiqueta | Familias | Significado |
|---|---|---|---|
| `g0` | Flagship | 2 | Anclas insignia obligatorias en la cobertura de funcionalidades compartidas |
| `g1` | Core | 10 | Conjunto de cobertura de detectores entrenables |
| `g2` | Supported | 14 | Conjunto de cobertura adicional de familias entrenables |
| `g3` | Inference only | 35 | Familias sin implementación de entrenamiento |
| `g4` | Museum | 5 | Familias históricas con cobertura de inferencia |
| `s` | Sibling tier | 21 | APIs hermanas (SAM, vocabulario abierto, VLM, zero-shot) cubiertas por separado |

Son 87 familias repartidas en seis grupos. `g3` por sí solo reúne más familias
que todos los demás grupos juntos, porque la mayor parte del registro es linaje
de solo inferencia y cobertura de museo, y no detectores entrenados activamente.

Para un lector que está eligiendo modelo, el grupo dice dónde esperar atención
de ingeniería, no lo precisa que es una familia. `g0` y `g1` son donde se diseña
una funcionalidad nueva y donde llega primero; `g2` se mantiene en verde en CI,
pero una funcionalidad llega ahí de forma oportunista y no en la misma oleada de
versión. `g3` declara una ausencia, no un límite: predecir, validar y, cuando la
familia lo admite, exportar siguen funcionando, y `train()` sobre una familia
`g3` o `g4` lanza `NotImplementedError` nombrando el motivo en lugar de hacer
algo parcial en silencio. Las familias `s` no participan en absoluto en este
equilibrio, porque se cargan a través de su propia factoría y no de
`LibreYOLO()`. Consulta [conceptos básicos](/docs/concepts) para ver cómo encaja
un grupo junto a la tarea, la familia y el tamaño al leer el nombre de archivo
de un checkpoint.

Un grupo no concede ni restringe por sí solo ninguna capacidad de cara al
usuario. El soporte viene de la API implementada de la familia y de las
comprobaciones de capacidad específicas de cada formato, nunca de la pertenencia
a un grupo por sí sola. Los grupos clasifican familias, no tareas, así que una
ejecución de cobertura acotada a una tarea nombra la tarea de forma explícita,
como en «g1 detect».

Hay dos sitios que leen el grupo en tiempo de ejecución y no solo en los tests.
`collect_model_inventory()`, en `libreyolo/models/inventory.py`, adjunta el
grupo a cada entrada que imprime el inventario de la CLI, y `pretrained=False`
activa la ruta especial de reinicialización desde cero solo para las familias de
`g0` y `g1`. Fuera de esos dos grupos la comprobación de
`libreyolo/models/base/model.py` se omite por completo, así que
`pretrained=False` llega al `train()` propio de la familia como un argumento por
nombre cualquiera.

## Entrenamiento

Una familia de `g3` o `g4` no tiene implementación de entrenamiento, y llamar a
`train()` sobre una de ellas lanza una excepción. Eso es una propiedad del
código de la familia, no de su grupo: el grupo registra el hecho, no lo provoca.

Para una familia que sí entrena, si un parámetro concreto de aumento de datos
llega al pipeline es una cuestión aparte, con su propio vocabulario de tres
valores: `used`, `gated_by_mosaic` e `ignored`. Consulta la
[matriz de aumento de datos](/docs/reference/augmentation-matrix).

## Qué no te dice un nivel

Un nivel no es una afirmación sobre la precisión. Una exportación validated dice
que el artefacto reproduce el modelo nativo dentro de un umbral declarado; no
dice nada sobre qué puntuación saca el modelo nativo en un dataset. Los números
de benchmark están en las páginas de cada modelo.

Un nivel tampoco es una declaración de licencia. Las licencias de los pesos
varían dentro de una misma familia y el repositorio que aloja un checkpoint
concreto es el que manda. Que una familia esté en la factoría de detectores no
dice nada sobre si sus pesos publicados permiten el uso comercial.
