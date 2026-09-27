---
title: Congelación de capas
seo_title: Congelar capas durante el entrenamiento en LibreYOLO
description: >-
  Congela parte de un modelo para transfer learning: un número entero de grupos
  de congelación de la familia, una lista explícita de índices, o selectores por
  nombre de módulo y de parámetro.
lead: >-
  Congelar mantiene fijos los pesos seleccionados mientras el resto del modelo
  entrena. Los selectores apuntan a los grupos de congelación ordenados de cada
  familia o a sus nombres de módulo, no a números de capa sueltos de un grafo
  YAML.
keywords:
  - congelar capas yolo
  - transfer learning yolo
  - congelar backbone
  - batchnorm congelado
  - grupos de congelacion
  - entrenar solo la cabeza
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Los diez primeros grupos son el backbone completo de YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Por nombre
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Varios selectores
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Listar en orden los grupos de congelación de una familia
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Congela algo

`freeze` es opcional y por defecto no congela nada.

<code-tabs name="train" />

La congelación se aplica después de construir el modelo y después de cualquier
reconstrucción de la cabeza para un nuevo número de clases, y antes de crear el
optimizador, de modo que el optimizador solo recibe parámetros entrenables.

## Qué puede ser un selector

| Valor | Significado |
|---|---|
| `None`, `False`, `""`, `"none"` | Entrena todos los parámetros |
| `10` o `"10"` | Congela los diez primeros grupos de congelación de la familia |
| `[0, 3, 7]` | Congela esos grupos, con índice desde cero |
| `"backbone"` | Congela el grupo, módulo o prefijo de parámetro que coincida |
| `["backbone", "neck"]` | Congela cada selector de la lista |
| `["backbone", 3]` | Las listas mixtas funcionan |

Una cadena se parsea antes de interpretarse, así que la CLI y una configuración
YAML aceptan las mismas formas que Python. `freeze="[0, 3, 'head']"` se parsea
como una lista literal, `freeze="backbone,neck"` se divide por la coma, y una
cadena decimal a secas se convierte en un recuento.

`freeze=True` se rechaza por ambiguo.

Los selectores por nombre coinciden con el nombre de un grupo de congelación, con
el nombre de un módulo o con un prefijo de nombre de parámetro, y los caracteres
glob `*`, `?` y `[` funcionan. Un `model.` inicial se trata con flexibilidad, así
que tanto `backbone` como `model.backbone` coinciden con la forma que la familia
use internamente.

## Los grupos los define cada familia

Un entero apunta a la lista ordenada de grupos de congelación propia de cada
familia, no a una posición dentro de un grafo compartido. Las familias de
LibreYOLO no son todas un mismo modelo secuencial indexado por YAML, así que un
número de capa suelto significaría algo distinto en cada una.

YOLOv9 ordena sus grupos desde el lado de la entrada: diez etapas de backbone,
luego seis etapas de neck, y luego la cabeza. Por eso `freeze=10` es exactamente
el backbone. `backbone`, `neck` y `head` son selectores por nombre estables
encima de eso.

Los grupos de RF-DETR son `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` y `head`. Aquí los nombres son la mejor
opción, porque los componentes de un transformer no se corresponden con un
recuento de capas. `backbone` coincide con ambos grupos de backbone por prefijo.

Las familias que no definen grupos semánticos recurren a un valor por defecto
conservador: cada hijo directo del modelo que tenga al menos un parámetro, en
orden de declaración. Esa suele ser una lista corta, así que un entero grande no
encontrará suficientes grupos:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Para ver la lista real en lugar de adivinar:

<code-tabs name="groups" />

## Los fallos son ruidosos

Cada manera de equivocarse con esto lanza un error en vez de entrenar algo que no
habías pedido.

Un selector que no coincide con nada lanza un error e indica los selectores que
no han coincidido:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Una congelación que dejaría el modelo sin nada entrenable lanza un error, tanto
en el momento de congelar como de nuevo al construir el optimizador:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Que es lo que hace `freeze="all"`, ya que `all` coincide con todos los
parámetros.

Cuando la congelación funciona, una línea deja constancia de lo ocurrido:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## El BatchNorm congelado deja de actualizarse

Un parámetro congelado sigue estando dentro de un módulo cuyas estadísticas
acumuladas seguirían moviéndose. Todo módulo de tipo BatchNorm cuyos parámetros
caigan en el conjunto congelado se pasa a modo eval, y el trainer vuelve a
aplicarlo después de la llamada a `model.train()` de cada epoch, así que las
estadísticas quedan fijas durante toda la ejecución.

Esto está activado por defecto y es lo que hace que congelar un backbone lo
congele de verdad.

## Combinar con LoRA

`freeze` y `lora=True` funcionan juntos. En RF-DETR, DEIM y ConvNeXt los
parámetros del adaptador se mantienen entrenables incluso cuando su grupo padre
está congelado, que es la combinación que quieres: un backbone congelado con
adaptadores que aprenden encima. Consulta [Fine-tuning con LoRA](/docs/train/lora).

## Alcance

Esto es congelación estática decidida al arrancar. La descongelación programada y
la congelación progresiva no forman parte de la interfaz.

## Relacionado

- [Hiperparámetros](/docs/train/hyperparameters) para el resto de `train()`.
- [Destilación](/docs/train/distillation) para la otra forma de trasladar el
  conocimiento de un modelo grande a un entrenamiento.
