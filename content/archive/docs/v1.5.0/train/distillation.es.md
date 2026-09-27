---
title: Destilación de conocimiento
seo_title: Destilación de conocimiento en LibreYOLO
description: >-
  Entrena un detector pequeño contra un teacher más grande o contra un backbone
  DINOv2 congelado: las losses MGD, CWD y feature-MSE, los puntos de toma y las
  familias soportadas.
lead: >-
  La destilación añade un segundo término de loss que acerca los mapas de
  características intermedios del student a los de un teacher congelado.
  LibreYOLO toma las características con forward hooks, así que la cabeza y la
  loss del propio teacher nunca intervienen.
keywords:
  - destilacion de conocimiento
  - masked generative distillation
  - destilacion channel-wise
  - destilacion de caracteristicas
  - teacher dinov2
  - entrenamiento teacher student
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un checkpoint más grande de la misma familia supervisa al pequeño.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un ViT autosupervisado y congelado supervisa una etapa del backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Ajustar la loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # peso global de la destilación
            distill_tau=1.0,   # temperatura del softmax de CWD
        )
source_hash: 7210031328f6826f
---

## Destilar desde un checkpoint más grande

Definir `distill_model` activa la destilación. El valor es un checkpoint de
teacher, cargado con la misma factoría que cualquier otro modelo.

<code-tabs name="detector" />

El teacher hace el forward bajo `no_grad`, y bajo autocast cuando AMP está
activo, de modo que el modelo congelado no paga cómputo en precisión completa en
cada paso. Los forward hooks capturan sus mapas de características en puntos de
toma con nombre, la loss los compara con los del student, y el resultado se suma
a la loss de entrenamiento y se reporta como un componente llamado `distill`.

## Destilar desde un backbone fundacional congelado

Un ViT autosupervisado puede supervisar en su lugar una sola etapa del backbone
del student. Las características del teacher vienen de su propio extractor de
características en lugar de hooks, y la loss se encarga del desajuste entre una
rejilla de parches y un stride convolucional.

<code-tabs name="foundation" />

`distill_model` reconoce `dinov2`, que es DINOv2-base, además de `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`,
y cualquier id de hub en bruto que empiece por `facebook/dinov2`. Cualquier otra
cosa se trata como la ruta de un checkpoint de teacher.

Esta vía usa `feat_mse` independientemente de `distill_loss_type`, y necesita
`transformers` instalado. Un teacher que se carga con claves de pesos ausentes
aborta en lugar de destilar contra un backbone parcialmente aleatorio.

## Qué familias

El soporte de destilación es un método del modelo student, y hay dos de ellos.

`get_distill_config()` proporciona los puntos de toma multiescala que supervisa
un teacher detector. YOLOv9, YOLOX y RF-DETR lo implementan.

`get_backbone_distill_config()` proporciona la única etapa del backbone que
supervisa un teacher fundacional. YOLOv9 lo implementa, y es la única familia que
lo hace.

Cualquier otra cosa lanza un error en lugar de entrenar sin la loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Puntos de toma

Los puntos de toma son fijos por familia y por rol, así que teacher y student no
necesitan ser la misma arquitectura; necesitan strides de características que
coincidan.

| Familia | Rol | Puntos de toma | Strides |
|---|---|---|---|
| YOLOv9 | teacher o student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | student fundacional | `backbone.elan3` | 16 |
| YOLOX | teacher o student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher o student | `model.backbone.0.projector.stages.0` | sondeado en la configuración |

Los strides que no coinciden lanzan un error antes de que empiece el
entrenamiento:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Esa comprobación se omite para los teachers fundacionales, cuyo sentido es
precisamente que las rejillas difieran.

## Las tres losses

`distill_loss_type` selecciona la loss de características para un teacher
detector. Un teacher fundacional siempre usa `feat_mse`.

`mgd`, masked generative distillation, enmascara una fracción de las posiciones
espaciales del student y entrena un pequeño generador de dos convoluciones para
reconstruir el mapa de características completo del teacher a partir de lo que
queda. `distill_mask_ratio` fija la fracción enmascarada, 0.65 por defecto.

`cwd`, channel-wise distillation, convierte las activaciones espaciales de cada
canal en una distribución de probabilidad y minimiza la divergencia KL canal a
canal. `distill_tau` es la temperatura del softmax, 1.0 por defecto.

`feat_mse` alinea los canales del student con los del teacher mediante una
convolución 1x1, redimensiona la rejilla del teacher a la del student de forma
bilineal, y toma el error cuadrático medio. `distill_normalize=True` normaliza
antes ambos mapas de características con L2 sobre la dimensión de canales, lo que
hace que la coincidencia sea solo de ángulo e invariante a la escala. Su valor
por defecto es `False`.

`dis` es el peso global que se aplica por encima. Si no se define, cada loss usa
su propio valor por defecto publicado: 2e-5 para MGD, 1.0 para CWD y 1.0 para
feature MSE. Difieren en cinco órdenes de magnitud, así que un peso ajustado para
un tipo de loss no significa nada para otro.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` y `distill_normalize` no tienen flags de CLI.
Son argumentos de Python o claves YAML de `cfg=`. RF-DETR también es solo Python
para la destilación en su conjunto, porque su mapeo de argumentos de CLI no
incluye las claves de destilación.

## Adaptadores, checkpoints y multi-GPU

Cada loss construye pequeños módulos entrenables que viven fuera del student: los
adaptadores de canales 1x1, y el generador de MGD. Reciben su propio grupo de
parámetros del optimizador con el learning rate efectivo de la ejecución.

Esos módulos se escriben en el checkpoint bajo una clave `distiller` y se
restauran al reanudar, así que una ejecución reanudada no reinicia sus
proyectores desde cero.

Bajo DDP los adaptadores quedan fuera del student envuelto, lo que significa que
el reducer de DDP nunca ve sus gradientes. El trainer hace un all-reduce
explícito de ellos en cada paso, así que todos los ranks entrenan los mismos
adaptadores.

La captura de CUDA graphs no está disponible en una ejecución con destilación.
Pasar `cuda_graph=True` registra una línea y entrena en modo eager. Consulta
[Rendimiento del entrenamiento](/docs/train/performance).

## Relacionado

- [Congelación de capas](/docs/train/layer-freezing) y
  [fine-tuning con LoRA](/docs/train/lora), a ninguno de los cuales se le impide
  combinarse con la destilación.
- [Hiperparámetros](/docs/train/hyperparameters) para el resto de `train()`.
