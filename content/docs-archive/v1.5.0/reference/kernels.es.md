---
title: Kernels
seo_title: Registro de kernels de LibreYOLO y kernels del Hub
description: >-
  Cómo selecciona LibreYOLO las implementaciones aceleradas: el registro de
  kernels en libreyolo/kernels, el kernel opcional MS-deform-attn de Hugging
  Face Hub y el interruptor de atención fusionada.
lead: >-
  Cada operación acelerada de LibreYOLO tiene una implementación portable por
  defecto y, a veces, una variante más rápida registrada por encima. La
  selección ocurre en tiempo de ejecución mediante un predicado, una dependencia
  opcional que falta es un fallback y no un error, y un grafo exportado toma
  siempre el camino portable.
keywords:
  - libreyolo kernels
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - extra hub-kernels
  - kernel ms_deform_attn
  - set_fused_attention
  - kernels triton libreyolo cuda
last_verified: 1.5.0
verification: >-
  API del registro leída de libreyolo/kernels/__init__.py en la v1.5.0, API de
  atención de libreyolo/kernels/attention/__init__.py y sdpa.py, proveedor del
  Hub de libreyolo/kernels/attention/ms_deform_attn.py incluyendo su revisión
  fijada y su predicado de elegibilidad. Estructura de directorios listada desde
  libreyolo/kernels/. Definición del extra desde pyproject.toml. Notas de
  comportamiento y cifras de benchmark de docs/kernels.md. El historial del
  gating de la v1.4.0, del commit que cableó el slot en RF-DETR y de la entrada
  del CHANGELOG de 1.5.0.
meta:
  - label: Paquete
    value: libreyolo.kernels
    mono: true
  - label: Extra opcional
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Forzar la referencia
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Ver qué se ha seleccionado
      language: python
      code: >
        import libreyolo.kernels as kernels


        # Slot de op al nombre de la implementación seleccionada, o
        "unavailable".

        print(kernels.active())
    - label: Forzar el camino de referencia
      language: bash
      code: |
        # off y reference significan lo mismo, y además evitan por completo
        # importar los proveedores acelerados.
        LIBREYOLO_KERNELS=off python train.py
    - label: Desactivar los kernels del Hub sin desinstalar
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Cambiar una familia a atención fusionada
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Devuelve cuántos módulos de atención han cambiado.
        print(set_fused_attention(model))
    - label: Registrar el tuyo propio
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## El registro

`libreyolo/kernels/` es un pequeño registro en tiempo de ejecución de
implementaciones intercambiables. Un slot de op es un nombre como
`fake_quant_fp8` o `ms_deform_attn`. Quien llama pide un slot al registro y
recibe la primera implementación registrada que pase su predicado, ganando el
registro más reciente, y cayendo a la implementación de referencia cuando no
aplica ninguna otra.

Esa estructura existe para que una dependencia opcional nunca sea un requisito
obligatorio. Una máquina sin Triton, sin CUDA o sin el paquete `kernels`
ejecuta el mismo código y produce los mismos números, solo que más despacio.

| Función | Propósito |
|---|---|
| `active()` | Slot de op al nombre de la implementación seleccionada, o `"unavailable"` |
| `resolve(op)` | El callable que se ejecutaría, o `None` |
| `register(op, impl, *, name, predicate=None)` | Añade una implementación, la más reciente primero |
| `unregister(op, name)` | Elimina una |
| `clear_cache()` | Descarta la resolución memoizada |

<code-tabs name="usage" />

Un predicado que lanza una excepción se captura y se avisa de ello, nunca se
propaga, de modo que una implementación de terceros rota degrada al camino
portable en lugar de romper la predicción.

### Estructura

El árbol se organiza primero por propósito y después por backend, de forma que
un slot se encuentra por lo que calcula y no por qué biblioteca lo implemente
hoy.

| Directorio | Contenido |
|---|---|
| `kernels/quant/simulate/` | Kernels de Triton de cuantización simulada, con backward straight-through, en cualquier dispositivo. Los usan tanto el QAT como la cuantización post-entrenamiento simulada |
| `kernels/quant/execute/` | Caminos de precisión real solo para modelos finalizados, sin backward: el GEMM FP8 sobre tensor cores, su prólogo y su epílogo fusionados en Triton, y los kernels de desempaquetado de pesos empaquetados |
| `kernels/attention/` | Ops de atención compartidas entre familias: el slot `ms_deform_attn` y la política de SDPA fusionada |

La frontera entre `simulate` y `execute` es si el modelo está finalizado, no si
está entrenando o desplegándose. Las implementaciones de referencia se quedan en
`libreyolo/quant/`, que define qué significan los números; `kernels/` solo los
hace rápidos. El empaquetado de pesos no tiene variantes en absoluto, porque es
el contrato del checkpoint.

Los slots de GEMM y de atención no tienen implementación de referencia. Quien
llama tiene que comprobar que `resolve()` ha devuelto algo y mantener su propio
camino portable, y por eso los grafos de ONNX, TensorRT y `torch.export`
contienen siempre las matemáticas portables.

### Anulaciones de la selección

`LIBREYOLO_KERNELS=off` o `=reference` fuerza las implementaciones de referencia
y corta por completo la importación de los proveedores acelerados. Cualquier
otro valor restringe la selección a las implementaciones registradas bajo ese
nombre. `LIBREYOLO_QUANT_KERNELS` se respeta como alias heredado de cuando el
registro vivía en `libreyolo/quant/`, y solo se lee cuando `LIBREYOLO_KERNELS`
no está definida. Ambas aparecen junto al resto en
[ajustes](/docs/reference/settings).

## Kernels del Hub

Los kernels CUDA compilados que se publican en Hugging Face Hub se cargan en
tiempo de ejecución a través del paquete opcional `kernels`. No se incorpora
nada al árbol de LibreYOLO; ese paquete descarga y cachea el artefacto, y cada
proveedor fija una revisión de commit auditada, así que subir un pin exige una
pasada de paridad en GPU antes de entrar.

Instalar el extra es la forma de activarlo:

```bash
pip install "libreyolo[hub-kernels]"
```

Sin el paquete no cambia nada y no se hace ninguna petición de red.
`LIBREYOLO_HUB_KERNELS=0` desactiva la descarga sin desinstalar nada. Un kernel
que falla al cargarse o al ejecutarse se desactiva a sí mismo durante el resto
del proceso y cae al fallback con un único aviso.

Hoy hay un slot respaldado por el Hub: `ms_deform_attn`, el forward y el
backward compilados de la atención deformable multiescala de Deformable DETR,
bajo Apache 2.0. Está cableado en todo el linaje deformable: RF-DETR, Deformable
DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4,
DEIM, DEIMv2, EC y OV-DEIM. Como el backward también está compilado, el
entrenamiento se beneficia igual que la predicción.

La elegibilidad es estrecha a propósito. Las entradas deben ser CUDA y float32,
y la ejecución debe ser eager: el proveedor se abstiene bajo
`torch.jit.is_tracing()`, `torch.compiler.is_compiling()`,
`torch.compiler.is_exporting()` y `torch.onnx.is_in_onnx_export()`. También caen
al camino portable dos disposiciones de entrada, un número de puntos por nivel
que varía entre niveles y el muestreo por índices enteros discretos. La variante
de pose de EC no está cableada.

### Este kernel es alcanzable por primera vez

Lee esto antes de instalar el extra en un proyecto que ya existe.

En la v1.4.0 el slot se consultaba desde dentro de un helper, detrás de una
condición que exigía que los pares de formas espaciales estuvieran ausentes.
RF-DETR siempre pasa esos pares por su decodificador, así que la condición nunca
se cumplía y el kernel no se ejecutaba en ningún forward eager. La consulta se
movió en la v1.5.0, y ahora el kernel sí se ejecuta de verdad.

La consecuencia práctica es que actualizar a la v1.5.0 *e* instalar
`libreyolo[hub-kernels]` sobre CUDA significa que RF-DETR y su linaje toman su
forward de un binario compilado por primera vez. Como resultado, las
predicciones y las métricas pueden moverse dentro de la tolerancia de coma
flotante. Una instalación estándar, sin el extra, no se ve afectada. Si estás
comparando métricas a ambos lados de la actualización, mantén el extra fijo o
pon `LIBREYOLO_HUB_KERNELS=0` en los dos lados.

## Atención fusionada

La atención fusionada de producto escalar escalado no necesita ninguna
dependencia opcional, solo PyTorch de serie, así que la gobierna una política y
no la disponibilidad. Se aplican dos reglas.

Primero, una captura de grafo nunca la usa. Cada punto de llamada sustituido
mantiene disponible la ecuación con ops primitivas detrás de una comprobación de
exportación, lo que cubre la exportación a ONNX, cuyo opset por defecto no tiene
un símbolo para SDPA, y `torch.jit.trace`, por donde pasan TorchScript, CoreML y
NCNN. Las capturas de Dynamo quedan deliberadamente fuera de la barrera, porque
`torch.compile` baja SDPA mejor que las matemáticas manuales, y tanto Core AI
como ExecuTorch descomponen SDPA a ATen core por su cuenta.

Segundo, el listón de paridad para convertirla en el valor por defecto es la
exactitud byte a byte. Las familias que lo superan usan SDPA por defecto:
SegFormer, Depth Anything y MoGe-2, BERT, Grounding DINO, SwinIR y PP-OCR. Las
que no lo superan mantienen las matemáticas manuales y exponen en su lugar un
flag `fused_attn`, que es lo que conmuta `set_fused_attention(model)`: Swin, el
backbone Swin de DINO-DETR, BiRefNet y FeyNobg, OWLv2, LW-DETR, SigLIP 2,
ZipDepth y MobileSAM. ViT y DeiT llevan el mismo flag pero lo traen activado por
defecto, siguiendo a upstream, así que la misma llamada con `enabled=False` los
desactiva.

Merece la pena donde se aplica. En una RTX 5070 Ti con autocast fp16, la
atención por ventanas de Swin pasa de 1,278 ms a 0,721 ms, una ganancia de
1,77x, y la atención de visión de OWLv2 de 6,483 ms a 1,735 ms, 3,74x.

## Hardware

| Plataforma | Comportamiento |
|---|---|
| CPU y MPS | Todos los predicados de CUDA y de Triton fallan, así que todo se ejecuta en referencia |
| NVIDIA CUDA | Se activan los kernels de Triton y los kernels del Hub y de GEMM que sean elegibles |
| AMD ROCm | Triton puede activarse, ya que las wheels de ROCm incluyen el backend de AMD de Triton, pero la paridad solo se ejercita sobre NVIDIA en CI |

## Añadir una implementación

Llama a `register()` con un nombre y un predicado. Los kernels compilados fuera
del árbol pueden distribuirse como un paquete `libreyolo_kernels` aparte que se
registra a sí mismo al importarse, lo que mantiene un backend privado
completamente fuera del árbol de LibreYOLO.

La paridad es la barrera para cualquier cosa dentro del árbol: una coincidencia
exacta del forward frente a la referencia, y gradientes dentro de 1e-6 del
estimador straight-through, sobre el conjunto de formas que lleva la suite de
tests.

La selección de kernels interactúa con los
[grafos CUDA](/docs/reference/cuda-graphs): la matriz de paridad de inferencia
se ejecutó sin el paquete `kernels` instalado, así que la seguridad de la
captura con un kernel compilado activo no queda cubierta por ella.
