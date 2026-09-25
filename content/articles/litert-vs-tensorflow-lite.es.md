---
title: "LiteRT vs TensorFlow Lite: qué es LiteRT y qué cambia con el nuevo nombre"
description: "Google cambió el nombre de TensorFlow Lite a LiteRT en septiembre de 2024. Es el mismo runtime, los mismos archivos .tflite y nada deja de funcionar. Descubre por qué lo hicieron, qué cambió realmente y cómo exportar un modelo LibreYOLO a este formato."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "¿TensorFlow Lite está obsoleto?"
    a: "Se está retirando el nombre, no la tecnología. El runtime sigue como LiteRT con las mismas API, y los modelos y el código TFLite existentes siguen funcionando. El nuevo desarrollo, como la API CompiledModel y la aceleración NPU, se publica bajo el nombre LiteRT."
  - q: "¿Tengo que volver a exportar mis modelos para LiteRT?"
    a: "No. Un archivo .tflite exportado para TensorFlow Lite es un modelo LiteRT. El archivo no ha cambiado, así que no hay nada que volver a exportar ni migrar."
---

**LiteRT es TensorFlow Lite con un nombre nuevo. No es un runtime nuevo ni un formato de archivo nuevo. Los modelos siguen siendo archivos `.tflite`, y todos los archivos `.tflite` que funcionaban con TensorFlow Lite funcionan con LiteRT sin cambios.**

Si has buscado «LiteRT vs TensorFlow Lite», «¿TensorFlow Lite está obsoleto?» o «¿qué es LiteRT?», ese párrafo es la respuesta. El resto de esta página resume por qué se cambió el nombre y cómo exportar un modelo a este formato con LibreYOLO.

## Por qué Google le cambió el nombre

TensorFlow Lite se lanzó en 2017 como la forma de ejecutar modelos TensorFlow en teléfonos y placas integradas. Con los años, su alcance creció mucho: Google creó rutas de conversión desde PyTorch, JAX y Keras, así que para 2024 una gran parte de los modelos que se ejecutaban con él nunca había pasado por TensorFlow.

Llegado ese punto, el nombre inducía claramente a error. Los usuarios de PyTorch lo descartaban porque parecía una herramienta exclusiva de TensorFlow. Así que en septiembre de 2024 Google [lo cambió a LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), abreviatura de «Lite Runtime». Eso es todo. El mismo equipo, el mismo código y un nombre independiente del framework.

## Qué cambió realmente

Muy poco, y nada de esto rompe nada:

* El código se trasladó a una nueva ubicación: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* La documentación se trasladó a [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* El paquete de Python para ejecutar modelos ahora se llama `ai-edge-litert`. El antiguo paquete `tflite-runtime` está desactualizado; el nuevo tiene la misma clase `Interpreter`.
* Desde el cambio de nombre, las nuevas funciones se publican bajo el nombre LiteRT: una API `CompiledModel` más sencilla y aceleración GPU/NPU, que Google declaró [lista para producción en enero de 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

La API clásica `Interpreter` también sigue funcionando tal cual. Hay una extensión realmente nueva, `.litertlm`, pero es un formato de empaquetado solo para LLM en dispositivos; en lo que respecta a los modelos de visión artificial, no existe. Así que cuando un documento dice «TFLite» y otro dice «LiteRT», se refieren al mismo archivo.

## Exportar un modelo LibreYOLO a LiteRT

LibreYOLO añadió una ruta de exportación a TFLite/LiteRT en v1.3.0. Necesita Python 3.12 o superior y un extra:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

También puedes usar la CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Las rutas validadas actualmente son la detección con YOLO9 y la detección, segmentación y estimación de pose con RF-DETR; todas siguen marcadas como experimentales. La matriz completa de compatibilidad está en la [documentación](/docs/v1.3.0).

Para ejecutar el archivo exportado, usa el paquete de Google `ai-edge-litert` en el dispositivo de destino:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Ten en cuenta que la entrada usa el formato NHWC (canales al final): la conversión desde ONNX transpone la disposición, lo cual es normal en este formato.

## Qué cambiamos en LibreYOLO

No mucho, porque no hacía falta cambiar gran cosa. Ahora la documentación identifica el formato como «TFLite (LiteRT)» para que se pueda encontrar con ambos nombres, y la API conserva `format="tflite"` como nombre del formato.

## Preguntas frecuentes

**¿TensorFlow Lite está obsoleto?**
Se está retirando el nombre, no la tecnología. El runtime sigue como LiteRT con las mismas API, y los modelos y el código TFLite existentes siguen funcionando. El nuevo desarrollo, como la API CompiledModel y la aceleración NPU, se publica bajo el nombre LiteRT.

**¿Tengo que volver a exportar mis modelos para LiteRT?**
No. Un archivo `.tflite` exportado para TensorFlow Lite es un modelo LiteRT. El archivo no ha cambiado, así que no hay nada que volver a exportar ni migrar.
