---
title: "LiteRT vs TensorFlow Lite: é só uma mudança de nome, não um novo formato"
description: "Em setembro de 2024, o Google renomeou TensorFlow Lite para LiteRT. O runtime e os arquivos .tflite continuam iguais, sem nada quebrar. Veja por que a mudança aconteceu, o que realmente mudou e como exportar um modelo LibreYOLO para esse formato."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "O TensorFlow Lite foi descontinuado?"
    a: "O nome está sendo aposentado, não a tecnologia. O runtime continua como LiteRT, com as mesmas APIs, e os modelos e códigos TFLite existentes continuam funcionando. Os novos recursos, como a API CompiledModel e a aceleração por NPU, são lançados sob o nome LiteRT."
  - q: "Preciso exportar meus modelos novamente para usar o LiteRT?"
    a: "Não. Um arquivo .tflite exportado para TensorFlow Lite é um modelo LiteRT. Nada mudou no arquivo, então não há nada a reexportar nem a migrar."
---

**LiteRT é o TensorFlow Lite com um novo nome. Não é um novo runtime nem um novo formato de arquivo. Os modelos continuam usando arquivos `.tflite`, e todo arquivo `.tflite` que rodava no TensorFlow Lite roda no LiteRT sem alterações.**

Se você pesquisou por "LiteRT vs TensorFlow Lite", "o TensorFlow Lite foi descontinuado" ou "o que é LiteRT", esse parágrafo responde. O restante desta página resume por que o nome mudou e como exportar um modelo para LiteRT com o LibreYOLO.

## Por que o Google mudou o nome

O TensorFlow Lite foi lançado em 2017 como uma forma de rodar modelos TensorFlow em celulares e placas embarcadas. Com o passar dos anos, ele foi muito além disso: o Google criou caminhos de conversão a partir de PyTorch, JAX e Keras. Em 2024, uma grande parte dos modelos que rodavam nele nunca tinha usado TensorFlow.

Nesse ponto, o nome passou a induzir ao erro. Usuários de PyTorch deixavam de usar a ferramenta porque ela parecia exclusiva do TensorFlow. Por isso, em setembro de 2024, o Google [renomeou o projeto para LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), abreviação de "Lite Runtime". É só isso. A mesma equipe, o mesmo código e um nome independente de framework.

## O que realmente mudou

Muito pouco, e nada disso quebra o que já existe:

* O código foi para um novo endereço: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* A documentação foi para [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* O pacote Python para rodar modelos agora se chama `ai-edge-litert`. O pacote antigo `tflite-runtime` está desatualizado; o novo tem a mesma classe `Interpreter`.
* Desde a mudança de nome, os novos recursos são lançados sob o nome LiteRT: uma API `CompiledModel` mais simples e aceleração por GPU/NPU, que o Google declarou [pronta para produção em janeiro de 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

A API clássica `Interpreter` também continua funcionando como antes. Há uma extensão realmente nova, `.litertlm`, mas ela é um formato de empacotamento apenas para LLMs no dispositivo. Para modelos de visão computacional, ela não se aplica. Portanto, quando um documento diz "TFLite" e outro diz "LiteRT", ambos estão falando do mesmo arquivo.

## Exportar um modelo LibreYOLO para LiteRT

O LibreYOLO adicionou uma opção de exportação para TFLite/LiteRT na versão v1.3.0. Ela requer Python 3.12+ e uma dependência extra:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Ou pela CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Os caminhos validados atualmente são detecção com YOLO9 e detecção, segmentação e pose com RF-DETR, todos ainda marcados como experimentais. A matriz completa de suporte está na [documentação](/docs/v1.3.0).

Para rodar o arquivo exportado, use o pacote `ai-edge-litert` do Google no dispositivo de destino:

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

Observe que a entrada usa NHWC (canais por último): a conversão a partir de ONNX transpõe o layout, o que é normal nesse formato.

## O que mudamos no LibreYOLO

Quase nada, porque quase nada precisava mudar. A documentação agora identifica o formato como "TFLite (LiteRT)" para que seja possível encontrá-lo pelos dois nomes, e a API mantém `format="tflite"` como nome do formato.

## FAQ

**O TensorFlow Lite foi descontinuado?**
O nome está sendo aposentado, não a tecnologia. O runtime continua como LiteRT, com as mesmas APIs, e os modelos e códigos TFLite existentes continuam funcionando. Os novos recursos, como a API CompiledModel e a aceleração por NPU, são lançados sob o nome LiteRT.

**Preciso exportar meus modelos novamente para usar o LiteRT?**
Não. Um arquivo `.tflite` exportado para TensorFlow Lite é um modelo LiteRT. Nada mudou no arquivo, então não há nada a reexportar nem a migrar.
