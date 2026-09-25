---
title: "LiteRT vs TensorFlow Lite: è solo un cambio di nome, non un nuovo formato"
description: "A settembre 2024 Google ha rinominato TensorFlow Lite in LiteRT. Stesso runtime, stessi file .tflite, non si rompe nulla. Ecco perché è avvenuto il cambio di nome, cosa è cambiato davvero e come esportare un modello LibreYOLO in questo formato."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite è deprecato?"
    a: "Viene ritirato il nome, non la tecnologia. Il runtime continua a esistere come LiteRT con le stesse API, e i modelli e il codice TFLite esistenti continuano a funzionare. Il nuovo sviluppo, come l'API CompiledModel e l'accelerazione NPU, viene rilasciato con il nome LiteRT."
  - q: "Devo esportare di nuovo i miei modelli per LiteRT?"
    a: "No. Un file .tflite esportato per TensorFlow Lite è un modello LiteRT. Il file non è cambiato, quindi non c'è niente da esportare di nuovo o migrare."
---

**LiteRT è TensorFlow Lite con un nuovo nome. Non è un nuovo runtime e non è un nuovo formato di file. I modelli sono ancora file `.tflite` e ogni file `.tflite` che funzionava con TensorFlow Lite funziona anche con LiteRT senza modifiche.**

Se hai cercato "LiteRT vs TensorFlow Lite", "is TensorFlow Lite deprecated" o "what is LiteRT", la risposta è nel paragrafo qui sopra. Il resto della pagina spiega in breve perché è avvenuto il cambio di nome e come esportare un modello con LibreYOLO.

## Perché Google l'ha rinominato

TensorFlow Lite è stato lanciato nel 2017 come soluzione per eseguire i modelli TensorFlow su telefoni e schede embedded. Negli anni è cresciuto ben oltre questo ambito: Google ha realizzato percorsi di conversione da PyTorch, JAX e Keras, così che nel 2024 gran parte dei modelli che lo usavano non aveva mai avuto a che fare con TensorFlow.

A quel punto il nome era decisamente fuorviante. Gli utenti PyTorch lo ignoravano perché sembrava uno strumento riservato a TensorFlow. Così, a settembre 2024, Google [l'ha rinominato LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), abbreviazione di "Lite Runtime". Tutto qui. Stesso team, stesso codice, nome indipendente dal framework.

## Che cosa è cambiato davvero

Ben poco, e nulla che comprometta il funzionamento:

* Il codice è stato spostato in una nuova sede: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* La documentazione è stata spostata su [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Il pacchetto Python per eseguire i modelli ora si chiama `ai-edge-litert`. Il vecchio pacchetto `tflite-runtime` è obsoleto; quello nuovo ha la stessa classe `Interpreter`.
* Dopo il cambio di nome, le nuove funzionalità vengono rilasciate con il nome LiteRT: un'API `CompiledModel` più semplice e l'accelerazione GPU/NPU, che Google ha dichiarato [pronta per la produzione a gennaio 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

Anche la classica API Interpreter continua a funzionare come prima. C'è una sola estensione davvero nuova, `.litertlm`, ma è un formato di packaging riservato agli LLM su dispositivo; per quanto ti riguarda, non esiste per i modelli di computer vision. Quindi, quando un documento dice "TFLite" e un altro dice "LiteRT", parlano dello stesso file.

## Esportare un modello LibreYOLO in LiteRT

LibreYOLO ha aggiunto il percorso di esportazione TFLite/LiteRT nella v1.3.0. Richiede Python 3.12+ e un pacchetto aggiuntivo:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Oppure dalla CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

I percorsi convalidati al momento sono il rilevamento con YOLO9 e il rilevamento, la segmentazione e la stima della posa con RF-DETR, tutti ancora contrassegnati come sperimentali. La matrice completa del supporto è nella [documentazione](/docs/v1.3.0).

Per eseguire il file esportato, usa il pacchetto Google `ai-edge-litert` sul dispositivo di destinazione:

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

Nota che l'input è in formato NHWC (canali per ultimi): la conversione da ONNX trasponde il layout, come previsto da questo formato.

## Cosa abbiamo cambiato in LibreYOLO

Poco, perché c'era poco da cambiare. Ora la documentazione indica il formato come "TFLite (LiteRT)" per renderlo rintracciabile con entrambi i nomi, e l'API mantiene `format="tflite"` come nome del formato.

## FAQ

**TensorFlow Lite è deprecato?**
Il nome viene ritirato, non la tecnologia. Il runtime continua a esistere come LiteRT con le stesse API, e i modelli e il codice TFLite esistenti continuano a funzionare. Il nuovo sviluppo, come l'API CompiledModel e l'accelerazione NPU, viene rilasciato con il nome LiteRT.

**Devo esportare di nuovo i miei modelli per LiteRT?**
No. Un file `.tflite` esportato per TensorFlow Lite è un modello LiteRT. Il file non è cambiato, quindi non c'è niente da esportare di nuovo o migrare.
