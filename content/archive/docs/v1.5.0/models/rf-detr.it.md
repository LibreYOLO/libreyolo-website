---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: addestra, fai fine-tuning ed esporta con licenza MIT'
description: >-
  Usa RF-DETR in LibreYOLO per rilevamento, segmentazione di istanze, posa e box
  orientati. Installa, fai predizioni, addestra, valida ed esporta, tutto con
  licenza MIT.
lead: >-
  Un detection transformer che predice un insieme fisso di oggetti invece di una
  griglia densa, quindi non richiede NMS durante l'inferenza. LibreYOLO lo
  supporta per quattro task.
keywords:
  - RF-DETR
  - real-time detection transformer
  - DETR
  - object detection python
  - rilevamento oggetti in tempo reale
  - segmentazione di istanze
  - stima della posa
  - box orientati
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, rilevamento su video a 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() restituisce un semplice dict, non un oggetto
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Su COCO
      language: bash
      code: |
        # Lo yaml di COCO incluso porta uno script di download integrato,
        # quindi serve un permesso esplicito a meno che il dataset non sia
        # già in locale.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # Argomenti accettati per ogni formato:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine" è un alias per tensorrt, "litert" per tflite.
        #   imgsz     int, oppure (altezza, larghezza). Di default la
        #             risoluzione nativa del checkpoint.
        #   batch     int, default 1.
        #   half      bool, esporta in FP16. Default False.
        #   int8      bool, esporta in INT8. Default False. Richiede `data`.
        #   data      percorso di uno YAML di dataset, usato per calibrare int8.
        #   fraction  float, quota di quel set di calibrazione da usare.
        #             Default 1.0.
        #   dynamic   bool, assi dinamici. Default True.
        #   simplify  bool, esegue la semplificazione del grafo ONNX.
        #             Default True.
        #   opset     int, opset ONNX. Scelto per famiglia se non indicato.
        #   device    str, dispositivo su cui tracciare. Di default quello
        #             del modello.
        #   output_path  str, di default un nome derivato dal checkpoint.
        #   verbose   bool, default False.
        #   allow_download_scripts  bool, default False. Consente il Python
        #             integrato in uno YAML di dataset che va scaricato.
        #
        # Qualche formato accetta argomenti extra propri, come una piattaforma
        # target RKNN. Sono documentati nella pagina di ciascun formato.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Senza LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Eseguire il grafo direttamente significa scrivere da sé il

        # preprocessing e il postprocessing. Ispeziona la firma prima di

        # collegare qualsiasi cosa.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Installazione

RF-DETR richiede un extra dedicato, che porta con sé `transformers` per il
backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` e `max_det`
filtrano la selezione delle query; non c'è nessun passaggio di NMS da regolare.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Quattro dimensioni, e quattro task che condividono una sola architettura:
segmentazione, posa e box orientati riutilizzano il decoder di rilevamento con
una testa diversa, quindi accettano gli stessi argomenti. Le dimensioni hanno un
numero di parametri simile e differiscono soprattutto per la risoluzione di
input.

<benchmark-table task="detect" />

<va-embed />

## Addestramento

L'addestramento parte da un checkpoint pubblicato, per tutti e quattro i task.
RF-DETR elenca `pretrained` tra gli argomenti che il suo trainer nativo ignora,
quindi passare `pretrained=False` qui non ti dà un modello inizializzato a caso.

<code-tabs name="train" />

Due argomenti contano più qui che su un rilevatore CNN. Tieni `lr0` a `1e-4` o
sotto, perché i detection transformer divergono a learning rate che un modello
YOLO tollera. Lascia `imgsz` alla risoluzione nativa del checkpoint, a meno che
tu non abbia un motivo per cambiarla. L'input deve essere divisibile
esattamente per la dimensione delle patch del backbone moltiplicata per il
numero di finestre; LibreYOLO lo verifica prima che l'esecuzione parta e indica
le dimensioni valide più vicine.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurati su qualsiasi dataset nel formato su cui hai
addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportato anche eseguire il grafo in un
runtime nudo, senza LibreYOLO installato, ma allora preprocessing e
postprocessing tocca scriverli a te.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
