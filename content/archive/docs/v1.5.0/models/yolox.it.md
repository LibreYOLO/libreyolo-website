---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: predizione, addestramento ed esportazione con licenza Apache-2.0'
description: >-
  Usa YOLOX in LibreYOLO per il rilevamento di oggetti: installa, fai
  predizioni, addestra, valida ed esporta con licenza Apache-2.0.
lead: >-
  YOLOX è un rilevatore single-stage anchor-free con una testa disaccoppiata per
  classificazione e regressione, addestrato con l'assegnazione delle etichette
  SimOTA. LibreYOLO lo supporta per il rilevamento.
keywords:
  - YOLOX
  - object detection
  - rilevamento oggetti python
  - anchor-free detection
  - decoupled head
  - SimOTA
  - real-time object detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Su COCO
      language: bash
      code: |
        # Il file yaml di COCO incluso porta uno script di download integrato,
        # quindi serve un permesso esplicito se il dataset non è già in locale.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Installazione

YOLOX non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` imposta la
soglia di confidenza e `iou` la soglia NMS applicata alle tre scale di predizione
disaccoppiate. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Sei dimensioni condividono lo stesso backbone CSP e lo stesso neck PAFPN. Le due
più piccole, `n` e `t`, girano a una risoluzione di input fissa più piccola
rispetto alle altre quattro; la tabella di benchmark qui sotto riporta la cifra
esatta per ciascuna.

<benchmark-table task="detect" />

<va-embed />

## Addestramento

<code-tabs name="train" />

Se non tocchi nient'altro, il trainer esegue 300 epoche a `lr0=0.01` con SGD
momentum 0.9, un warmup di 5 epoche e le augmentation mosaic e mixup disattivate
nelle ultime 15 epoche. `train()` accetta anche un argomento `pretrained`, ma il
valore non viene mai letto dentro il metodo: l'addestramento riparte sempre dai
pesi con cui il modello è stato costruito, quindi `pretrained=False` non
reinizializza la rete.

`imgsz` come predefinito prende un valore fisso dalla configurazione base di
addestramento, non la risoluzione nativa del checkpoint caricato. Questo riguarda
in particolare i checkpoint `n` e `t`: continuare ad addestrare uno dei due senza
impostare `imgsz` esplicitamente lo porta al valore predefinito più grande invece
che alla dimensione più piccola con cui è stato pubblicato.

Vedi [addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui hai
addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce gli stessi `Results`. Anche eseguire il grafo in un runtime nudo,
senza LibreYOLO installato, è supportato, ma in quel caso preprocessing e
postprocessing li scrivi tu. Un'esportazione CoreML può integrare l'NMS nel grafo
con `nms=True`; YOLOX e YOLOv9 sono le uniche due famiglie per cui quel flag al
momento è accettato.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
