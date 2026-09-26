---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet in LibreYOLO: predizione, addestramento ed esportazione'
description: >-
  Esegui RTMDet in LibreYOLO per il rilevamento di oggetti e RTMDet-Ins per la
  segmentazione di istanze. Installa, fai predizioni, addestra, valida ed
  esporta con licenza Apache-2.0.
lead: >-
  RTMDet è un rilevatore single-stage che fa predizioni a partire da un prior
  puntuale per ogni posizione della griglia, senza anchor, attraverso una testa
  le cui convoluzioni sono condivise tra i livelli di feature. LibreYOLO lo
  supporta per il rilevamento e per la segmentazione di istanze RTMDet-Ins.
keywords:
  - RTMDet
  - RTMDet-Ins
  - object detection
  - rilevamento oggetti python
  - instance segmentation
  - segmentazione di istanze
  - detector anchor-free
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -seg nel nome del file seleziona la testa per le maschere
        # di RTMDet-Ins, quindi qui non serve nessun argomento task.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maschere
        print(metrics["metrics/mAP50-95(B)"])   # box
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Installazione

RTMDet non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. Un nome di file con
`-seg` risolve da solo al task RTMDet-Ins, e `result.masks` porta allora le
maschere di istanza accanto ai box. `conf` imposta la soglia di confidenza e
`iou` la soglia NMS. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Cinque dimensioni, da `t` a `x`, condividono un'unica architettura alla stessa
risoluzione di input. Questa famiglia qui non porta una tabella di benchmark:
confronta le dimensioni in base alla dimensione del file di checkpoint nella
tabella qui sotto.

## Addestramento

<code-tabs name="train" />

Il rilevamento si addestra con `train()`. I componenti QualityFocalLoss, GIoU e
DynamicSoftLabelAssigner sono portati da mmdetection upstream, e il forward pass
e l'esportazione ONNX sono bit-equivalenti a quelli originali, con un
postprocessing che corrisponde all'output di mmdet entro 0.001 mAP su
sottoinsiemi di val2017.

Quello che non è stato verificato, secondo la docstring di `train()` stessa: la
convergenza del fine-tuning su dataset piccoli, la parità con il paper
addestrando da zero, il comportamento multi-GPU, il throughput di Mosaic e MixUp
con la cache, il passaggio alla pipeline stretta a due fasi di upstream e gli
override paramwise del weight decay che azzerano il decay sui parametri di norma
e di bias.

RTMDet-Ins non ha un percorso di addestramento. Chiamare `train()` su un
checkpoint `-seg`, oppure con `task="segment"`, solleva `NotImplementedError`;
la segmentazione di istanze supporta solo inferenza e validazione.

`train()` accetta anche un argomento `pretrained`, ma il valore non viene mai
letto dentro il metodo: l'addestramento riparte sempre dai pesi con cui il
modello è stato costruito, quindi `pretrained=False` non reinizializza la rete.

Se non tocchi nient'altro, il trainer esegue 300 epoche con AdamW a `lr0=0.004`
e `weight_decay=0.05`, un warmup di 1 epoca su uno schedule a coseno, e Mosaic e
MixUp disattivati nelle ultime 20 epoche.

Vedi [addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui hai
addestrato.

<code-tabs name="val" />

Su un checkpoint `-seg` la chiave semplice `metrics/mAP50-95` contiene il
punteggio delle maschere, e la stessa esecuzione riporta anche i box sotto `(B)`
e le maschere sotto `(M)`, così sono disponibili entrambi da un solo passaggio.

## Esportazione

<export-matrix />

Il rilevamento si esporta nella maggior parte dei formati; la segmentazione di
istanze al momento non si esporta in nessuno di essi; la matrice qui sopra
riflette questa divisione. Un artefatto di rilevamento esportato si ricarica con
`LibreYOLO()` in base al suffisso del file, quindi un file `.onnx` o `.engine` si
comporta come un checkpoint e restituisce gli stessi `Results`. Anche eseguire il
grafo in un runtime nudo, senza LibreYOLO installato, è supportato, ma in quel
caso preprocessing e postprocessing li scrivi tu.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
