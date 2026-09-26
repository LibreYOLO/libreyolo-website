---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet in LibreYOLO: fai predizioni, addestra ed esporta'
description: >-
  Usa PicoDet in LibreYOLO per il rilevamento di oggetti su mobile. Installa,
  fai predizioni, addestra, valida ed esporta sotto licenza Apache-2.0.
lead: >-
  PicoDet è un rilevatore single-stage costruito per le CPU mobile ed edge: un
  backbone ESNet, un neck CSP-PAN e una testa condivisa con Generalized Focal
  Loss. LibreYOLO lo supporta per il rilevamento.
keywords:
  - PicoDet
  - PP-PicoDet
  - object detection
  - rilevamento oggetti python
  - object detection mobile
  - detection su dispositivi edge
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # Vale la pena impostare imgsz: la CLI lo porta a 640 di default, mentre

        # il checkpoint s è nativo a 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Installazione

PicoDet non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` imposta la
soglia di confidenza e `iou` la soglia di NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre dimensioni, ciascuna alla propria risoluzione di input fissa: `s` la più
piccola e `l` la più grande. La risoluzione cresce con la dimensione, quindi i
checkpoint più grandi sono anche più costosi da eseguire per immagine, oltre a
portare più parametri.

<benchmark-table task="detect" />

<va-embed />

## Addestramento

<code-tabs name="train" />

I componenti della loss e l'assigner seguono la ricetta upstream: VFL, DFL,
GIoU e SimOTA, con pesatura per qualità della classificazione e target VFL a
IoU dinamico. L'inferenza è bit-equivalente a quella upstream sullo stesso
checkpoint.

Quello che non è stato verificato, secondo la docstring di `train()` stessa: la
convergenza sull'intero dataset, il comportamento multi-GPU e qualsiasi data
augmentation oltre al flip orizzontale. Anche il checkpoint `s` alla sua
risoluzione nativa di 320 non ha superato in modo affidabile il minimo di
accuratezza di LibreYOLO sulla fixture da 30 immagini e due classi con cui la
libreria testa i fine-tuning piccoli. Quella dimensione si adatta meglio alla
scala di COCO completo.

`train()` accetta anche un argomento `pretrained`, ma il valore non viene mai
letto dentro il metodo: l'addestramento continua sempre dai pesi con cui il
modello è stato costruito, quindi `pretrained=False` non reinizializza la
rete. Se lasci `imgsz` non impostato in Python, prende la risoluzione nativa
del checkpoint caricato, 320 per `s`, 416 per `m` e 640 per `l`. La CLI invia
sempre un `imgsz`, con 640 di default, quindi lì va impostato per farlo
combaciare con il checkpoint.

Se non tocchi altro, il trainer esegue 300 epoche con SGD a `lr0=0.01`,
momentum 0.9, weight decay 4e-5 e un warmup di 1 epoca su uno schedule cosine.
Il flip orizzontale è l'unica data augmentation applicata.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato su cui hai
addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in un
runtime nudo, senza LibreYOLO installato, ma allora il preprocessing e il
postprocessing tocca scriverli a te.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

Il port di LibreYOLO segue Bo396543018/Picodet_Pytorch, una
re-implementazione in PyTorch dell'originale PP-PicoDet di PaddleDetection, con
mmcv rimosso e ogni attivazione replicata esattamente, così i checkpoint
PaddlePaddle convertiti attraverso la pipeline di Bo si caricano senza deriva
numerica. Entrambe le fonti portano gli stessi termini Apache-2.0 degli autori
del paper.

</provenance-box>

## Citazione

<citation-block />
