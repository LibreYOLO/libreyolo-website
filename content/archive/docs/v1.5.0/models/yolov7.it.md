---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 in LibreYOLO: predizione, addestramento ed esportazione sotto MIT'
description: >-
  Usa YOLOv7 in LibreYOLO per il rilevamento di oggetti: installa, fai
  predizioni, addestra, valida ed esporta, con codice e pesi sotto licenza MIT.
lead: >-
  YOLOv7 è un rilevatore single-stage basato su anchor, la cui testa aggiunge
  offset di conoscenza implicita appresi prima della convoluzione finale.
  LibreYOLO ne supporta l'unica dimensione pubblicata per il rilevamento.
keywords:
  - YOLOv7
  - object detection
  - yolov7 python
  - addestrare yolov7
  - rilevamento oggetti tempo reale
  - conoscenza implicita
  - ImplicitA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Partenza a caldo da un modello nuovo
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True carica sempre il checkpoint pubblicato LibreYOLO7b.pt,
        # indipendentemente da come è stata costruita questa istanza. Costruire
        # la classe direttamente, invece che tramite LibreYOLO(), parte senza
        # nessun peso caricato.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Installazione

YOLOv7 non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` imposta la
soglia di confidenza e `iou` la soglia di NMS applicata dopo la decodifica della
testa basata su anchor. Vedi [predizione](/docs/predict) per sorgenti, streaming
e gestione dei risultati.

## Varianti

LibreYOLO include una sola dimensione, `b`. Upstream pubblica un unico modello
YOLOv7, quindi non c'è nessuna dimensione tra cui scegliere.

## Addestramento

<code-tabs name="train" />

`pretrained` viene letto davvero, a differenza dell'argomento omonimo che non fa
niente in alcune altre famiglie qui presenti: passa `True` per partire a caldo
dal checkpoint pubblicato `LibreYOLO7b.pt` (scaricato automaticamente), oppure
un percorso o un nome per qualsiasi altra cosa. Quel checkpoint pubblicato è
COCO a 80 classi, quindi richiederlo su un modello già ricostruito per un numero
di classi diverso prima lo ricostruisce di nuovo a 80, lo carica e poi
trasferisce ogni tensore con forma corrispondente nel numero di classi della
testa di destinazione, una volta letto il numero di classi del dataset.
`resume=True` non si può combinare con `pretrained`. Lasciato al valore
predefinito `None`, l'addestramento continua da ciò con cui il modello è stato
costruito, o da un'inizializzazione casuale se non è stato caricato niente.

Se non tocchi altro, il trainer esegue 300 epoche con `lr0=0.01` e SGD con
momentum 0.937, un warmup di 3 epoche, e la stessa assegnazione SimOTA e la
stessa fase finale di 15 epoche senza data augmentation che usa YOLOX, adattate
alla testa basata su anchor. L'unica differenza: durante quelle epoche finali
YOLOX aggiunge un raffinamento L1 della regressione dei box che v7 salta, perché
la loss SimOTA di v7 non porta un ramo L1 sugli offset grezzi da raffinare.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in un
runtime nudo, senza LibreYOLO installato, ma allora il preprocessing e il
postprocessing sono a carico tuo.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
