---
title: DINO-DETR
families: [dinodetr]
seo_title: "DINO-DETR: predizione ed esportazione con licenza Apache-2.0"
description: "Esegui DINO-DETR in LibreYOLO per il rilevamento di oggetti. Installa, fai predizioni, valida ed esporta tre dimensioni con ancore di denoising, tutte con licenza Apache-2.0."
lead: "DINO-DETR, pubblicato da IDEA Research come DINO, unisce l'addestramento con denoising contrastivo alla mixed query selection, costruita sopra l'attenzione sparsa di Deformable DETR. LibreYOLO distribuisce tre dimensioni per il rilevamento di oggetti, solo in inferenza."
keywords: [DINO-DETR, DINO, detection transformer, denoising anchor boxes, mixed query selection, object detection python, rilevamento oggetti python, IDEA Research]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDINODETRr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() restituisce un semplice dict, non un oggetto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800 half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Installazione

DINO-DETR non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base, e usa lo stesso core di attenzione deformabile
multiscala in PyTorch puro della famiglia Deformable DETR di LibreYOLO.

```bash
pip install libreyolo
```

Installare `libreyolo[hub-kernels]` è opzionale. Quando il pacchetto `kernels`
è presente, LibreYOLO scarica a runtime un kernel compilato di attenzione
deformabile multiscala dall'Hugging Face Hub e lo usa al posto del core in
PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` lo disattiva di nuovo.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituiscono tutte le famiglie,
quindi passare a un altro detector è una modifica di una riga. `conf` e
`max_det` filtrano la selezione delle query; `iou` è accettato per parità di
API ma non ha effetto, perché il decoder è un predittore di insiemi senza
passaggio di NMS. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

In LibreYOLO DINO-DETR è solo per l'inferenza. Upstream lo addestra con
denoising contrastivo e matching ungherese; quella ricetta non è implementata
qui, quindi `train()` solleva `NotImplementedError`.

## Varianti

Tre checkpoint, tutti alla stessa risoluzione di input. `r50` e `r50s5`
condividono un backbone ResNet-50 e differiscono per quante scale di feature
map alimentano il decoder, quattro contro cinque. `swinl` sostituisce il
backbone con Swin-L e campiona anch'esso cinque scale.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

I tre checkpoint ufficiali provengono dalla cartella di rilascio su Google
Drive degli autori, non da una model card di Hugging Face. Il repository
upstream dichiara Apache-2.0 a livello di repository, ma non allega ai
checkpoint stessi né un file di licenza né metadati di licenza, quindi la base
per la ridistribuzione è quella dichiarazione a livello di repository e non una
concessione specifica per i checkpoint. Tutti i mirror di LibreYOLO includono
il testo letterale della licenza Apache-2.0 di upstream insieme a un avviso che
lo spiega.

</provenance-box>

## Citazione

<citation-block />
