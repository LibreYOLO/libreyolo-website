---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: predici la segmentazione semantica, di istanze e panottica'
description: >-
  Usa EoMT in LibreYOLO per la segmentazione semantica, di istanze e panottica
  su un vision transformer DINOv2 semplice, senza bisogno di decoder. Con
  licenza MIT.
lead: >-
  Una rete di segmentazione costruita su un vision transformer semplice, senza
  un decoder di pixel dedicato: sono delle query apprese aggiuntive, inserite
  nell'encoder stesso, a predire le maschere. LibreYOLO la supporta per la
  segmentazione semantica, di istanze e panottica.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - panoptic segmentation python
  - segmentazione panottica
  - segmentazione di istanze python
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantica
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreEoMTl-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id delle classi

        print(mask.classes)      # id delle classi presenti nell'immagine,
        ordinati
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -seg nel nome del file seleziona il task di istanze,
        # quindi qui non serve nessun argomento task.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panottica
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) id dei segmenti
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Semantica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maschere
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: Panottica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory smista in base al suffisso del file, quindi un artefatto

        # esportato si carica come un checkpoint e restituisce lo stesso
        Results.

        model = LibreYOLO("LibreEoMTl-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Installazione

EoMT non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale. Il suffisso di task nel nome del file (`-sem`, `-seg`,
`-panoptic`) seleziona il task, e `LibreYOLO()` lo deduce da quel nome, quindi
non serve nessun argomento `task=`.

<code-tabs name="predict" />

La segmentazione semantica popola `result.semantic_mask`, un array `(H, W)` di
id di classe in `.data`. La segmentazione di istanze popola `result.boxes` e
`result.masks`, la stessa forma che restituisce ogni altra famiglia di
segmentazione. La segmentazione panottica popola `result.panoptic`: una mappa
`(H, W)` di id di segmento in `.data`, più `.segments_info`, una lista di dict
`{"id", "category_id"}`, uno per segmento. `conf` filtra la selezione delle
query; `iou` non ha alcun effetto sul task semantico, perché assegna una classe
a ogni pixel per argmax, senza passo di NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre dimensioni di encoder, s/b/l, tutte basate su DINOv2. Il checkpoint
semantico è addestrato su ADE20K a 512 px; i checkpoint di istanze e panottico
sono addestrati su COCO a 640 px, con un secondo checkpoint di istanze
addestrato a 1280 px. Upstream distribuisce i pesi DINOv2 per la segmentazione
di istanze solo alla dimensione l; s e b sono pubblicati solo per il semantico
e il panottico. Le varianti di EoMT basate su DINOv3 esistono upstream ma non
sono distribuite qui, perché dipendono da pesi DINOv3 ad accesso limitato e non
commerciali.

LibreYOLO non addestra EoMT: `train()` solleva `NotImplementedError` per questa
famiglia, cosa che il [livello di supporto](/docs/models) qui sopra segnala
come solo inferenza.

## Validazione

`val()` smista in base al task. Il semantico restituisce `metrics/mIoU` e
`metrics/pixel_accuracy`. La segmentazione di istanze restituisce le stesse
chiavi di mAP per maschere e box delle altre famiglie di segmentazione. Il
panottico restituisce la Panoptic Quality come `metrics/PQ`, suddivisa in
`metrics/SQ` (segmentation quality) e `metrics/RQ` (recognition quality), più
`metrics/PQ_things` e `metrics/PQ_stuff`.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Oggi si esporta solo il task semantico: la segmentazione di istanze e quella
panottica chiamano `export()` e ottengono `NotImplementedError`, perché il loro
output di maschere da query non ha ancora un contratto di esportazione a
runtime. Un artefatto semantico esportato si ricarica con `LibreYOLO()` in base
al suffisso del file, quindi un file `.onnx` o `.engine` si comporta come un
checkpoint e restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
