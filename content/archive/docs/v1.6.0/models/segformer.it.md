---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: segmentazione semantica in LibreYOLO'
description: >-
  Usa SegFormer in LibreYOLO per la segmentazione semantica su ADE20K, nelle
  taglie da b0 a b5. Installa, fai predizioni, addestra ed esporta; i pesi
  preaddestrati non sono per uso commerciale.
lead: >-
  SegFormer è un transformer per la segmentazione semantica che unisce un
  encoder gerarchico Mix Transformer (MiT) a una testa di decodifica leggera
  fatta di soli MLP, evitando i decoder pesanti e le codifiche posizionali fisse
  di cui avevano bisogno i transformer di segmentazione precedenti. LibreYOLO lo
  supporta per un solo task, la segmentazione semantica, in sei taglie.
keywords:
  - SegFormer
  - semantic segmentation
  - segmentazione semantica python
  - Mix Transformer
  - MiT
  - transformer segmentazione semantica
  - ADE20K
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (fine-tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Da zero
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # Senza model_path: init casuale, nessun download. L'unica via a pesi
        # liberi dalla clausola non commerciale dei checkpoint preaddestrati.
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come un checkpoint qualsiasi e restituisce lo
        # stesso oggetto Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Installazione

SegFormer non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

`result.semantic_mask` porta la mappa densa delle classi: `.data` è un tensore
`(H, W)` di id di classe alla dimensione originale dell'immagine, e `.classes`
elenca gli id di classe effettivamente presenti. `result.boxes` è `None`,
perché non ci sono rilevamenti per singola istanza. `conf` e `iou` sono
accettati per parità di API ma non cambiano l'output: il modello restituisce
una classe per pixel, non rilevamenti per istanza da filtrare o deduplicare.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Sei taglie, da b0 a b5, che allargano e approfondiscono l'encoder Mix
Transformer a ogni passo mantenendo lo stesso design della testa di decodifica
fatta di soli MLP.

<checkpoint-table />

## Addestramento

Per impostazione predefinita `train()` fa fine-tuning di un checkpoint
pubblicato. Se invece non passi nessun `model_path` a `LibreSegformer(...)`, il
modello viene costruito con encoder e testa inizializzati a caso e addestrato
da zero: è l'unica strada per ottenere pesi che non portano nessuna delle
restrizioni non commerciali dei checkpoint preaddestrati (vedi
[Licenze](#licensing)).

<code-tabs name="train" />

Se lo lasci ai valori predefiniti, il trainer segue la ricetta ADE20K del paper
di SegFormer: AdamW con un learning rate base per il backbone e la testa di
decodifica addestrata a 10x quel valore, weight decay ovunque tranne che sui
LayerNorm e sulla convoluzione posizionale del Mix-FFN, e uno schedule a
decadimento lineare con warmup. La convergenza per le taglie più grandi, da b3
a b5, non è stata validata end to end.

Vedi [addestramento](/docs/train) per dataset, augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/`: mIoU e accuratezza per
pixel, misurate su qualsiasi dataset nel formato con cui hai addestrato.

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

L'encoder e la testa di decodifica di LibreSegformer sono un port in PyTorch
dell'implementazione SegFormer di Hugging Face Transformers, sotto licenza
Apache-2.0, e non di NVlabs/SegFormer: il repository originale di NVIDIA non è
mai stato letto né copiato, ed è citato qui solo per attribuzione agli autori
del paper. Solo i checkpoint preaddestrati qui sopra portano la restrizione
non commerciale di NVIDIA; l'architettura e il codice di LibreYOLO restano MIT
dall'inizio alla fine.

</provenance-box>

## Citazione

<citation-block />
