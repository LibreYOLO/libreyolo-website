---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): rilevamento di oggetti in LibreYOLO'
description: >-
  Esegui SSD300 in LibreYOLO: un rilevatore single-shot basato su VGG16 per
  predizione, validazione ed esportazione in ONNX con licenza BSD-3-Clause.
  Nessun percorso di addestramento.
lead: >-
  SSD (Single Shot MultiBox Detector) predice ogni box e ogni punteggio di
  classe da una griglia densa di box predefiniti in un solo passaggio in avanti,
  senza uno stadio separato di proposta delle regioni. LibreYOLO include il
  checkpoint SSD300 basato su VGG16 come rilevatore solo per inferenza.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - object detection python
  - rilevamento oggetti python
  - VGG16
  - detector single-shot
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgsz è omesso di proposito: SSD300 viene tracciato sul canvas nativo
        # del suo checkpoint e qualsiasi altro valore solleva un errore prima
        # che l'esportazione inizi.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Installazione

SSD non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. SSD decodifica la
sua griglia di box predefiniti con punteggi per classe e poi esegue la
non-maximum suppression, quindi qui `conf`, `iou` e `max_det` hanno tutti un
effetto reale, a differenza dei rilevatori basati su query presenti in questa
libreria. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Varianti

SSD include un solo checkpoint: la rete SSD300 basata su VGG16 al suo canvas
nativo fisso. In questa famiglia non c'è nessuna scelta di dimensione o scala;
predizione, validazione ed esportazione usano tutte quell'unico grafo.

Il file di pesi è `LibreSSD300.pt`, il prefisso della famiglia seguito dalla sua
unica chiave di dimensione, `"300"`. La classe sottostante è `LibreSSD`, quindi
una costruzione diretta è `LibreSSD(size="300")` e non una classe che prende il
nome dal file.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

SSD esporta solo in ONNX; ogni altro formato è al momento bloccato per questa
famiglia. L'esportazione usa sempre il canvas nativo del checkpoint e il grafo
espone la testa grezza compattata di SSD invece di un output con
non-maximum-suppression fusa, quindi `nms=True` non viene accettato quando
esporti. I backend di LibreYOLO eseguono il passaggio di decode e soppressione
dopo aver ricaricato il grafo.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box>

Il codice SSD300 di LibreYOLO non è portato dalla release Caffe degli autori
dell'articolo; deriva dall'implementazione SSD300 di torchvision, con licenza
BSD-3-Clause, ed è quello il repository indicato sopra come sorgente upstream. I
pesi VGG16 del backbone risalgono a loro volta alla VGGNet ridotta e
completamente convoluzionale di Oxford, distribuita con licenza CC BY 4.0 da
Karen Simonyan e Andrew Zisserman.

</provenance-box>

## Citazione

<citation-block />
