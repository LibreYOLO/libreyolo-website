---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: localizzazione a punti, addestramento ed esportazione in LibreYOLO'
description: >-
  Esegui FOMO (Faster Objects, More Objects) in LibreYOLO: un rilevatore
  minuscolo a localizzazione di punti per contare tanti oggetti piccoli.
  Installazione, predizione, addestramento ed esportazione.
lead: >-
  FOMO è un localizzatore di punti basato su griglia: ogni cella di una griglia
  a bassa risoluzione viene classificata come sfondo o come centro di un
  oggetto, senza nessuna regressione dei bounding box. LibreYOLO lo supporta per
  il task point.
keywords:
  - FOMO
  - Faster Objects More Objects
  - point localization
  - rilevamento oggetti piccoli
  - contare oggetti in un'immagine python
  - tiny object detection
  - edge AI
  - rilevamento su microcontrollore
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # I pesi di LibreFOMO non si scaricano da soli (vedi Checkpoint sotto).
        # Puntalo a un checkpoint che hai già scaricato in locale.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz va passato: la CLI lo imposta a 640 di default, e il

        # checkpoint s accetta solo i suoi 96 nativi.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Installazione

FOMO non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

A differenza di ogni altra famiglia di questo sito, i pesi di LibreFOMO non si
scaricano da soli: `LibreYOLO("LibreFOMOs-point.pt")` cerca quel file su disco
e solleva un `ValueError` che lo nomina, invece di prenderlo da Hugging Face.
Scarica prima un checkpoint dall'[organizzazione LibreYOLO](https://huggingface.co/LibreYOLO)
e caricalo indicando il percorso locale, oppure addestrane uno tuo (vedi
Addestramento sotto).

<code-tabs name="predict" />

Il risultato porta un payload `points` invece di `boxes`: ogni riga è
`x, y, classe, confidenza`, disponibile come `result.points.data` oppure
attraverso gli accessor `.xy`, `.xyn`, `.cls` e `.conf`. Non c'è nessuna soglia
`iou` da impostare, perché non ci sono box da sopprimere; `predict(...,
nms_radius=1)` controlla di quante celle della griglia devono essere distanti
due rilevamenti perché sopravvivano entrambi, e il nome del file deve portare
il suffisso di task `-point` di FOMO perché il loader lo riconosca. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre taglie, `s`, `m` e `l`, usano backbone in stile MobileNetV2 via via più
larghi, a risoluzioni di input fisse corrispondentemente maggiori, ciascuno
dietro un'unica testa di classificazione 1x1. Questa famiglia qui non ha
nessuna tabella di benchmark; la dimensione del file del checkpoint nella
tabella sotto è il segnale per taglia più chiaro tra quelli pubblicati finora.

## Addestramento

<code-tabs name="train" />

`imgsz` non è una scelta libera: di default vale la risoluzione nativa del
checkpoint caricato, e passare un valore diverso solleva un `ValueError` che
nomina la dimensione attesa. Quelle dimensioni sono 96 per `s`, 192 per `m` e
224 per `l`. La CLI imposta `imgsz` a 640 di default, quindi un comando
`libreyolo train` deve impostarlo esplicitamente perché corrisponda al
checkpoint.

Se non tocchi altro, il trainer esegue 40 epoche con batch 32 usando Adam a
`lr0=3e-4`, senza weight decay e con la classe di primo piano pesata 100x
rispetto allo sfondo nella loss di cross-entropy per cella, dato che in una
scena tipica quasi ogni cella della griglia è sfondo. EMA e precisione mista
sono entrambe disattivate di default, e non viene applicata nessuna delle
augmentation geometriche o di colore usate altrove in LibreYOLO: mosaic, mixup,
jitter HSV, flip, rotazione, traslazione e shear sono tutti a zero.

È il percorso con cui sono stati addestrati i checkpoint LibreFOMO pubblicati,
da zero su COCO.

Vedi [addestramento](/docs/train) per dataset e logger.

## Validazione

`val()` instrada verso un validatore a livello di griglia costruito per questa
famiglia. Oltre alle chiavi `metrics/precision`, `metrics/recall` e
`metrics/mAP@` basate sull'abbinamento dei punti e condivise con gli altri task
point, esplora soglie di confidenza e valori di `nms_radius` e pubblica la
combinazione con l'F1 migliore sotto `metrics/grid_F1`,
`metrics/grid_precision`, `metrics/grid_recall` e
`metrics/grid_mean_distance`, più la soglia e il raggio che l'hanno prodotta
sotto `decode/threshold` e `decode/nms_radius`.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica attraverso `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in
un runtime nudo, senza LibreYOLO installato, ma allora il preprocessing e il
postprocessing tocca scriverli a te.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia. Nessuno di questi si
scarica automaticamente: prendi il file che ti serve dalla pagina Hugging Face
collegata e passa il suo percorso locale a `LibreYOLO()`.

<checkpoint-table />

## Licenza

<provenance-box>

Non c'è nessun repository di codice upstream di FOMO da collegare: Edge Impulse
descrive la tecnica in un post del blog e nella documentazione del suo
prodotto, ma non ha rilasciato il codice di addestramento o di inferenza di
FOMO. L'architettura e l'addestramento presenti qui sono l'implementazione che
LibreYOLO ha fatto di quella descrizione pubblicata, e i checkpoint LibreFOMO
pubblicati sono addestrati da zero su COCO, quindi sia il codice sia questi
pesi sono MIT, di LibreYOLO. Il nome FOMO e la tecnica che descrive restano di
Edge Impulse.

</provenance-box>
