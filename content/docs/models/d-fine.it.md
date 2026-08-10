---
title: D-FINE
families: [dfine]
seo_title: "D-FINE: fai fine-tuning, valida ed esporta con licenza MIT"
description: "Usa D-FINE in LibreYOLO per il rilevamento di oggetti e la segmentazione di istanze. Installa, fai predizioni, fine-tuning, validazione ed esportazione, con codice sotto licenza MIT."
lead: "Un detection transformer che riformula la regressione dei box come una distribuzione di probabilità su ogni bordo del box, raffinata attraverso i layer del decoder. LibreYOLO lo supporta per il rilevamento e per la segmentazione di istanze."
keywords: [D-FINE, detection transformer, real-time object detection, instance segmentation, "rilevamento oggetti in tempo reale", "segmentazione di istanze python", DETR]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -seg nel nome del file seleziona la testa delle maschere,
        # quindi qui non serve l'argomento task.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentazione di istanze
      language: bash
      code: |
        # Continua dai pesi di segmentazione pubblicati, testa delle maschere inclusa.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentazione dai pesi di rilevamento
      language: bash
      code: |
        # I pesi di rilevamento non portano una testa delle maschere, quindi
        # questo è un trasferimento esplicito: la testa parte non addestrata ed
        # è utile solo una volta addestrata. Chiedere task=segment qui è ciò
        # che autorizza il trasferimento.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maschere
        print(metrics["metrics/mAP50-95(B)"])   # box
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640
        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640 half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Installazione

D-FINE non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

Il fine-tuning con adattatori tramite `lora=True` è l'eccezione, e richiede
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. Un nome di file con
`-seg` risolve da solo al task di segmentazione, e `result.masks` porta allora
le maschere di istanza insieme ai box. `conf` e `max_det` filtrano la selezione
delle query; `iou` è accettato per parità di API ma non ha effetto, perché il
decoder è un predittore di insiemi senza passaggio di NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Cinque dimensioni. Girano tutte alla stessa risoluzione di input, quindi la
tabella le separa per numero di parametri e accuratezza.

<benchmark-table task="detect" />

<va-embed />

La segmentazione riutilizza il backbone, l'encoder e il decoder del rilevamento
e aggiunge una testa per le maschere, quindi un checkpoint `-seg` accetta gli
stessi argomenti del suo equivalente di rilevamento. La famiglia RT-DETRv4 di
LibreYOLO è scritta come sottoclasse del wrapper di D-FINE: eredita questa linea
di decoder e poi riporta la sua lista di task al solo rilevamento, perché non
porta una testa delle maschere.

## Addestramento

L'addestramento parte da un checkpoint pubblicato, per entrambi i task.

<code-tabs name="train" />

Se non tocchi niente, il trainer esegue 132 epoche con `lr0=2e-4` e `amp=False`,
un batch di 16 ed early stopping dopo 50 epoche senza miglioramenti. I pesi di
rilevamento sono un punto di partenza lecito per l'addestramento di
segmentazione, ma solo come trasferimento esplicito, dato che la testa delle
maschere parte non addestrata e altrimenti restituirebbe maschere prive di
senso. Passare `task=segment` alla CLI è ciò che lo autorizza. La via Python è
più stretta: `LibreDFINE` va costruito direttamente con
`allow_detect_to_segment_transfer=True`, perché la factory `LibreYOLO()` non
accetta un argomento del genere, e la costruzione diretta non scarica niente,
quindi il file dei pesi deve già essere su disco.

`lora=True` si applica al rilevamento. L'addestramento di segmentazione lo
rifiuta e rimanda invece a `freeze='backbone'`, perché la testa delle maschere
non è stata testata con gli adattatori. Su Apple silicon il trainer sposta
l'intera esecuzione su CPU: il backward pass del matmul a bin dell'Integral
incontra un errore di compilazione Metal. L'inferenza su MPS non ne risente.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario indicizzato per nome della metrica, e stampa i
risultati per classe se `verbose` resta attivo.

<code-tabs name="val" />

Su un checkpoint `-seg` la chiave `metrics/mAP50-95` da sola contiene il
punteggio delle maschere, e la stessa esecuzione riporta anche i box sotto `(B)`
e le maschere sotto `(M)`, così entrambi sono disponibili da una sola passata.

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. I percorsi OpenVINO, Paddle, MNN e Core AI
esportano a canvas fisso invece che con forme dinamiche.
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta e gli
extra che qualcuno di essi aggiunge.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

I pesi di segmentazione hanno un secondo upstream: il loro decoder delle
maschere, il matching delle maschere e la loss delle maschere vengono da
ArgoHA/D-FINE-seg, anch'esso Apache-2.0, il cui maintainer ha approvato il
riutilizzo con attribuzione.

</provenance-box>

## Citazione

<citation-block />
