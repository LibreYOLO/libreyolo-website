---
title: OCR
seo_title: 'OCR: rilevamento e riconoscimento del testo in LibreYOLO'
description: >-
  Trova e leggi il testo nelle immagini con LibreYOLO. Predici quadrilateri e
  trascrizioni, etichetta un dataset JSONL e valida con hmean, F1 end-to-end e
  1-NED.
lead: >-
  L'OCR localizza il testo in un'immagine e lo legge. LibreYOLO lo espone come
  task ocr, che restituisce un poligono a quattro punti più una trascrizione per
  ogni regione di testo, in ordine di lettura.
keywords:
  - ocr python
  - riconoscimento testo immagini python
  - scene text recognition
  - PP-OCRv5 python
  - estrarre testo da immagini python
last_verified: 1.5.0
snippets:
  predict:
    - label: Leggere il testo in un'immagine
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Il tier t è il più leggero dei due, pensato per la CPU. SAMPLE_IMAGE

        # mantiene lo snippet eseguibile; puntalo a una tua immagine con del
        testo.

        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(len(regions), "regions")

        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Leggere i quadrilateri
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(regions.data.shape)   # poligoni (N, 4, 2), TL TR BR BL

        print(regions.xyxy)         # inviluppi allineati agli assi di quei
        poligoni

        print(regions.det_conf)     # punteggio di rilevamento, distinto da
        .conf
    - label: Filtrare per confidenza di riconoscimento
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Indicizza con le posizioni, non con una maschera booleana: lo slicing

        # porta con sé le trascrizioni e i due array di punteggi oltre alla
        geometria.

        regions = result.ocr.numpy()

        keep = regions[np.flatnonzero(regions.conf >= 0.9)]

        print(keep.texts)
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definizione

Il task `ocr` fa due cose in una sola chiamata: localizza ogni regione di testo
di un'immagine e la trascrive. Le regioni tornano come poligoni a quattro punti
invece che come box allineati agli assi, perché il testo nelle scene reali è
spesso ruotato, e in ordine di lettura, dall'alto verso il basso e poi da
sinistra a destra.

Una predizione riempie `result.ocr`, un payload `OCRRegions`. `.data` è un
array float `(N, 4, 2)` di poligoni in pixel dell'immagine originale, ordinati
in alto a sinistra, in alto a destra, in basso a destra, in basso a sinistra;
`.texts` è la lista delle N trascrizioni; `.conf` è il punteggio di
riconoscimento per regione e `.det_conf` quello di rilevamento; `.xyxy` dà
l'inviluppo allineato agli assi di ogni poligono. Poiché i quadrilateri sono
poligoni veri e propri, non popolano `result.boxes`. Lo slicing di un
`OCRRegions` porta con sé le trascrizioni e i due array di punteggi oltre alla
geometria.

## Modelli

Due famiglie servono `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) è la pipeline dedicata: un rilevatore a
binarizzazione differenziabile trova i quadrilateri di testo e un riconoscitore
SVTR/CTC li legge, con entrambi gli stadi impacchettati in un unico file `.pt`
insieme al charset di riconoscimento. È disponibile in due tier, uno più leggero
per la CPU e uno server per un'accuratezza più alta, e un solo dizionario copre
cinese semplificato e tradizionale, inglese, giapponese e pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) affronta l'OCR generando le
parole come testo con tag dallo stesso checkpoint da 7B che serve gli altri sei
task, caricato con `LibreVLM("sensenova-vision", task="ocr")`. Richiede l'extra
`sensenova` e i suoi pesi sono limitati all'uso non commerciale; la licenza è
sulla sua pagina.

## Predizione

I pesi si scaricano da Hugging Face al primo uso e restano in cache in locale.

<code-tabs name="predict" />

PP-OCRv5 esegue il rilevamento con un limite fisso sul lato lungo e poi
riconosce a batch le regioni ritagliate, con `rec_batch` che controlla quanti
ritagli passano nel riconoscitore a ogni forward pass. Le sorgenti
multi-immagine vengono elaborate in sequenza, perché una pipeline a due stadi
non fa batch tra immagini diverse. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Formato del dataset

Le etichette OCR sono un file JSONL per split, un oggetto JSON per immagine,
accanto alle immagini stesse.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Ogni riga nomina un'immagine ed elenca le sue regioni:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` è un quadrilatero a quattro punti in coordinate pixel assolute,
ordinate in alto a sinistra, in alto a destra, in basso a destra, in basso a
sinistra. Una regione il cui testo non è leggibile si etichetta con
`"text": "###"`, la convenzione don't-care di ICDAR: viene esclusa dal punteggio
di riconoscimento, e una predizione che vi si sovrappone viene ignorata invece
che contata come falso positivo.

Basta passare la directory radice come `data=`. L'alternativa è uno YAML di
dataset, con `path` più i nomi opzionali delle directory `images` e `labels`, e
`nc: 1` con `names: {0: text}` come segnaposto di schema, dato che un modello
OCR restituisce `Results.ocr` invece di rilevamenti. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto completo.

## Addestramento

Nessuna delle due famiglie OCR ha un'implementazione dell'addestramento:
`train()` solleva `NotImplementedError` su entrambe, e il supporto OCR copre
solo predizione e validazione. La pagina di PP-OCRv5 indica il codice di
addestramento upstream sotto Apache-2.0 e lo script di conversione che riporta
in LibreYOLO un checkpoint sottoposto a fine-tuning.

## Validazione

`val()` valuta l'intera pipeline, rilevamento e riconoscimento insieme,
accoppiando uno a uno i poligoni predetti con quelli del ground truth a un IoU
superiore a 0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` e `metrics/det_hmean` valutano
solo la localizzazione: per una corrispondenza basta la sovrapposizione dei
poligoni, qualunque cosa dica la trascrizione. `metrics/e2e_precision`,
`metrics/e2e_recall` e `metrics/e2e_f1` aggiungono la lettura: serve la stessa
sovrapposizione dei poligoni più una trascrizione identica dopo la
normalizzazione NFKC e la rimozione degli spazi, e il confronto resta sensibile
alle maiuscole. `metrics/e2e_f1` è anche `fitness`, il numero letto dalla
selezione del checkpoint migliore.

`metrics/rec_1-NED` valuta il riconoscitore da solo, sulle coppie che il
rilevamento ha già associato: uno meno la distanza di edit normalizzata, così
una trascrizione sbagliata di un carattere ottiene quasi 1 dove l'F1 end-to-end
le dà 0.

## Esportazione

Per questo task non è disponibile nessun formato di esportazione. PP-OCRv5 è
fatto di due reti che lavorano insieme e non di un unico grafo tracciabile, e
`export()` solleva un errore per ogni formato su entrambe le famiglie. Per
mettere in produzione fuori da LibreYOLO, fai fine-tuning upstream e usa il
percorso di deployment upstream.
