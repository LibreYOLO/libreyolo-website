---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: rilevamento e riconoscimento del testo in LibreYOLO'
description: >-
  Usa PP-OCRv5 in LibreYOLO per l'OCR multilingua di testo in scena. Installa,
  fai predizioni e valida i checkpoint t e l, con licenza Apache-2.0.
lead: >-
  PP-OCRv5 è la pipeline di rilevamento e riconoscimento del testo di PaddleOCR:
  un rilevatore a binarizzazione differenziabile individua i quadrilateri di
  testo e un riconoscitore SVTR/CTC li legge. LibreYOLO la porta su PyTorch in
  due livelli.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - text detection
  - text recognition
  - scene text
  - ocr python
  - riconoscimento testo immagini
  - estrarre testo da immagini python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Quadrilateri
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRl-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Poligoni (N, 4, 2) in ordine di lettura: alto-sinistra, alto-destra,

        # basso-destra, basso-sinistra. I quadrilateri di rilevamento sono veri

        # poligoni (testo ruotato), quindi popolano result.ocr, non
        result.boxes.

        print(result.ocr.data.shape)

        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # metrica principale
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Installazione

PP-OCRv5 non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache
in locale.

<code-tabs name="predict" />

Ogni checkpoint racchiude entrambi gli stadi, rilevamento e riconoscimento, in
un unico file `.pt`, con il charset di riconoscimento e i valori predefiniti
della pipeline conservati nei metadati del checkpoint. Il riconoscitore legge
cinese semplificato e tradizionale, inglese, giapponese e pinyin con un unico
dizionario. `result.ocr` è un payload `OCRRegions`: `.data` contiene i poligoni
a quattro punti, `.texts` le trascrizioni, `.conf` il punteggio di
riconoscimento per regione e `.det_conf` il punteggio di rilevamento. Le
sorgenti multi-immagine vengono eseguite in sequenza: la pipeline a due stadi
non raggruppa in batch fra immagini diverse. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Due livelli: `t`, costruito sui backbone più leggeri PP-LCNetV3/PP-OCRv5_mobile
per l'uso su CPU, e `l`, costruito sui backbone server PP-HGNetV2 per
un'accuratezza maggiore. Entrambi i livelli eseguono il rilevamento con un
limite fisso sul lato lungo e riconoscono i ritagli in batch; `rec_batch`
controlla quanti ritagli passano nel riconoscitore a ogni forward pass.

## Validazione

`val()` misura la pipeline su una directory di immagini più un file
`labels/<split>.jsonl`, oppure sull'equivalente YAML del dataset, dove ogni
etichetta elenca i poligoni delle regioni di testo per immagine e le loro
trascrizioni. Riporta l'hmean di rilevamento (precisione/recall/F1 con match su
IoU), l'F1 end-to-end (l'hmean più una corrispondenza esatta della trascrizione
dopo normalizzazione, la metrica di fitness del checkpoint) e 1-NED, la
distanza di edit normalizzata media sulle coppie corrispondenti.

<code-tabs name="val" />

## Esportazione

<export-matrix />

PP-OCRv5 è una pipeline a due reti, rilevamento e riconoscimento che si muovono
insieme, non un unico grafo tracciabile, e l'esportazione non è implementata:
nessun formato è ancora supportato. Fai fine-tuning direttamente sul codice di
addestramento upstream con licenza Apache-2.0 e converti il risultato con
`weights/convert_ppocr_weights.py` se ti serve un checkpoint fuori da questo
formato.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
