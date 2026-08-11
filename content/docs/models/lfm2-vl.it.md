---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: rilevamento a vocabolario aperto in LibreYOLO'
description: >-
  Usa LFM2-VL in LibreYOLO per il rilevamento di oggetti a vocabolario aperto
  on-device. Fai predizioni con qualsiasi etichetta testuale; addestramento,
  validazione ed esportazione non sono supportati.
lead: >-
  LFM2-VL è un modello vision-language compatto e on-device pubblicato da Liquid
  AI. LibreYOLO lo avvolge come rilevatore di oggetti a vocabolario aperto:
  qualsiasi lista di etichette testuali diventa l'insieme delle classi, senza
  testa fissa e senza bisogno di fine-tuning.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - vision-language model
  - open-vocabulary detection
  - rilevare oggetti con il testo
  - VLM on-device
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


        # Vocabolario aperto: vale qualsiasi parola, non una testa di classi

        # fissa. Persiste in ogni predict()/track() successivo finché non lo
        reimposti.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat diretta
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # La via di fuga sotto la comodità del rilevamento: domande libere,
        # conteggi o qualsiasi prompt che il wrapper dei box non copre.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Installazione

LFM2-VL richiede l'extra `vlm`, che porta con sé `transformers` per il
backbone a template di chat.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

`LibreLFM2VL` è una classe Python, non un checkpoint `.pt`: non si carica
attraverso la factory `LibreYOLO()` e la CLI `libreyolo` non lo risolve. Anche
la factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) raggiunge questa
famiglia tramite alias, per esempio `LibreVLM("lfm2-vl-450m")`; la classe usata
qui sotto è quella che costruisce. I pesi arrivano dal repository Hugging Face
di Liquid AI, non da un mirror di LibreYOLO; la prima chiamata li scarica e li
mette in cache in locale, e prima di farlo registra un avviso di licenza una
tantum.

<code-tabs name="predict" />

`result.boxes` porta i rilevamenti parsati come in qualsiasi altra famiglia. La
confidenza è un segnaposto: LFM2-VL non emette nessun punteggio per box,
quindi ogni rilevamento riceve la stessa confidenza costante, e `conf=` scarta
soltanto le righe sotto quella costante, non le ordina. `iou` scarta i box
quasi duplicati della stessa classe oltre la sovrapposizione indicata, un
effetto collaterale del decoding greedy che ripete un oggetto; non è una
passata di NMS per classi. Se salti `set_classes()`, il vocabolario ricade sui
nomi di COCO-80. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Due dimensioni: 450m e 1.6b, entrambe dalla release LFM2.5-VL di Liquid AI,
costruite per il deployment on-device. L'harness di benchmark di LibreYOLO non
ha misurato questa famiglia, quindi non ci sono numeri di accuratezza
pubblicati con cui confrontarle; scegli una dimensione in base al tuo budget di
calcolo.

LibreYOLO espone questa famiglia solo per la predizione. `train()`, `val()` ed
`export()` sollevano tutti `NotImplementedError`: fai fine-tuning upstream e
carica il risultato, la validazione su dataset è saltata perché una confidenza
segnaposto renderebbe fuorviante il mAP di COCO, e l'esportazione è fuori
ambito per un modello generativo senza state dict da tracciare.

## Licenze

<provenance-box>

La LFM Open License v1.0 permette uso commerciale, riproduzione e modifica, ma
solo sotto una soglia di 10 milioni di dollari di ricavi annui; un'entità
giuridica pari o superiore a quella soglia non è affatto licenziata da questo
accordo per l'uso commerciale, e deve contattare direttamente Liquid AI. Le
organizzazioni non profit qualificate sono esentate dalla soglia per l'uso non
commerciale o di ricerca. LibreYOLO non distribuisce nessun codice sorgente di
LiquidAI, dato che il modello si carica attraverso la libreria `transformers`
Apache-2.0, e non ospita né ridistribuisce i pesi: `LibreLFM2VL` scarica la
dimensione corrispondente direttamente dal repository Hugging Face di Liquid AI
la prima volta che viene eseguito, e registra un avviso una tantum prima di
quel download.

</provenance-box>

## Citazione

<citation-block />
