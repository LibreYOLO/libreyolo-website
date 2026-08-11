---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: rilevamento a vocabolario aperto e a punti'
description: >-
  Usa LocateAnything in LibreYOLO per il rilevamento a vocabolario aperto e a
  punti. Fai predizioni con qualsiasi etichetta testuale; addestramento,
  validazione ed esportazione non sono supportati.
lead: >-
  LocateAnything è un modello di grounding visione-linguaggio pubblicato da
  NVIDIA che decodifica bounding box e punti in parallelo, invece di procedere
  un token di coordinata alla volta. LibreYOLO lo avvolge come rilevatore e
  localizzatore a punti a vocabolario aperto: qualsiasi lista di etichette
  testuali diventa l'insieme delle classi, senza testa fissa e senza bisogno di
  fine-tuning.
keywords:
  - LocateAnything
  - NVIDIA
  - vision-language model
  - open-vocabulary detection
  - point detection
  - rilevare oggetti senza addestramento
  - VLM
  - grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Vocabolario aperto: vale qualsiasi parola, non una testa di classi

        # fissa. Persiste in ogni predict()/track() successivo finché non lo
        reimposti.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt a punti
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" restituisce un punto per ogni oggetto trovato, non un
        box.

        # Cambia task su un modello già caricato con model.set_task("point").

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Chat diretta
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # La via di fuga sotto la comodità del rilevamento: domande libere,
        # conteggi o qualsiasi prompt che il wrapper dei box non copre.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Installazione

LocateAnything richiede l'extra `vlm`, che porta con sé `transformers` insieme
ai pacchetti `decord`, `lmdb` e `peft` che il suo codice remoto su Hugging Face
importa al caricamento.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

`LibreLocateAnything` è una classe Python, non un checkpoint `.pt`: non si
carica attraverso la factory `LibreYOLO()` e la CLI `libreyolo` non lo risolve.
Anche la factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) raggiunge
questa famiglia tramite alias, per esempio `LibreVLM("locate-anything")`; la
classe usata qui sotto è quella che costruisce. Caricarla scarica ed esegue il
codice remoto del modello pubblicato da NVIDIA su Hugging Face, quindi LibreYOLO
fissa il download a una revisione di commit precisa invece che al ramo mutabile
`main`, e registra un avviso di licenza una tantum prima del primo download.

<code-tabs name="predict" />

`result.boxes` (task `detect`) e `result.points` (task `point`) portano
l'output parsato come in qualsiasi altra famiglia. La confidenza è un
segnaposto: LocateAnything non emette nessun punteggio per box, quindi ogni
rilevamento riceve la stessa confidenza costante, e `conf=` scarta soltanto le
righe sotto quella costante, non le ordina. Se salti `set_classes()`, il
vocabolario ricade sui nomi di COCO-80. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Una sola dimensione pubblicata, 3b. Due task condividono gli stessi pesi:
`detect` (quello predefinito) restituisce i box, mentre `task="point"`
restituisce invece un singolo punto per ogni oggetto trovato, in
`result.points`; si passa dall'uno all'altro su un modello già caricato con
`model.set_task("point")`. L'harness di benchmark di LibreYOLO non ha misurato
questa famiglia, quindi non ci sono numeri di accuratezza pubblicati con cui
confrontarla.

LibreYOLO espone questa famiglia solo per la predizione. `train()`, `val()` ed
`export()` sollevano tutti `NotImplementedError`: fai fine-tuning upstream e
carica il risultato, la validazione su dataset è saltata perché una confidenza
segnaposto renderebbe fuorviante il mAP di COCO, e l'esportazione è fuori
ambito per un modello generativo senza state dict da tracciare.

## Licenze

<provenance-box>

La NVIDIA License permette l'uso, la riproduzione e la modifica, ma limita il
modello e qualsiasi suo derivato al solo uso non commerciale, di ricerca o di
valutazione, per chiunque non sia NVIDIA e le sue affiliate: non c'è nessuna
soglia di fatturato né eccezione a pagamento. LocateAnything-3B inoltre mette
insieme altri due componenti con licenza propria: un backbone linguistico
Qwen2.5-3B-Instruct sotto la Qwen Research License e un encoder visivo
MoonViT-SO-400M sotto MIT. LibreYOLO non ospita, non replica e non
ridistribuisce niente di tutto questo: `LibreLocateAnything` scarica i pesi e
il codice remoto necessario direttamente da `nvidia/LocateAnything-3B` su
Hugging Face, fissati a un commit preciso, la prima volta che viene eseguito.

</provenance-box>

## Citazione

<citation-block />
