---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: rilevamento a vocabolario aperto in LibreYOLO'
description: >-
  Usa InternVL3 in LibreYOLO per il rilevamento di oggetti a vocabolario aperto.
  Fai predizioni con qualsiasi etichetta testuale; addestramento, validazione ed
  esportazione non sono supportati.
lead: >-
  InternVL3 è un large language model multimodale nativo pubblicato da OpenGVLab
  che impara visione e linguaggio insieme in un'unica fase di pre-addestramento.
  LibreYOLO lo avvolge come rilevatore di oggetti a vocabolario aperto:
  qualsiasi lista di etichette testuali diventa l'insieme delle classi, senza
  testa fissa e senza bisogno di fine-tuning.
keywords:
  - InternVL3
  - InternVL
  - vision-language model
  - open-vocabulary detection
  - rilevare oggetti con il testo
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE


        model = LibreInternVL3(size="2b")


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
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # La via di fuga sotto la comodità del rilevamento: domande libere,
        # conteggi o qualsiasi prompt che il wrapper dei box non copre.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Installazione

InternVL3 richiede l'extra `vlm`, che porta con sé `transformers` per il
backbone a template di chat.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

`LibreInternVL3` è una classe Python, non un checkpoint `.pt`: non si carica
attraverso la factory `LibreYOLO()` e la CLI `libreyolo` non lo risolve. Anche
la factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) raggiunge questa
famiglia tramite alias, per esempio `LibreVLM("internvl3-2b")`; la classe usata
qui sotto è quella che costruisce. I pesi arrivano dai repository `-hf` di
Hugging Face di OpenGVLab, non da un mirror di LibreYOLO; la prima chiamata li
scarica e li mette in cache in locale, e prima di farlo registra un avviso di
licenza una tantum per i pesi Qwen, che sono ad accesso limitato.

<code-tabs name="predict" />

`result.boxes` porta i rilevamenti parsati come in qualsiasi altra famiglia. La
confidenza è un segnaposto: InternVL3 non emette nessun punteggio per box,
quindi ogni rilevamento riceve la stessa confidenza costante, e `conf=` scarta
soltanto le righe sotto quella costante, non le ordina. `iou` scarta i box
quasi duplicati della stessa classe oltre la sovrapposizione indicata, un
effetto collaterale del decoding greedy che ripete un oggetto; non è una
passata di NMS per classi. Se salti `set_classes()`, il vocabolario ricade sui
nomi di COCO-80. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Tre dimensioni: 1b, 2b e 8b, tutti checkpoint `-hf` nativi di OpenGVLab (un
backbone LLM Qwen, non l'architettura a due torri descritta dal paper originale
di InternVL). L'harness di benchmark di LibreYOLO non ha misurato questa
famiglia, quindi non ci sono numeri di accuratezza pubblicati con cui
confrontarle; scegli una dimensione in base al tuo budget di calcolo.

LibreYOLO espone questa famiglia solo per la predizione. `train()`, `val()` ed
`export()` sollevano tutti `NotImplementedError`: fai fine-tuning upstream e
carica il risultato, la validazione su dataset è saltata perché una confidenza
segnaposto renderebbe fuorviante il mAP di COCO, e l'esportazione è fuori
ambito per un modello generativo senza state dict da tracciare.

## Licenze

<provenance-box>

Il codice di InternVL3 è MIT, permissivo e utilizzabile in prodotti commerciali
e a codice chiuso. I checkpoint `-hf` che questa famiglia carica portano un
backbone LLM Qwen e sono licenziati a parte, sotto la Qwen License di Alibaba
Cloud: liberi da usare, modificare e ridistribuire con l'obbligo di attribuire
tramite un "Built with Qwen" o "Improved using Qwen", e con un tetto di 100
milioni di utenti attivi mensili nell'uso commerciale, oltre il quale serve
l'autorizzazione di Alibaba stessa. LibreYOLO non ospita né ridistribuisce
questi pesi: `LibreInternVL3` scarica la dimensione corrispondente direttamente
da `OpenGVLab/InternVL3-<size>-hf` su Hugging Face la prima volta che viene
eseguito, e registra un avviso una tantum per la Qwen License prima di quel
download.

</provenance-box>

## Citazione

<citation-block />
