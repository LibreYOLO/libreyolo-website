---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS in LibreYOLO: analisi di immagini any-to-any'
description: >-
  Usa LibreMODUS in LibreYOLO per profondità, normali, contorni e rilevamento, e
  per comporli con any2any(). Solo inferenza; i pesi si caricano da EPFL-VILAB.
lead: >-
  LibreMODUS è un'integrazione solo inferenza del checkpoint MODUS 14B-A7B, un
  modello any-to-any che trasforma un input derivato da un'immagine in un altro:
  RGB in ingresso, profondità in uscita; profondità in ingresso, normali in
  uscita; una qualsiasi di queste più una frase, box in uscita. LibreYOLO
  supporta quattro task attraverso l'API predict standard e un insieme più ampio
  tramite any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - depth estimation
  - stima della profondità da immagine
  - surface normals
  - normali alla superficie
  - edge detection
  - referring detection
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Senza un vocabolario personalizzato, detect decodifica i token di
        # etichetta COCO del checkpoint in class id COCO-80 contigui.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Phrase grounding
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() porta il rilevamento al phrase grounding: ogni frase
        # gira in modo indipendente e torna con lo stesso contratto Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # Da uno a tre input derivati da immagini (rgb, depth, normal,
        canny/edge),

        # più un testo ausiliario opzionale, composti verso un solo target.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Il grounding tramite any2any() richiede un input di testo con la
        frase.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Installazione

LibreMODUS richiede un extra dedicato, che porta con sé `accelerate` per il dispatch dei modelli grandi di cui questo checkpoint ha bisogno.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO non ridistribuisce né fa da mirror ai pesi di MODUS. Per impostazione predefinita, il caricamento di un modello `LibreMODUS` scarica i file necessari direttamente da `EPFL-VILAB/MODUS` a una revisione Hugging Face fissata, e un download nuovo richiede sempre un account Hugging Face autenticato dell'utente, anche se il gate di hosting upstream è temporaneamente aperto. Leggi e accetta i termini upstream, poi autenticati:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Per evitare qualsiasi richiesta di rete, punta a uno snapshot che hai già:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Quella directory deve contenere `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` e `merges.txt`. Vedi Licenze più sotto per cosa permettono i termini del checkpoint.

## Predizione

<code-tabs name="predict" />

L'API standard dei task copre quattro task, ognuno mappato su un target MODUS: `depth` sulla profondità relativa (`result.depth_map`), `normal` sulle normali alla superficie (`result.normal_map`), `edge` sui contorni in stile Canny (`result.edges`) e `detect` sui box COCO-80 (`result.boxes`), a meno che `set_classes()` non lo porti al phrase grounding. `set_task()` passa dall'uno all'altro sullo stesso modello già caricato. La ricetta rilasciata usa dieci passi di flow sampling con text guidance 4.0 e image guidance 2.0; puoi sovrascriverli con `inference_steps=`, `inference_cfg=` e `inference_image_cfg=` alla costruzione.

`any2any()` raggiunge la superficie di analisi pubblica più ampia: da uno a tre input derivati da immagini (`rgb`, `depth`, `normal`, `canny`/`edge`), più un testo ausiliario opzionale, composti verso uno qualsiasi tra profondità, normali, contorni, contorni derivati da SAM, rilevamento COCO o phrase grounding. Tutti gli input derivati da immagini devono descrivere lo stesso canvas allineato; LibreMODUS rifiuta larghezze e altezze che non combaciano invece di ridimensionarle in modo indipendente. `chain=(...)` genera target intermedi e li reimmette nello stesso contesto, entro il budget di tre condizioni con cui il checkpoint è stato addestrato. `verify=N` (N >= 2) genera N candidati e tiene quello che ottiene il punteggio più alto in un controllo vincolato di auto-consistenza, esposto come `result.verification_score`.

`dtype="bf16"` (il valore predefinito) corrisponde alla precisione del checkpoint rilasciato; `dtype="fp8"` memorizza i pesi lineari idonei del tronco del decoder come E4M3 con una scala per canale di output, converte una volta sola in una cache locale sotto `~/.cache/libreyolo/modus/fp8` e dequantizza al dtype dell'input a ogni moltiplicazione di matrici, quindi baratta memoria invece di barattare accuratezza a livello di attivazioni.

`train()`, `val()` ed `export()` sollevano tutti un'eccezione: LibreMODUS è solo inferenza, la validazione su dataset non è offerta e non esiste alcun percorso di esportazione ONNX, TensorRT o TFLite. Nemmeno la `predict()` a batch e la test-time augmentation sono supportate; ogni chiamata gestisce una sola immagine.

## Licenze

<provenance-box>

LibreYOLO non ospita né fa da mirror al checkpoint MODUS da nessuna parte, nemmeno sulla propria org Hugging Face: caricarlo tira sempre la revisione fissata direttamente da EPFL-VILAB/MODUS, oppure legge uno snapshot già presente su disco in `checkpoint_path`.

</provenance-box>

## Citazione

<citation-block />
