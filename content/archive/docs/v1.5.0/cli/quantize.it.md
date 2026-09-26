---
title: libreyolo quantize
seo_title: riferimento del comando libreyolo quantize
description: >-
  Quantizza un checkpoint in PyTorch dalla riga di comando: ricette, argomenti
  di calibrazione, valori predefiniti e le famiglie accettate da ogni ricetta.
lead: >-
  Sostituisce i moduli float di un modello con moduli quantizzati, li calibra su
  immagini senza etichette quando la ricetta ha bisogno di statistiche e salva
  il risultato come checkpoint PyTorch.
keywords:
  - libreyolo quantize cli
  - quantizzazione int8 riga di comando
  - quantizzazione fp8
  - post training quantization yolo
  - argomenti libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo quantize
    mono: true
  - label: Obbligatorio
    value: model
    mono: true
  - label: Output
    value: >-
      Il percorso di origine con -<recipe> prima del suffisso, es.
      LibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # Calibra su coco128 e scrive LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Solo cast, senza calibrazione'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Calibrazione più ampia, poi recupero'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # L'addestramento quantization-aware sul checkpoint quantizzato recupera
        accuratezza.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Sinossi

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`recipe=int8` e `--recipe int8` sono lo stesso argomento.

## Argomenti

| Argomento | Default | Significato |
|---|---|---|
| `model` | | Pesi del modello `.pt`. Obbligatorio |
| `recipe` | `int8` | Ricetta di quantizzazione: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Immagini di calibrazione: un YAML di dati o il nome di un dataset integrato. Senza etichette, solo forward. `none` salta la calibrazione |
| `samples` | `128` | Numero massimo di immagini di calibrazione |
| `batch` | `8` | Dimensione del batch di calibrazione |
| `algorithm` | `auto` | Stima dell'intervallo delle attivazioni: `auto`, che seleziona minmax, oppure `minmax`, oppure `percentile` |
| `out` | | Percorso del checkpoint di output. Per default è il percorso di origine con `-<recipe>` prima del suffisso |
| `device` | `auto` | Dispositivo |
| `allow_download_scripts` | `false` | Consente il Python incorporato nei blocchi di download del YAML del dataset |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `help_json` | `false` | Scrive lo schema del comando come JSON ed esce |

## Esempi

<code-tabs name="examples" />

## Note

### Quali famiglie la accettano

La quantizzazione copre quattro famiglie: `yolo9`, `rfdetr`, `birefnet` e
`feynobg`. Qualsiasi altra famiglia esce con `quantize_failed`, che porta con sé
l'elenco.

### Cosa tocca ogni ricetta

`fp16` e `bf16` sono cast. Cambiano solo il dtype, non hanno bisogno di
calibrazione e `calib=none` è l'impostazione giusta per loro.

`int8` e `fp8` quantizzano i moduli `Conv2d` e `Linear`, ed è per questo che si
adattano alle famiglie convoluzionali.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2` quantizzano solo `nn.Linear`, quindi
puntano alle famiglie transformer. Chiedere una di queste su `yolo9` viene
rifiutato con una spiegazione invece di produrre in silenzio un modello non
quantizzato, perché lì l'accelerazione sotto gli 8 bit riguarda solo le GEMM e
le convoluzioni resterebbero a precisione più alta.

`int8`, `fp8`, `w4a8` e `int2` hanno bisogno di statistiche di calibrazione per
le loro attivazioni. `int2` richiede anche un addestramento successivo per
recuperare, quindi viene rifiutato su `birefnet` e `feynobg`, che non hanno un
trainer.

Ogni famiglia mantiene in float un insieme di moduli qualunque sia la ricetta: i
primi layer, le teste di predizione e, su YOLOv9, la convoluzione DFL, che è un
operatore di aspettazione integrale fisso e non va quantizzato.

### I dati di calibrazione non sono dati di addestramento

`calib` punta a un piccolo insieme di immagini senza etichette, usato solo in
avanti, per ricavare gli intervalli delle attivazioni. Non ci si valuta contro e
le sue etichette non vengono mai lette. Il `coco128.yaml` predefinito si scarica
al primo utilizzo da un URL, quindi non serve alcun permesso aggiuntivo; un YAML
con uno script di download in Python incorporato richiede
`allow_download_scripts=true`.

`algorithm=percentile` è disponibile e può ridurre l'accuratezza sulle famiglie
transformer, ed è per questo che `auto` seleziona minmax.

### Recuperare l'accuratezza

L'output è un normale checkpoint PyTorch, quindi
[`libreyolo train`](/docs/cli/train) lo accetta direttamente. Addestrare un
checkpoint quantizzato è addestramento quantization-aware; aggiungere
`distill_model=<teacher>` lo rende distillazione quantization-aware.

### Output e codici di uscita

Il risultato stampa il percorso salvato, la ricetta, la modalità di esecuzione,
se la calibrazione è stata eseguita e il conteggio dei moduli sostituiti per
tipo. Il codice di uscita è `0` in caso di successo, `4` quando il modello non
si può caricare, `5` quando la quantizzazione o il salvataggio falliscono, e `1`
per altri errori a runtime.

Correlato: [`libreyolo export`](/docs/cli/export), che esce da PyTorch e scrive
invece un artefatto di deployment.
