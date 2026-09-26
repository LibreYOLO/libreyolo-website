---
title: libreyolo val
seo_title: riferimento del comando libreyolo val
description: >-
  Valuta un checkpoint su uno split di un dataset dalla riga di comando: ogni
  argomento con il suo valore predefinito e le chiavi delle metriche che ogni
  task restituisce.
lead: >-
  Valuta un modello su uno split di un dataset e stampa le metriche. L'insieme
  delle metriche dipende dal task del modello, e i numeri sono quelli con cui si
  costruisce una riga di benchmark.
keywords:
  - libreyolo val cli
  - comando validazione libreyolo
  - valutare modello yolo riga di comando
  - calcolare mAP50-95 terminale
  - argomenti libreyolo val
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo val
    mono: true
  - label: Obbligatori
    value: 'model, data'
    mono: true
  - label: Output
    value: Metriche su stdout. Grafici e JSON COCO in runs/val/exp quando richiesti
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Grafici e JSON COCO
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Leggibile da una macchina
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Sinossi

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`batch=8` e `--batch 8` sono lo stesso argomento.

## Argomenti

| Argomento | Predefinito | Significato |
|---|---|---|
| `model` | | Percorso dei pesi del modello o nome CLI. Obbligatorio |
| `data` | | Percorso dello YAML del dataset (formato YOLO, ad esempio `coco8.yaml`). Obbligatorio |
| `data_dir` | | Directory del dataset diretta, ignorando il percorso indicato nello YAML |
| `split` | `val` | Split del dataset: `val`, `test`, `train` |
| `batch` | `16` | Dimensione del batch |
| `imgsz` | | Dimensione dell'immagine: `640` (quadrata) o `480x640` (HxW). Se non impostata, la dimensione di input propria del modello |
| `conf` | `0.001` | Soglia di confidenza |
| `iou` | `0.6` | Soglia IoU della NMS |
| `max_det` | `300` | Numero massimo di predizioni per immagine dopo la NMS |
| `eval_max_det` | | Limite dell'evaluator COCO. Se non impostato, la convenzione AP@100 di pycocotools |
| `faster_coco_eval` | `true` | Usa il backend C++ faster-coco-eval per le metriche COCO quando è installato; altrimenti ricade su pycocotools |
| `half` | `false` | Inferenza FP16 |
| `amp_dtype` | `float16` | Dtype dell'autocast CUDA quando `half=true`: `float16` o `bfloat16` |
| `save_json` | `false` | Salva i risultati in JSON in formato COCO |
| `save_plots` | `false` | Salva i grafici di validazione: metriche, AP per classe, matrice di confusione, campioni |
| `workers` | `4` | Worker del dataloader |
| `device` | `auto` | Dispositivo |
| `project` | `runs/val` | Radice della directory di output |
| `name` | `exp` | Nome dell'esperimento |
| `exist_ok` | `false` | Riusa la directory di output |
| `allow_download_scripts` | `false` | Consente il Python incorporato nei blocchi di download dello YAML del dataset |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `true` | Output verboso |
| `help_json` | `false` | Stampa lo schema del comando come JSON ed esce |

## Esempi

<code-tabs name="examples" />

## Note

### Che cosa sono le metriche

L'insieme stampato segue il task del modello, e l'output JSON usa le stesse
chiavi.

Rilevamento, segmentazione e box orientati riportano `mAP50`, `mAP50_95`,
`precision` e `recall`. Quando un modello predice più di un tipo di output, i
gruppi per tipo compaiono accanto come `box_metrics`, `mask_metrics` e
`obb_metrics`, ognuno con le stesse quattro chiavi.

La classificazione riporta `accuracy_top1` e `accuracy_top5`. Il rilevamento di
punti riporta `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` e `mAP_sweep`.
La profondità riporta `abs_rel`, `rmse`, `delta1`, `delta2` e `delta3`. La
segmentazione semantica riporta `mIoU` e `pixel_accuracy`. Il restauro riporta
`PSNR` e `SSIM`.

Il risultato JSON porta anche `eval_backend`, che indica la libreria di
valutazione COCO e la versione che hanno prodotto i numeri, così due esecuzioni
si possono confrontare sapendo se lo stesso backend ha valutato entrambe.

### Soglie

I valori predefiniti qui sono valori di valutazione, non di predizione: `conf` è
`0.001` e `iou` è `0.6`, mentre [`libreyolo predict`](/docs/cli/predict) usa
`0.25` e `0.45`. Alzare `conf` a una soglia di visualizzazione abbassa il recall
e con esso la mAP, quindi un numero ottenuto in quel modo non è confrontabile
con uno pubblicato.

`imgsz` non è impostato per default, il che significa la dimensione di input
propria del modello. Impostarlo valuta alla dimensione indicata, che è il modo
in cui un checkpoint viene misurato fuori dalla sua risoluzione nativa.

### Dataset che si scaricano

Uno YAML di dataset il cui campo `download` è una URL si scarica al primo uso
senza alcun permesso aggiuntivo. Uno che porta uno script di download Python
incorporato richiede `allow_download_scripts=true`, e il comando avvisa su
stderr che l'esecuzione di codice locale è stata abilitata. I `coco8.yaml` e
`coco128.yaml` inclusi sono basati su URL, quindi non richiedono nulla.

### Output e codici di uscita

stdout porta le metriche; l'avanzamento va su stderr. `json=true` stampa un
singolo oggetto con `schema_version`, e `quiet=true` silenzia stderr.

Il codice di uscita è `0` in caso di successo, `2` per un errore d'uso o di
configurazione, `3` quando il dataset non viene trovato, `4` quando il modello
non può essere caricato e `1` per altri errori a runtime.

Correlato: [`libreyolo train`](/docs/cli/train), che esegue questa stessa
valutazione secondo il proprio calendario tramite `eval_interval`.
