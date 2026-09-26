---
title: libreyolo train
seo_title: riferimento del comando libreyolo train
description: >-
  Addestra un modello dalla riga di comando: tutti i 59 argomenti con i loro
  valori predefiniti, come i valori predefiniti di ogni famiglia li
  sostituiscono e quali argomenti una famiglia ignora.
lead: >-
  Addestra un modello su un dataset e scrive checkpoint, metriche e log in una
  directory di esecuzione. Ogni argomento qui sotto ha un valore predefinito
  preso dalla definizione del comando, che la configurazione di addestramento
  propria di una famiglia di modelli può sostituire.
keywords:
  - libreyolo train cli
  - addestrare yolo da riga di comando
  - comando libreyolo train
  - argomenti libreyolo train
  - addestrare yolo dataset personalizzato
  - congelare layer yolo
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo train
    mono: true
  - label: Obbligatorio
    value: data
    mono: true
  - label: Output
    value: 'Checkpoint, metriche e log in runs/train/exp'
snippets:
  examples:
    - label: Base
      language: bash
      code: >
        # coco8.yaml è incluso nel pacchetto e scarica le sue 8 immagini al
        primo uso.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Controllare prima la configurazione risolta
      language: bash
      code: >
        # Stampa quello che l'esecuzione userebbe, valori predefiniti della
        famiglia

        # inclusi, ed esce senza addestrare né caricare dati.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Esecuzione con nome e ricetta esplicita
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Sinossi

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`epochs=50` e `--epochs 50` sono lo stesso argomento. I booleani accettano
`true` e `false`: `amp=false` diventa `--no-amp` dove il flag ha una forma
negativa.

## Argomenti

### Modello e dati

| Argomento | Predefinito | Significato |
|---|---|---|
| `data` | | Percorso dello YAML del dataset (formato YOLO, ad esempio `coco8.yaml`). Obbligatorio |
| `model` | `yolox-s` | Nome del modello o percorso dei pesi |
| `task` | | Override esplicito del task: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Usa pesi preaddestrati. `false` costruisce l'architettura e addestra da zero |
| `allow_download_scripts` | `false` | Consente il Python incorporato nei blocchi di download dello YAML del dataset |

### Ciclo di addestramento

| Argomento | Predefinito | Significato |
|---|---|---|
| `epochs` | `300` | Epoche di addestramento |
| `batch` | `16` | Dimensione del batch per dispositivo |
| `imgsz` | `640` | Dimensione delle immagini di addestramento: `640` (quadrata) o `480x640` (altezza x larghezza) |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Worker del dataloader |
| `cache` | `false` | Mette in cache le immagini per velocizzare il caricamento dei dati: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Seed casuale |
| `resume` | | Riprende l'addestramento: `true`, oppure il percorso di un checkpoint |
| `amp` | `true` | Automatic Mixed Precision |
| `amp_dtype` | `float16` | Tipo di dato dell'AMP su CUDA: `float16` o `bfloat16` |
| `cuda_graph` | `false` | Cattura il forward e il backward dell'addestramento in CUDA graph. Solo GPU singola e solo per le famiglie supportate; le altre girano in modalità eager |
| `lora` | `false` | Fine-tuning con LoRA, per le famiglie transformer elencate nelle Note |
| `freeze` | | Congela i layer: un numero intero, una lista di indici o nomi di moduli |

### Distillazione

| Argomento | Predefinito | Significato |
|---|---|---|
| `distill_model` | | Insegnante: un checkpoint di un rilevatore, oppure l'id di un foundation teacher come `dinov2` per la distillazione delle feature del backbone |
| `dis` | | Peso della loss di distillazione. Se non impostato, il valore pubblicato per quel tipo di loss |
| `distill_loss_type` | `mgd` | Loss sulle feature per gli insegnanti rilevatori: `mgd`, `cwd`. I foundation teacher usano sempre `feat_mse` |

### Ottimizzatore

| Argomento | Predefinito | Significato |
|---|---|---|
| `optimizer` | `sgd` | Ottimizzatore: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate iniziale (tasso di apprendimento) |
| `momentum` | `0.937` | Momento di SGD, e coefficiente del primo momento per gli ottimizzatori Adam |
| `weight_decay` | `0.0005` | Regolarizzazione L2 |
| `nesterov` | `true` | Momento di Nesterov |

### Scheduler

| Argomento | Predefinito | Significato |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Tipo di schedule del learning rate |
| `warmup_epochs` | `5` | Durata del warmup |
| `warmup_lr_start` | `0.0` | Learning rate iniziale del warmup |
| `min_lr_ratio` | `0.05` | Rapporto minimo del learning rate |
| `lr_drop` | `100` | Epoca del calo a gradino del learning rate di RF-DETR |

### Augmentation

| Argomento | Predefinito | Significato |
|---|---|---|
| `mosaic` | `1.0` | Probabilità del mosaic |
| `mixup` | `1.0` | Probabilità del mixup |
| `hsv_prob` | `1.0` | Probabilità del jitter HSV |
| `flip_prob` | `0.5` | Probabilità del flip orizzontale |
| `degrees` | `10.0` | Intervallo di rotazione, in più e in meno, in gradi |
| `translate` | `0.1` | Rapporto di traslazione |
| `shear` | `2.0` | Angolo di shear |
| `mosaic_scale` | `(0.1,2.0)` | Intervallo di scala del mosaic |
| `mixup_scale` | `(0.5,1.5)` | Intervallo di scala del mixup |
| `no_aug_epochs` | `15` | Disattiva la data augmentation per le ultime N epoche |

### EMA

| Argomento | Predefinito | Significato |
|---|---|---|
| `ema` | `true` | Media mobile esponenziale |
| `ema_decay` | `0.9998` | Fattore di decadimento dell'EMA |

### Validazione durante l'addestramento

| Argomento | Predefinito | Significato |
|---|---|---|
| `val` | `true` | Valida durante l'addestramento |
| `eval_interval` | `10` | Valida ogni N epoche |
| `max_det` | `300` | Numero massimo di predizioni per immagine dopo la NMS di validazione |
| `eval_max_det` | | Limite del valutatore COCO. Se non impostato, la convenzione AP@100 di pycocotools |
| `faster_coco_eval` | `true` | Usa il backend C++ faster-coco-eval per le metriche COCO quando è installato; altrimenti ricade su pycocotools |
| `save_plots` | `false` | Salva i grafici della validazione finale durante l'addestramento |
| `patience` | `50` | Pazienza dell'early stopping. `0` lo disattiva |

### Output

| Argomento | Predefinito | Significato |
|---|---|---|
| `project` | `runs/train` | Directory radice di output |
| `name` | `exp` | Nome dell'esperimento |
| `exist_ok` | `false` | Riusa la directory di output esistente |
| `save_period` | `10` | Salva un checkpoint ogni N epoche |
| `log_interval` | `10` | Registra la loss ogni N batch |

### Flag per agenti

| Argomento | Predefinito | Significato |
|---|---|---|
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Sopprime stderr |
| `dry_run` | `false` | Risolve e stampa la configurazione senza eseguire |
| `help_json` | `false` | Stampa lo schema del comando in JSON ed esce |

## Esempi

<code-tabs name="examples" />

## Note

### I valori predefiniti qui sopra non sono sempre quelli usati

Ogni famiglia di modelli porta con sé la propria configurazione di
addestramento, e dove quella configurazione differisce da quella di base, il
suo valore sostituisce il valore predefinito del comando per ogni argomento che
non hai impostato esplicitamente. Impostare l'argomento tu vince sempre.
`libreyolo cfg` stampa i valori predefiniti di base e le sostituzioni per
famiglia, ed è il modo per vedere che cosa userà davvero una data famiglia.

`imgsz` è l'argomento per cui questo conta di più. Il valore predefinito del
comando è `640`, che non è l'input nativo di ogni checkpoint: le dimensioni
pubblicate per il rilevamento con RF-DETR sono 384, 512, 576 e 704, e i
checkpoint YOLOX `n` e `t` sono a 416. RF-DETR e DEIMv2 sono gestiti inoltrando
`imgsz` solo quando è stato impostato esplicitamente, così altrimenti resta in
vigore la loro dimensione. Alle altre famiglie il valore arriva così com'è e ci
addestrano sopra. FOMO è la famiglia rigida: ogni dimensione accetta solo il
proprio input nativo (96, 192 e 224), quindi un addestramento FOMO ha bisogno
di `imgsz` impostato di conseguenza, altrimenti si ferma con un errore. RF-DETR
richiede anche che il valore sia divisibile per la dimensione delle sue patch
moltiplicata per il numero delle sue finestre, e quando non lo è riporta le due
dimensioni valide più vicine.

### Argomenti che una famiglia ignora

Non tutte le famiglie leggono tutti gli argomenti, e quelli della data
augmentation sono il caso in cui si nota di più. RF-DETR, D-FINE, DEIM, DEIMv2,
RT-DETRv4 e DINOv2 si addestrano con pipeline pass-through senza mosaic, senza
mixup e senza warp affine, quindi lì `mosaic`, `mixup`, `hsv_prob`, `degrees`,
`translate`, `shear`, `mosaic_scale` e `mixup_scale` non raggiungono nulla. EC
condivide quella pipeline ma legge `hsv_prob`, `degrees` e `translate` quando
il suo task è la stima della posa. Le famiglie di classificazione, SegFormer e
NAFNet ignorano tutto quell'insieme e con esso `flip_prob`, perché il loro flip
gira a una probabilità fissa e non configurabile. YOLO-NAS ignora solo
`mosaic`, dato che al suo posto fa data augmentation con una trasformazione
affine per campione sempre attiva. RF-DETR ne ignora altri tre oltre a quella
lista: `optimizer`, `momentum` e `nesterov`.

Impostarne uno non è un errore. L'esecuzione scrive una riga su stderr con il
nome della famiglia e gli argomenti che ignorerà, poi addestra, e quella riga è
la lista autorevole per la versione installata. È anche l'unico segnale, quindi
un'esecuzione da script con `quiet=true` sopprime l'avviso insieme a tutto il
resto su stderr.

`val=false` è un caso collegato. Imposta `eval_interval` a `0` per la maggior
parte delle famiglie; RF-DETR non può disattivare la validazione in quel modo e
registra di aver ignorato la richiesta.

### Altri comportamenti da conoscere

`lora=true` è accettato da RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4,
EC e ConvNeXt. Qualsiasi altra famiglia esce con `config_unsupported` invece di
addestrare senza LoRA.

`pretrained=false` combinato con `resume` viene rifiutato per le famiglie che
supportano l'addestramento da zero, dato che i due chiedono cose opposte.

`mosaic` e `mixup` sono la scrittura da riga di comando dei campi di
configurazione `mosaic_prob` e `mixup_prob`. Nelle famiglie il cui mixup si
applica solo ai campioni con mosaic, `mixup` sopra zero con `mosaic` a zero non
scatta mai, e l'esecuzione lo dice.

`dry_run=true` risolve il riferimento al modello, applica i valori predefiniti
della famiglia e stampa la configurazione con cui addestrerebbe. Non carica il
dataset, quindi è il modo economico per confermare che un argomento sia
arrivato al valore che ti aspettavi.

stdout porta l'oggetto con il risultato finale; avanzamento e avvisi vanno su
stderr. Il codice di uscita è `0` in caso di successo, `2` per un errore d'uso
o di configurazione, `3` quando il dataset non si trova o non si può leggere,
`4` quando il modello non si può caricare, e `1` per gli altri errori a runtime.

Correlati: [`libreyolo doctor`](/docs/cli/doctor) per controllare un dataset
prima di impegnarsi in un addestramento, [`libreyolo monitor`](/docs/cli/monitor)
per seguire un'esecuzione nel browser, [`libreyolo val`](/docs/cli/val) per
misurare il risultato.
