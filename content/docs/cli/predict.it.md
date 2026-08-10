---
title: libreyolo predict
seo_title: riferimento del comando libreyolo predict
description: >-
  Esegue l'inferenza da riga di comando: ogni argomento, il suo valore
  predefinito letto dalla definizione della CLI e i flag che cambiano ciò che
  finisce su stdout.
lead: >-
  Esegue un modello caricato su una sorgente e stampa le predizioni. La sorgente
  può essere un'immagine, una directory, un video, un URL o uno stream dal vivo;
  il modello può essere un checkpoint o un artefatto esportato.
keywords:
  - libreyolo predict cli
  - inferenza yolo da riga di comando
  - comando predict libreyolo
  - argomenti libreyolo predict
  - yolo output json terminale
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo predict
    mono: true
  - label: Obbligatorio
    value: source
    mono: true
  - label: Output
    value: 'Predizioni su stdout. Con save=true, file annotati in runs/detect/predict'
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Salvare le immagini annotate
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Classi filtrate, JSON su stdout'
      language: bash
      code: >
        # la classe 0 è person nella lista di classi COCO fornita con il
        checkpoint.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Sinossi

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Gli argomenti sono coppie `key=value`. Lo stesso comando accetta anche la forma
POSIX, quindi `conf=0.4` e `--conf 0.4` sono intercambiabili, e un booleano
scritto `save=true` diventa `--save`. I nomi con un underscore accettano
entrambe le grafie: `max_det=50` e `--max-det 50` raggiungono la stessa opzione.

`libreyolo detect predict ...` è accettato e si comporta in modo identico; la
parola del task viene rimossa prima del parsing.

## Argomenti

| Argomento | Predefinito | Significato |
|---|---|---|
| `source` | | Percorso di un'immagine, directory o URL. Obbligatorio |
| `model` | `yolox-s` | Nome o percorso del modello |
| `conf` | `0.25` | Soglia di confidenza |
| `iou` | `0.45` | Soglia di IoU per NMS |
| `imgsz` | | Dimensione dell'immagine di input: `640` (quadrata) o `480x640` (altezza x larghezza). Se non impostata, la dimensione di input propria del modello |
| `classes` | | Filtra per ID di classe, ad es. `[0,2,5]`. È accettato anche un intero singolo |
| `max_det` | `300` | Numero massimo di rilevamenti per immagine |
| `half` | `false` | Inferenza in FP16 (solo CUDA, richiede il supporto del modello) |
| `save` | `false` | Salva le immagini annotate |
| `batch` | `1` | Immagini per passaggio in avanti per le sorgenti di tipo directory. Sopra 1 esegue una vera inferenza in batch sui modelli che la supportano |
| `stream` | `false` | Restituisce i risultati in modo incrementale. Si attiva automaticamente per webcam e stream dal vivo |
| `stream_buffer` | `false` | Mette in buffer ogni frame dal vivo invece di conservare solo il più recente |
| `vid_stride` | `1` | Elabora un frame video o dal vivo ogni N |
| `show` | `false` | Mostra i risultati di video e dal vivo; `q` interrompe |
| `tiling` | `false` | Inferenza a tasselli per immagini grandi |
| `overlap_ratio` | `0.2` | Rapporto di sovrapposizione tra i tasselli |
| `output_path` | | Percorso di output esplicito. Altrimenti `project/name` quando `save=true` |
| `color_format` | `auto` | Colore in input: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Formato di output: `jpg`, `png`, `webp` |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Modello per il rilevamento dei volti (percorso o nome CLI). Obbligatorio per i modelli di gaze |
| `gallery` | | Galleria di volti `.npz` prodotta da `libreyolo enroll` con cui identificare i volti. Solo per i modelli di embedding facciali |
| `gallery_threshold` | `0.4` | Soglia di coseno per una corrispondenza di identità nella galleria |
| `project` | `runs/detect` | Radice della directory di output |
| `name` | `predict` | Nome dell'esperimento |
| `exist_ok` | `false` | Riusa la directory di output esistente |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `false` | Output dettagliato su stderr |
| `help_json` | `false` | Stampa lo schema del comando come JSON ed esce |

## Esempi

<code-tabs name="examples" />

## Note

Un artefatto esportato si carica esattamente come un checkpoint, quindi
`model=weights/LibreYOLO9s.onnx` e `model=weights/LibreYOLO9s.engine` sono
valori validi per `model`. Tre opzioni vengono rifiutate su quei runtime invece
di essere ignorate: `tiling`, `overlap_ratio` e `output_file_format` escono con
`config_unsupported` quando un backend di runtime non può rispettarle.

`half` va nella direzione opposta. I runtime esportati lo ricevono ed eseguono
in FP16; l'inferenza nativa in PyTorch registra che è stato ignorato e prosegue
in FP32.

I modelli di gaze (stima dello sguardo) sono a due stadi e non hanno un
rilevatore proprio, quindi `face_detector` è obbligatorio per loro. `gallery` si
applica solo ai modelli il cui task è `embed`; passarlo a qualsiasi altro esce
con `config_unsupported`.

stdout trasporta i risultati e nient'altro; avanzamento, avvisi ed errori vanno
su stderr. `json=true` stampa un oggetto JSON per invocazione, o uno per frame
quando è in streaming, ciascuno con il proprio `schema_version`. `quiet=true`
silenzia stderr. Insieme danno a un lettore automatico uno stream di stdout
pulito.

Il codice di uscita è `0` in caso di successo, `2` per un errore di uso o di
configurazione, `3` quando la sorgente non viene trovata, `4` quando il modello
non può essere caricato e `1` per altri errori a runtime.

`help_json=true` stampa i parametri, i tipi, i valori predefiniti e i flag del
comando come JSON senza eseguire nulla, ed è il modo affidabile per rileggere
questa tabella da una versione installata.

Correlati: [`libreyolo val`](/docs/cli/val) per metriche misurate su un dataset,
[`libreyolo export`](/docs/cli/export) per produrre gli artefatti di runtime
citati sopra.
