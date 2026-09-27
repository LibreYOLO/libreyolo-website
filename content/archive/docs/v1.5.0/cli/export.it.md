---
title: libreyolo export
seo_title: riferimento del comando libreyolo export
description: >-
  Esporta un checkpoint in un formato di deployment: ogni argomento con il suo
  valore predefinito, dove finisce l'artefatto e le combinazioni che il comando
  rifiuta.
lead: >-
  Converte un checkpoint in un formato di deployment e scrive l'artefatto sotto
  weights/. Il formato decide quali degli argomenti qui sotto si applicano.
keywords:
  - libreyolo export cli
  - esportare yolo onnx
  - comando libreyolo export
  - esportare yolo tensorrt
  - argomenti libreyolo export
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo export
    mono: true
  - label: Obbligatorio
    value: model
    mono: true
  - label: Output
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # Scrive weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS dentro il grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Eseguire l'artefatto
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # La factory si basa sul suffisso del file, quindi l'esportazione si
        carica come un checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Sinossi

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`format=onnx` e `--format onnx` sono lo stesso argomento.

## Argomenti

| Argomento | Default | Significato |
|---|---|---|
| `model` | | Pesi del modello `.pt`. Obbligatorio |
| `format` | `onnx` | Formato di esportazione: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Piattaforma di destinazione RKNN, per ora solo `rk3588`. Rifiutato con qualsiasi altro formato |
| `imgsz` | | Dimensione dell'immagine di input: `640` oppure `480x640` (altezza x larghezza). Si accetta anche `480,640`. La dimensione propria del modello se non impostato |
| `batch` | `1` | Dimensione del batch dell'esportazione |
| `half` | `false` | Precisione FP16 |
| `int8` | `false` | Quantizzazione INT8 |
| `dynamic` | `false` | Forme di input dinamiche (ONNX) |
| `simplify` | `true` | Semplificazione del grafo ONNX |
| `nms` | `false` | Incorpora l'NMS nel modello. Solo ONNX e CoreML |
| `conf` | `0.25` | Soglia di confidenza per l'NMS incorporato |
| `iou` | `0.45` | Soglia di IoU per l'NMS incorporato |
| `max_det` | `300` | Numero massimo di rilevamenti per l'NMS incorporato di ONNX |
| `opset` | | Versione dell'opset ONNX. Scelta automaticamente se non impostata |
| `data` | | Dati di calibrazione per INT8 |
| `fraction` | `1.0` | Frazione dei dati di calibrazione da usare |
| `device` | `auto` | Dispositivo per il tracing |
| `allow_download_scripts` | `false` | Consente Python incorporato nei blocchi di download del YAML del dataset |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `false` | Log dettagliato dell'esportazione |
| `verify` | `false` | Esegue il simulatore PC di RKNN Toolkit2 e lo confronta con ONNX Runtime. Solo RKNN |
| `help_json` | `false` | Stampa lo schema del comando come JSON ed esce |

`engine` è un alias di `tensorrt` e `litert` un alias di `tflite`. Entrambi
vengono risolti nel nome canonico prima che venga scritto qualsiasi cosa, quindi
l'output JSON e la riga di log riportano sempre `tensorrt` o `tflite`.

## Esempi

<code-tabs name="examples" />

## Note

### Dove finisce il file

Il comando non accetta un percorso di output. L'artefatto viene scritto in
`weights/`, con il nome base del checkpoint di origine più il suffisso del
formato, e con `_fp16` o `_int8` inserito quando è stata richiesta una di quelle
precisioni. `LibreYOLO9s.pt` esportato in ONNX a FP16 diventa
`weights/LibreYOLO9s_fp16.onnx`. Il risultato JSON porta l'`output_path`
risolto, la dimensione del file in MB e la forma dell'input come
`[batch, 3, height, width]`.

### Combinazioni che vengono rifiutate

`nms=true` è accettato per ONNX e CoreML e rifiutato per ogni altro formato con
`nms_unsupported_format`. Su ONNX forza `dynamic` a off, dato che il grafo
incorporato è fissato a batch 1, e lo segnala su stderr. Su CoreML accetta
`conf` e `iou` ma non `max_det`, quindi un `max_det` diverso da quello
predefinito insieme a `format=coreml nms=true` esce con `config_unsupported`.

`half=true` insieme a `int8=true` non è un errore. Vince INT8, `half` viene
scartato e un avviso va su stderr.

`name` e `verify` oggi sono opzioni di RKNN. Passarne una qualsiasi con un altro
formato esce con `config_unsupported` invece di essere ignorata.

### Quali formati supporta una famiglia

Il supporto è per famiglia e per task, non globale. `libreyolo formats
family=<family> task=<task>` stampa il livello di ogni formato per quella
combinazione, con il motivo e l'eventuale vincolo associato. Vedi
[`libreyolo formats`](/docs/cli/utilities) per gli argomenti.

Alcuni formati richiedono un'installazione opzionale e altri richiedono un
toolchain. Una dipendenza Python mancante esce con `export_dep_missing`; una
precisione che il formato non può produrre esce con
`format_precision_unsupported`.

### Eseguire ciò che hai esportato

Gli artefatti esportati si caricano attraverso la stessa factory di modelli dei
checkpoint, in base al suffisso del file, quindi
`libreyolo predict model=weights/LibreYOLO9s.onnx` funziona senza nessuna
ulteriore conversione. Tre opzioni di predizione fanno eccezione e vengono
rifiutate sui backend di runtime: `tiling`, `overlap_ratio` e
`output_file_format`.

Due destinazioni di deployment hanno una pagina propria:
[NVIDIA DeepStream](/docs/export/deepstream) e
[NVIDIA Jetson](/docs/export/jetson).

### Output e codici di uscita

stdout porta il risultato; l'avanzamento va su stderr. Il codice di uscita è `0`
in caso di successo, `2` per un errore d'uso o di configurazione, `4` quando il
modello non può essere caricato, `5` per un formato sconosciuto, una dipendenza
di esportazione mancante, una precisione non supportata o una richiesta di NMS
incorporato rifiutata, e `1` per altri errori a runtime.

Correlato: [`libreyolo quantize`](/docs/cli/quantize), che resta in PyTorch e
scrive un checkpoint invece di un artefatto di deployment.
