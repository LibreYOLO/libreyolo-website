---
title: API del modello
seo_title: Metodi e firme dell'oggetto modello di LibreYOLO
description: >-
  Tutti i metodi di un modello LibreYOLO caricato: predict, embed, track, val,
  train, export, save, quantize, info e i controlli dei grafi CUDA, con i valori
  predefiniti reali.
lead: >-
  Un modello LibreYOLO caricato è un'istanza di BaseModel. Questa pagina elenca
  i metodi che quell'istanza porta con sé, con le firme e i valori predefiniti
  letti da libreyolo/models/base/model.py.
keywords:
  - metodi modello libreyolo
  - argomenti predict libreyolo
  - argomenti val libreyolo
  - esportare modello libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Firme e valori predefiniti letti da libreyolo/models/base/model.py e
  libreyolo/models/base/inference.py alla v1.5.0. Le classi di famiglia possono
  restringerli o estenderli; train() è definito per famiglia e qui è documentato
  solo il suo wrapper condiviso cfg=.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True restituisce un generatore, un Results per frame o
        immagine.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Costruzione

La factory restituisce un'istanza della classe di famiglia. Costruire quella
classe direttamente accetta gli stessi argomenti, salvo che `size` è obbligatorio:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` seleziona CUDA quando è disponibile, poi MPS, poi la CPU. Un
intero o una stringa di cifre viene letto come ordinale CUDA, quindi `device=0`
e `device="0"` significano entrambi `cuda:0`. `task` viene validato rispetto ai
`SUPPORTED_TASKS` della famiglia. Passare `model_path=None` costruisce
l'architettura e la lascia in modalità di addestramento; passare un `dict`
carica direttamente quello state dict.

## predict e \_\_call\_\_

`predict` è un alias di `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `source` | `None` | Immagine, lista o tupla di immagini in memoria, directory, file video, oppure una sorgente schermo come `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Soglia di confidenza |
| `iou` | `0.45` | Soglia di IoU per la NMS |
| `imgsz` | `None` | Override della dimensione di input; con `None` si usa la dimensione nativa del modello |
| `device` | `None` | Override del dispositivo per questa chiamata |
| `classes` | `None` | Mantiene solo questi ID di classe |
| `max_det` | `300` | Numero massimo di rilevamenti per immagine |
| `augment` | `False` | Data augmentation al momento del test |
| `save` | `False` | Scrive un'immagine o un video annotato |
| `batch` | `1` | Immagini per forward pass con sorgenti directory e liste |
| `stream` | `False` | Restituisce un generatore invece di una lista materializzata |
| `stream_buffer` | `False` | Mantiene ogni frame catturato dal vivo invece del solo più recente |
| `vid_stride` | `1` | Elabora un frame video o schermo ogni N |
| `show` | `False` | Mostra i frame annotati in una finestra |
| `output_path` | `None` | Percorso di output quando `save=True` |
| `color_format` | `"auto"` | Indicazione sul formato colore per gli array in memoria |
| `tiling` | `False` | Inferenza a tasselli per immagini grandi |
| `overlap_ratio` | `0.2` | Rapporto di sovrapposizione dei tasselli |
| `output_file_format` | `None` | `"jpg"`, `"png"` o `"webp"` |
| `cuda_graph` | `False` | `True` cattura al primo utilizzo per ogni forma di input, `"auto"` aspetta che una forma si ripeta |

Una sorgente con una sola immagine restituisce un `Results`. Una lista, una
tupla o una directory ne restituiscono una lista, e `stream=True` restituisce
un generatore in ogni caso.

Le sorgenti di stream dal vivo sono illimitate e richiedono `stream=True`.
`tiling` e `augment` non possono essere combinati. La data augmentation al
momento del test solleva un errore per i task `embed`, `point` ed `edge`.

<code-tabs name="usage" />

Con `batch > 1`, le famiglie il cui `SUPPORTS_BATCHED_PREDICT` è true eseguono
un forward impilato per ogni chunk; `batch=1` mantiene un forward per immagine.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Un wrapper di comodo su `predict` che impila ogni riga di embedding in un unico
tensore `(N_total, D)`. Il modello deve essere stato costruito con
`task="embed"`, altrimenti solleva `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Produce un `Results` per frame con `track_id` impostato. `tracker` è
`"bytetrack"`, `"botsort"`, `"ocsort"` o `"deepocsort"`, e viene ignorato
quando si passa `tracker_config`, perché è il tipo di config a selezionare il
tracker. `track_conf` corrisponde a `track_high_thresh` per ByteTrack e
BoT-SORT e a `det_thresh` per OC-SORT e Deep OC-SORT. Il valore predefinito di
`output_path` è `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Restituisce un dizionario di metriche le cui chiavi dipendono dal task; il
rilevamento restituisce `metrics/precision`, `metrics/recall`, `metrics/mAP50`
e `metrics/mAP50-95`. `imgsz` accetta un int quadrato o una tupla
`(height, width)` e per impostazione predefinita usa la dimensione di input
nativa del modello. `plots` è un alias di `save_plots`.
`allow_download_scripts` regola l'esecuzione del Python incorporato che un YAML
di dataset può portare nel suo campo `download`.

`faster_coco_eval` è accettato tramite `**kwargs` e vale `True` per
impostazione predefinita, con ripiego su pycocotools quando il pacchetto non è
installato. Il backend che è stato usato viene riportato in
`model.last_eval_backend`.

La validazione con augmentation solleva un errore per i task `obb` e `pose`.

## train

`train` è definito per famiglia, quindi i suoi argomenti cambiano. Due
comportamenti sono comuni, perché la classe base fa da wrapper al `train` di
ogni famiglia:

- `cfg=` accetta un percorso YAML le cui chiavi vengono unite alla chiamata.
  Gli argomenti keyword espliciti vincono sul file.
- `pretrained=False` su una famiglia del gruppo di copertura `g0` o `g1`
  reinizializza il modello da zero prima dell'addestramento, e non può essere
  combinato con `resume=True`.

Quali parametri di augmentation una famiglia rispetti davvero è una questione
che varia da famiglia a famiglia; vedi la
[matrice della data augmentation](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Restituisce il percorso dell'artefatto scritto. `format` viene risolto tramite
il registro degli exporter, dove `engine` è un alias di `tensorrt` e `litert` è
un alias di `tflite`. Argomenti condivisi da tutti gli exporter:

| Argomento | Predefinito | Significato |
|---|---|---|
| `output_path` | `None` | Percorso del file di output; generato sotto `weights/` quando è omesso |
| `imgsz` | `None` | Tupla `(height, width)` o un singolo int; per impostazione predefinita la dimensione nativa |
| `opset` | `None` | Versione dell'opset ONNX |
| `simplify` | `True` | Esegue la semplificazione del grafo ONNX |
| `dynamic` | `True` | Abilita gli assi dinamici |
| `half` | `False` | Precisione FP16 |
| `int8` | `False` | Precisione INT8 |
| `batch` | `1` | Dimensione del batch integrata nell'artefatto |
| `device` | `None` | Dispositivo su cui fare il tracing |
| `data` | `None` | data.yaml per la calibrazione INT8 |
| `fraction` | `1.0` | Frazione del dataset di calibrazione da usare |
| `allow_download_scripts` | `False` | Consente il Python incorporato nei download degli YAML di dataset |
| `verbose` | `False` | Log dettagliati dell'exporter |

Le combinazioni bloccate sollevano `NotImplementedError` nel preflight, prima
del tracing. La copertura e le sue regole sono nella pagina
[matrice di esportazione](/docs/reference/export-matrix). Quando sono presenti
adattatori LoRA attivi, vengono ripiegati nei pesi densi, e quella fusione
avviene solo dopo ogni rifiuto della richiesta.

## save

```python
model.save(path) -> str
```

Scrive un checkpoint LibreYOLO con schema v1.0: lo state dict più i metadati
descritti nello [schema dei checkpoint](/docs/reference/checkpoint-schema).
Un modello quantizzato porta con sé anche il suo manifest `quant`, così
`LibreYOLO(path)` ripristina la struttura quantizzata e le scale.

## quantize, quant_info e dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantizza in place e restituisce il modello. `recipe` è uno dei cast `fp16` e
`bf16`, le ricette per Conv e Linear `int8` e `fp8`, oppure le ricette solo per
Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2`, che le famiglie transformer
come RF-DETR supportano. `int2` richiede il QAT. `calib` accetta un percorso
data.yaml o il nome di un dataset integrato e legge le immagini solo in avanti;
le etichette non vengono mai lette. Passa `calib=None` per saltare la
calibrazione. `algorithm` è `"minmax"`, `"percentile"` o `"auto"`.

`model.quant_info()` restituisce il riepilogo dello stato di quantizzazione, o
`None` per un modello float. `model.dequantize()` ripristina in place i moduli
float mantenendo i pesi master addestrati con la quantizzazione, che è il ponte
dal QAT a `export(format="onnx", int8=True, data=...)`.

## info e layer

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` restituisce un dizionario compatibile con JSON e scrive nel log un
riepilogo leggibile quando `verbose` è true. `get_available_layer_names` elenca
i layer che una config di distillazione o di estrazione di feature può
nominare.

## Grafi CUDA

Disponibili sulle famiglie il cui attributo di classe `SUPPORTS_CUDA_GRAPH` è
true. Il replay è identico bit a bit all'esecuzione eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # gestore di contesto
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Un grafo catturato è valido solo per la forma esatta con cui è stato catturato,
quindi `batch` e `imgsz` devono coincidere con la successiva chiamata a
`predict`. `capture_graph` toglie il costo della cattura dalla prima richiesta.
`mode` accetta `True` o `"on"` per catturare al primo utilizzo, `"auto"` per
aspettare che una forma si ripeta, e `False` per non fare nulla.
`capture_graph` solleva `NotImplementedError` quando la famiglia non ha aderito
e `CudaGraphUnavailable` quando la cattura fallisce.

## Dispositivo e dtype

Gli oggetti `Results` portano `.to()`, `.cpu()`, `.cuda()` e `.numpy()`; vedi
[Tipi di Results](/docs/reference/results-types). Il modello stesso si sposta
passando `device=` a `predict`, oppure al momento della costruzione.
