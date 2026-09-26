---
title: Schema del checkpoint
seo_title: Schema dei metadati dei checkpoint LibreYOLO v1.0
description: >-
  I metadati che ogni checkpoint .pt di LibreYOLO porta con sé: chiavi
  obbligatorie, aggiunte per task, chiavi di runtime dell'esportazione, manifest
  quantizzati e campi di addestramento.
lead: >-
  Un file .pt di LibreYOLO è un dizionario piatto salvato con torch.save. La
  chiave model contiene lo state dict; le altre chiavi di primo livello sono
  metadati che identificano il checkpoint senza dover analizzare il nome del
  file o annusare lo state dict.
keywords:
  - schema checkpoint libreyolo
  - schema_version 1.0
  - model_family
  - metadati checkpoint libreyolo
  - quant manifest
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Rispecchia docs/checkpoint_schema.md nel repository libreyolo alla v1.5.0,
  verificato per confronto con libreyolo/utils/serialization.py e
  BaseModel.save.
snippets:
  usage:
    - label: Leggere i metadati di un checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Scarica un checkpoint, poi risalvalo così esiste un percorso locale.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Schema v1.0

Ogni checkpoint `.pt` ufficiale di LibreYOLO contiene:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Chiave | Tipo | Significato |
|---|---|---|
| `model` | state dict | I pesi del modello |
| `schema_version` | str | Versione del contratto dei metadati; la v1.0 usa la stringa `"1.0"` |
| `libreyolo_version` | str | La versione che ha prodotto il checkpoint |
| `model_family` | str | Una famiglia registrata, come `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variante all'interno della famiglia, come `t`, `s`, `r18`, `atto` |
| `task` | str | Nome canonico del task |
| `nc` | int | Numero di classi, positivo |
| `names` | dict | `dict[int, str]` con chiavi in `0..nc-1` |
| `imgsz` | int | Risoluzione di input quadrata positiva, oppure lo scalare legacy per un contratto rettangolare |

`task` è uno tra `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` o `mesh`.

I checkpoint ufficiali scrivono tutte le chiavi di `names`. I lettori possono
riempire le chiavi mancanti con etichette `class_i` per le vecchie mappature
sparse, ma le chiavi fuori intervallo non sono valide.

I checkpoint rettangolari mantengono un `imgsz` scalare per i lettori legacy,
impostato a `max(imgsz_h, imgsz_w)`, e scrivono in aggiunta `imgsz_h` e
`imgsz_w` con le dimensioni reali. Un lettore che comprende i campi
rettangolari deve preferirli allo scalare. Le famiglie con un contratto
rettangolare fisso, come HRNet pose, rifiutano le dimensioni di runtime
incompatibili.

Lo schema è volutamente piatto, e `model` è volutamente uno state dict.

<code-tabs name="usage" />

## Aggiunte per la stima della posa

La posa di solito è a classe singola, `nc: 1` con `person`, ma la testa pose di
YOLO-NAS supporta anche la posa multiclasse con un unico scheletro di keypoint
condiviso, nel qual caso `nc` e `names` descrivono le classi come nel
rilevamento. Le esportazioni pose di runtime emettono `scores` con forma
`[batch, anchors, nc]`.

| Chiave | Significato |
|---|---|
| `num_keypoints` | Numero positivo di keypoint usati dalla testa pose |
| `keypoint_dim` | `2` per etichette `x,y` o `3` per etichette `x,y,visibility`; gli output del modello espongono sempre `x,y,visibility` |
| `oks_sigmas` | Sigma OKS opzionali per keypoint; se assenti si usa il valore predefinito del task per `num_keypoints` |
| `num_keypoints_per_class` | Conteggi opzionali di keypoint per classe, per le teste in stile GroupPose il cui tensore dei keypoint è riempito per classe; `0` per le classi senza keypoint |

## Aggiunte per le mesh

I checkpoint mesh usano `task: "mesh"`, `nc: 1` e `names: {0: "person"}`. La
disposizione dei parametri cambia tra i modelli del corpo, quindi le dimensioni
vengono registrate anziché date per scontate.

| Chiave | Significato |
|---|---|
| `body_model` | La parametrizzazione, come `mhr`; obbligatoria, e usata per interpretare ogni campo qui sotto |
| `num_betas` | Numero di coefficienti di identità e forma; 45 per MHR |
| `num_body_pose` | Ampiezza del blocco di parametri della posa del corpo; 130 per MHR. Un vettore piatto, non una tripletta per giunto, perché i giunti del rig hanno gradi di libertà diversi |
| `num_vertices` | Numero di vertici che il decoder emette; 18439 per MHR |
| `num_joints` | Numero di giunti che il decoder emette; 127 per MHR |
| `rotation_format` | Come sono codificate le rotazioni, ad esempio `euler_zyx` per MHR o `axis_angle`. Mai dedotto dalla forma del tensore, dato che un vettore a 3 componenti è ambiguo |

## Segnaposto per i task densi

Diversi task predicono mappe dense anziché classi, quindi gli slot legati alle
classi esistono solo per compatibilità con lo schema.

| Task | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Le predizioni edge sono mappe di probabilità dense float32 in `[0, 1]`.

I checkpoint restore possono aggiungere `degradation`, una breve etichetta di
corruzione come `deblur`, `denoise` o `super-resolution`; `dataset`, un'etichetta
di provenienza come `GoPro` o `SIDD`; e `scale`, un fattore intero positivo di
ingrandimento da input a output, per esempio `4` per un modello di
super-risoluzione x4. Assente o `1` significa che l'immagine ripristinata
mantiene la risoluzione dell'input. Il runtime deriva la scala anche dalla
famiglia e dalla size, quindi `scale` è un metadato di provenienza più che un
requisito al caricamento.

## Aggiunte per l'OCR

La famiglia `ppocr` distribuisce un checkpoint composito per ogni tier, il cui
state dict `model` contiene due sottomodelli sotto gli spazi di nomi di chiave
`det.*` e `rec.*`.

| Chiave | Significato |
|---|---|
| `charset` | L'alfabeto CTC completo nell'ordine degli indici di output: l'indice 0 è il blank CTC, poi il dizionario di riconoscimento, poi il carattere spazio. I loader devono leggerlo dal checkpoint, mai da un file esterno |
| `pipeline` | Valori predefiniti della pipeline fissati al momento della conversione: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Gli argomenti di runtime possono sovrascriverli chiamata per chiamata |
| `components` | Riservata a stadi opzionali della pipeline come l'orientamento del documento, il raddrizzamento e la rotazione delle righe di testo. Vuota nella v1 |

## Metadati di runtime dell'esportazione

Gli artefatti esportati usano la stessa convenzione di doppia scrittura per il
caso rettangolare: `imgsz_h` e `imgsz_w` sono scritti accanto allo scalare
legacy `imgsz`, e un lettore che non comprende i campi rettangolari non deve
trattare silenziosamente lo scalare come un contratto quadrato.

Il supporto rettangolare a runtime dipende dalla famiglia e dal formato. Le
esportazioni della famiglia YOLO9, di HRNet, NAFNet e Real-ESRGAN possono usare
`imgsz_h` e `imgsz_w` non quadrati nei formati supportati; le famiglie o i
formati senza supporto rettangolare esplicito rifiutano i metadati anziché
preelaborare quegli artefatti come quadrati. Le esportazioni HRNet sono teste
fisse, batch uno, FP32 su ritagli di persona, dove W32 accetta 256x192 e W48
accetta 384x288, e il rilevatore di persone non è incorporato nel grafo.

Le esportazioni con NMS incorporato possono aggiungere queste chiavi piatte:

| Chiave | Significato |
|---|---|
| `nms` | Booleano come stringa; `"true"` significa che il grafo include un output di post-elaborazione incorporato |
| `nms_conf` | Soglia di confidenza fissata nell'output incorporato |
| `nms_iou` | Soglia IoU fissata nell'output incorporato |
| `max_det` | Numero massimo di righe di rilevamento post-NMS che l'output incorporato emette |
| `nms_raw_output` | Booleano come stringa; `"true"` significa che il grafo espone anche un output ausiliario grezzo del rilevatore |

Per le esportazioni ONNX di rilevamento YOLO9 con `nms=true`, l'output `0`
(chiamato `output`) è il tensore post-NMS autonomo alle soglie fissate in fase
di esportazione. Quando `nms_raw_output=true`, l'output `1` (chiamato `raw`) è
riservato ai backend LibreYOLO, così da poter applicare il ritaglio nativo sul
canvas originale e la semantica di runtime di
`predict(conf=..., iou=..., max_det=...)`. I consumatori di terze parti
dovrebbero usare il primo output.

Le esportazioni pose possono aggiungere `num_keypoints`; `keypoint_dim`, dove le
esportazioni grezze in stile GroupPose possono usare valori più grandi come `8`
quando il tensore include campi di precisione o logit di classe;
`num_keypoints_per_class` come lista codificata in JSON, dove gli slot di classe
con zero keypoint vanno preservati perché definiscono lo schema; e `pose_input`,
dove `"person_crop"` significa che il grafo consuma un ritaglio già estratto e
non contiene alcun rilevatore. Le esportazioni HRNet di runtime richiedono quel
valore.

Le esportazioni di classificazione possono aggiungere `crop_pct`, un rapporto
float di ritaglio centrale il cui obiettivo di ridimensionamento prima del
ritaglio è `round(imgsz / crop_pct)` e che vale `0.875` quando è assente, e
`interpolation`, `"bilinear"` o `"bicubic"`, con valore predefinito
`"bilinear"`.

Le esportazioni ExecuTorch scrivono i metadati piatti in un file affiancato
`<program>.pte.json` obbligatorio. Il contratto v1 è CPU, FP32, batch 1 e canvas
di input fisso, e richiede in aggiunta `executorch_version`,
`executorch_delegate` uguale a `"xnnpack"` e un `executorch_delegate_partitions`
positivo. Il loader rifiuta un file affiancato che dichiari un altro delegate,
forme dinamiche o precisione diversa da FP32.

Le esportazioni MNN scrivono i metadati piatti in un file affiancato
`<model>.mnn.json` obbligatorio. Il contratto v1 è CPU, FP32, solo rilevamento e
forma di input NCHW fissa, e richiede in aggiunta `mnn_version`, `mnn_backend`
uguale a `"cpu"`, `mnn_input_names` e `mnn_output_names` ordinati e non vuoti,
`mnn_input_shape` come quattro interi positivi nell'ordine
`[batch, channels, height, width]`, e `mnn_batch` uguale a
`mnn_input_shape[0]`. Il loader rifiuta metadati di forma dinamici, non FP32,
non di rilevamento, di famiglie non supportate o incoerenti.

Un `.pte` e un `.mnn` sono artefatti specifici del backend, non checkpoint
PyTorch.

## Checkpoint quantizzati

Un modello quantizzato aggiunge una chiave piatta opzionale, `quant`, che
contiene un manifest dict con `schema`, `recipe`, `keep_high_precision`,
`execution`, la provenienza della calibrazione, `module_count` e `state`. I
manifest FP8 possono portare anche `fp8_tensorwise_weights`, l'elenco esatto dei
nomi dei moduli `QuantLinear` la cui scala dei pesi è per tensore anziché per
canale di output. Un loader che vede `quant` ricostruisce la struttura dei
moduli quantizzati e la politica di scaling prima di `load_state_dict`.

`state` distingue le due forme di artefatto.

`"prepared"`, il valore predefinito, contiene i pesi master FP32 più i buffer di
scala `_q_*` ed è addestrabile. Un lettore senza supporto alla quantizzazione
può ignorare la chiave `quant` e caricare i master come modello float.

`"finalized"` è la forma di deployment scritta da `export(format="pt")`. I
master vengono rimossi e ogni modulo quantizzato porta invece pesi impacchettati:

| Ricetta | Tensori impacchettati | Dequantizzazione |
|---|---|---|
| int8 | `weight_packed` int8 nella forma originale dei pesi, `_q_w_scale` FP32 per canale | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn nella forma originale, `_q_w_scale` FP32 con una voce per canale di output | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, due codici a 4 bit per byte, nibble basso per primo, codice `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, gruppo 128 lungo in_features | Scala per gruppo |
| int2 | Quattro codici a 2 bit per byte, codice `q + 2`, gruppo 64 | Scala per gruppo |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, codice `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 per tensore | `block_scale * amax / (448 * 6)` |
| mxfp4 | Come nvfp4 ma con blocchi da 32 elementi, più `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

I buffer di intervallo delle attivazioni `_q_act_lo`, `_q_act_hi` e
`_q_calibrated` sono conservati per int8. Il manifest registra `remainder`,
`"fp16"` o `"fp32"`, per i tensori non quantizzati. Lo spacchettamento riproduce
la simulazione bit per bit, quindi l'inferenza finalized coincide esattamente
con quella prepared sul dispositivo che esegue la finalizzazione. Questa
disposizione è il contratto stabile per esportatori e runtime esterni.

## Checkpoint di addestramento

I checkpoint del trainer usano lo stesso nucleo di metadati obbligatori e
possono aggiungere campi piatti di addestramento e di ripresa:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` dichiara se il `model` di primo livello è stato smussato con
EMA. Quando l'EMA è attiva, `train_model`, `ema` e `ema_updates` preservano lo
stato di ripresa. I pesi di inferenza pubblicati dovrebbero essere snelli e non
dovrebbero includere optimizer, epoca, config, loss o stato di ripresa EMA, a
meno che non vengano distribuiti intenzionalmente come checkpoint di
addestramento.

Per compatibilità tra release, i lettori accettano i vecchi alias della metrica
migliore `best_mAP50_95`, `best_mAP50`, `best_metric` e `best_metric_name`.

## Snapshot esterni

Lo schema governa i file `.pt` prodotti da LibreYOLO. Non rinomina né incapsula
gli snapshot upstream multi-file usati dai tier di modelli separati.

La size `14b-a7b` di LibreMODUS è un'eccezione esplicita: l'alias si risolve
attraverso `LibreVLM(...)` in una directory di file upstream fissati, e
LibreYOLO non le aggiunge i metadati v1.0 né la ripubblica come `.pt`.

## Pesi legacy e di terze parti

I nuovi writer validano in modo rigoroso e devono emettere i metadati v1.0.
Quando i metadati mancano o sono incompleti, i checkpoint legacy che sembrano
LibreYOLO si caricano attraverso il percorso di compatibilità con un avviso e le
istruzioni di conversione, mentre i checkpoint upstream di terze parti vengono
instradati alla conversione automatica. Vedi
[checkpoint upstream](/docs/reference/upstream-checkpoints).

## Helper

Gli helper dello schema vivono in `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` non muta nulla e restituisce l'elenco degli
errori; con `strict=True` solleva invece `CheckpointMetadataError`.
`model.save(path)` è il modo supportato per scrivere un checkpoint conforme.
