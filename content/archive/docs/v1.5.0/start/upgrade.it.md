---
title: Aggiornare alla 1.5.0
seo_title: Aggiornare LibreYOLO dalla 1.4.0 alla 1.5.0
description: >-
  Le quattro modifiche al codice richieste dalla 1.5.0, le tre modifiche che
  spostano le metriche e i cambiamenti di comportamento minori da conoscere
  prima di confrontare le esecuzioni.
lead: >-
  Non è stato rimosso nulla dall'API pubblica dei modelli: ogni classe e
  funzione che funzionava nella 1.4.0 si importa ancora. Quattro argomenti hanno
  cambiato forma e tre valori predefiniti spostano numeri con cui potresti fare
  confronti.
keywords:
  - aggiornare libreyolo
  - migrazione libreyolo 1.5.0
  - allow_experimental rimosso
  - libreyolo breaking changes
  - yolox bn eps
  - faster-coco-eval default
last_verified: 1.5.0
meta:
  - label: Si applica a
    value: Dalla 1.4.0 alla 1.5.0
  - label: Modifiche al codice richieste
    value: 'Quattro, tutte circoscritte'
  - label: Risultati che cambiano
    value: 'Backend COCO, eps di BN in YOLOX, multi-scala di D-FINE'
  - label: Rimozioni dall'API pubblica
    value: Nessuna
source_hash: ab38d8ef7b53f596
---

Questa pagina parla dell'aggiornamento di LibreYOLO stesso. Se cerchi come
caricare un checkpoint da un progetto upstream, quello è
[importare pesi esistenti](/docs/migrate), un argomento diverso.

La voce completa della release è il [changelog](/docs/changelog). Quello che
segue è solo la parte che richiede qualcosa da te.

## Modifiche al codice che devi fare

### `allow_experimental=True` non esiste più

Il gate di conferma non c'è più, insieme al meccanismo
`ddp_aware(experimental_key=...)` che lo implementava. L'addestramento e
l'esportazione di EC, RTMDet, PicoDet e FOMO in precedenza richiedevano
quell'argomento, quindi è interessato qualsiasi script che addestri una di
quelle famiglie.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: elimina l'argomento
model.train(data="data.yaml", epochs=100)
```

Non esiste nessuno shim di deprecazione. Una chiamata che lo passa ancora
solleva `TypeError`. Insieme a esso è stato rimosso
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`. L'hook `get_download_notice()`
sopravvive, ed è ancora sovrascritto da MiDaS, SegFormer e YOLO9-P2.

I livelli di supporto vengono ancora pubblicati, semplicemente non sono più un
argomento: vedi [livelli di stabilità](/docs/reference/stability-tiers).

### Il livello di esportazione `"experimental"` non esiste più

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Il codice che si dirama in base alla stringa del livello dovrebbe leggere
`"available"` dove leggeva `"experimental"`. `BaseExporter` non emette più un
`RuntimeWarning` per quei formati. Lo stato di ogni singolo formato è elencato
nella [matrice di esportazione](/docs/reference/export-matrix).

### `pretrained=False` insieme a `resume` ora viene rifiutato

Prima la combinazione proseguiva in modo incoerente. Ora solleva:

```
ValueError: pretrained=False cannot be combined with resume.
```

Scegline uno. `pretrained=False` parte da un'inizializzazione nuova con seed,
che nella 1.5.0 funziona per ogni famiglia addestrabile invece che per tre
soltanto, mentre `resume` riprende un'esecuzione interrotta a partire dal suo
checkpoint. Entrambi sono documentati in [addestramento](/docs/train).

### `--imgsz` della CLI è una stringa, non un int

È più circoscritto di quanto sembri. Nessuno di questi due casi è interessato:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # va ancora bene
```

```python
model.predict("img.jpg", imgsz=640)   # va ancora bene
```

Deve cambiare solo il codice che chiama le funzioni dei comandi della
[CLI](/docs/cli) direttamente da Python, perché `predict`, `train` e `val`
hanno allargato `--imgsz` da `int` a `str` in modo che possa accettare
dimensioni rettangolari:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, e ora funziona anche "480x640"
```

Il valore predefinito di `train` ora è la stringa `"640"`. `export --imgsz` era
già una stringa e `profile` non cambia.

## Numeri che cambiano

Tre modifiche spostano le metriche con le impostazioni predefinite. Se segui i
risultati da una versione all'altra, leggile prima di confrontare
un'esecuzione della 1.5.0 con una della 1.4.0.

### faster-coco-eval è il backend predefinito per le metriche COCO

`val()` e la validazione per epoca durante l'addestramento ora calcolano le
metriche COCO con il backend C++ faster-coco-eval invece che con pycocotools.

Il passaggio è stato deciso sulla base della parità misurata su tutti i 100
split di test di RF100-VL: 1381 valori di metrica su 1400 identici bit a bit,
deviazione massima 2.22e-16, delta principali esattamente 0, con una velocità
15,6x superiore in generale e 56x sui dataset densi di rilevamenti. I tuoi
numeri non dovrebbero cambiare. Sono comunque prodotti da un'implementazione
diversa, ed è per questo che compare in questo elenco.

pycocotools resta il fallback automatico quando faster-coco-eval non è
installato. Per forzarlo:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` fa la stessa cosa a livello globale. Il backend
effettivamente usato viene registrato a livello INFO, esposto come
`model.last_eval_backend` dopo `val()` e incluso come `eval_backend` nel
payload JSON della [CLI](/docs/cli/val). Installa il backend veloce con
`pip install libreyolo[fast-eval]`.

### I checkpoint YOLOX addestrati prima della 1.5.0 richiedono un override di eps

Questa è la trappola della release. Leggila se hai fatto fine-tuning di
[YOLOX](/docs/models/yolox).

YOLOX specifica BatchNorm con `eps=1e-3` e `momentum=0.03`. Fino alla 1.5.0
quei valori venivano applicati come una correzione a posteriori che non
sopravviveva alla ricostruzione per numero di classi che `train()` esegue
quando l'`nc` del tuo dataset è diverso da quello del checkpoint. Un
fine-tuning del genere si addestrava e riportava la validazione durante
l'addestramento con l'`eps=1e-5` predefinito di torch, per poi essere
ricaricato per l'inferenza a `1e-3`: gli stessi tensori sotto una
normalizzazione diversa.

Le taglie con convoluzioni normali cambiano appena. La `n` depthwise cambia
molto, perché il suo `running_var` per canale è abbastanza piccolo da far
dominare eps. Su `ball` di RF100-VL, lo stesso checkpoint nano ottiene
**0.566** di mAP50-95 valutato con l'eps con cui è stato addestrato e
**0.151** dopo un ricaricamento standard.

Un checkpoint addestrato prima della 1.5.0 porta con sé la semantica eps=1e-5.
Per riportare numeri fedeli, puoi valutarlo forzando l'eps di BN a 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

oppure incorporare una volta per tutte `sqrt((var + 1e-3) / (var + 1e-5))` nei
pesi di BN e salvare il risultato. I checkpoint addestrati con la 1.5.0 e
successive non richiedono né l'una né l'altra cosa.

### L'addestramento multi-scala di D-FINE usa la ricetta upstream per ogni taglia

`base_size_repeat` era fissato nel codice a 3 per ogni taglia. Ora viene
risolto per taglia come specifica l'upstream: la **n** si addestra a dimensione
fissa con il multi-scala disattivato, la **s** a 20, la **m** a 6, la **l** a
4, la **x** a 3. Prima corrispondeva solo la x, quindi n, s, m e l vedono una
distribuzione di scale diversa e convergono a metriche diverse.

Per ripristinare il vecchio comportamento, impostalo esplicitamente:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM usa ancora il 3 fissato nel codice. I dettagli della famiglia sono su
[D-FINE](/docs/models/d-fine).

## Da sapere, senza azioni da fare

- **I risultati con `imgsz` rettangolare sono cambiati perché prima erano
  sbagliati.** Le coordinate dei box, il ridimensionamento delle maschere di
  RTMDet, il riscalamento di YOLO-NAS e la scalatura del ground truth nel
  validatore ora usano altezza e larghezza per asse invece di un solo scalare.
  Con `imgsz` quadrato non cambia un bit. L'inferenza o la validazione
  rettangolare eseguite sulla 1.4.0 erano scalate male. YOLO-NAS ora rifiuta a
  priori un `imgsz` rettangolare invece di produrre in silenzio un output
  sbagliato.
- **I dizionari delle metriche hanno nuove chiavi.** `max_det`, `ar_max_det` e
  `AR_max_det` dal valutatore COCO, e `metrics/loss` più `metrics/loss/ce` da
  FOMO. Con i valori predefiniti i numeri non cambiano, ma tutto ciò che itera
  sulle chiavi delle metriche, inclusi i
  [logger](/docs/train/loggers) personalizzati e le intestazioni CSV, vede
  nuove colonne.
- **Le esecuzioni YOLO9 con seed che innescano una ricostruzione della testa**
  partono da un'inizializzazione diversa, perché il seed ora viene applicato
  prima della ricostruzione e non dopo. Un fine-tuning con seed fatto sulla
  1.4.0 verso un numero di classi diverso non è riproducibile bit per bit sulla
  1.5.0.
- **`libreyolo[hub-kernels]` su CUDA ora attiva davvero il kernel nativo
  MS-deform-attn.** La 1.4.0 lo teneva dietro una condizione che RF-DETR non
  soddisfaceva mai, quindi il kernel non veniva mai eseguito. Le predizioni possono
  spostarsi entro la tolleranza float per RF-DETR e per le altre famiglie ad
  attenzione deformabile. Le installazioni standard non sono interessate, e
  `LIBREYOLO_HUB_KERNELS=0` lo disattiva.
- **`libreyolo predict` scarta le opzioni non supportate invece di sollevare un
  errore.** La CLI filtra i kwargs rispetto alla firma di `__call__` del
  modello, quindi un'opzione che una famiglia non accetta viene ignorata invece
  di sollevare `TypeError`. Un errore di battitura nel nome di un flag ora
  viene ignorato in silenzio.
- **Le sorgenti live cambiano la forma dell'output JSON.** Le webcam, gli
  stream RTSP e la cattura dello schermo attivano implicitamente lo streaming,
  che emette un record per frame invece di uno per chiamata. Queste
  [sorgenti](/docs/predict/sources) sono nuove nella 1.5.0, quindi non è
  interessato nessuno script della 1.4.0.
- **Riesportare `rfdetr-pose` o `yolonas-pose` in ONNX produce nomi di output
  diversi.** La 1.4.0 interpretava male le loro teste di posa multi-tensore
  come segmentazione, attraverso un'euristica basata sul conteggio degli
  output. I file `.onnx` già presenti su disco restano intatti.
- **In un'installazione senza torch**, i risultati contengono array numpy
  invece di `torch.Tensor`, quindi `.boxes.data` restituisce un tipo diverso e
  la risoluzione dei pari merito nella NMS può differire da torchvision. Con
  torch installato il comportamento non cambia di un byte. Vedi
  [installazione leggera](/docs/lightweight-install).
- **Gli oggetti di configurazione eseguono più controlli al momento della
  costruzione.**
  `TrainConfig` ha ora un `__post_init__` che prima non aveva, quindi una
  configurazione già non valida solleva subito un errore invece di fallire a
  esecuzione inoltrata. La serializzazione di `ValidationConfig` ha una nuova
  chiave `edge_thresholds`, che rompe un round-trip rigoroso
  `ValidationConfig(**dump)` a partire da un dump della 1.4.0.
- **I nomi dei file dei pesi per le famiglie con suffisso di task si risolvono
  in modo diverso.** `segformer-b0` ora si risolve in
  `LibreSegformerb0-sem.pt`. Questo corregge i 404 del download automatico, e
  rompe qualsiasi script che aveva fissato nel codice il vecchio nome senza
  suffisso.
- **Il marker pytest `experimental_backend` ora è `extended_backend`.**
  Rilevante solo se esegui la suite di test con `-m`.

## Checkpoint e dataset

I checkpoint scritti dalla 1.4.0 si caricano senza modifiche. Lo
[schema](/docs/reference/checkpoint-schema) ha aggiunto `imgsz_h` e
`imgsz_w` per i modelli rettangolari, e scrive ancora lo scalare
`imgsz = max(h, w)` per i lettori più vecchi. Le esportazioni
[ExecuTorch](/docs/export/executorch) e [MNN](/docs/export/mnn) ora richiedono
un file sidecar, rispettivamente `<program>.pte.json` e `<model>.mnn.json`,
e le esportazioni HRNet portano `pose_input: "person_crop"`. I formati dei
dataset non cambiano.
