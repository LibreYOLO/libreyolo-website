---
title: Aggiornamento a 1.7.0
seo_title: Aggiornare LibreYOLO da 1.6.0 a 1.7.0
description: >-
  Passi di migrazione per i checkpoint YOLO9, i nomi dei moduli e le ricette di
  addestramento, le cartelle di esecuzione della CLI, la matrice di confusione e
  i nuovi errori in LibreYOLO 1.7.0.
lead: >-
  La versione 1.7.0 continua a caricare i vecchi checkpoint YOLO9 con gli stessi
  box in virgola mobile. Modifica l'addestramento della famiglia YOLO9, i nomi
  che il codice vede dentro la rete e alcuni contratti di validazione e
  predizione.
keywords:
  - aggiornare libreyolo
  - migrazione libreyolo 1.6.0
  - libreyolo breaking changes
  - yolo9 module names
  - clip_max_norm
  - confusion matrix
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## Da 1.6.0 a 1.7.0

- I checkpoint YOLO9, YOLO9-E2E e YOLO9-P2 salvati da 1.7.0 usano nuovi nomi dei tensori e non si caricano in 1.6.x: 1.6.x carica quelli YOLO9 e YOLO9-P2 con un avviso di chiavi mancanti e non predice nessun box, e solleva un errore di `model_family` non corrispondente per quelli YOLO9-E2E. Aggiorna ogni ambiente che li legge. I checkpoint float, quantizzati e di addestramento scritti dalle release precedenti si caricano in 1.7.0, e le esportazioni ONNX non sono interessate.

- Il codice che accede a `model.model` di un modello della famiglia YOLO9 vede i nomi dei moduli di MultimediaTechLab/YOLO, senza alias. `DDetect` è `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` è `RepConv`, `RepNBottleneck` è `Bottleneck`, `DFL` è `Anchor2Vec` (`anchor2vec`), `nc` è `num_classes`, e i sottolayer dei blocchi sono `conv1`/`conv2`/`bottleneck`. In YOLO9-E2E, `YOLO9E2EDetect` è `YOLO9E2EHead` e `one2one_cv2` / `one2one_cv3` sono `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` e `yolo9.utils.decode_boxes` sono rimossi. Il [changelog](/docs/changelog) elenca ogni ridenominazione.

- I nomi dei nodi ONNX seguono i nuovi nomi dei moduli nella testa, nel backbone e nel neck, per esempio `/head/anchor_convs.0/...`. Aggiorna gli strumenti che si basano sui nomi dei nodi, come le configurazioni dei nodi finali di Hailo. Input, output e nomi degli output del modello non cambiano.

- Un fine-tuning YOLO9 da pesi senza tensori PGI, che include i `LibreYOLO9{t,s,m,c}.pt` pubblicati, addestra solo la testa principale, come in 1.5.0; 1.6.0 collegava un ramo PGI inizializzato in modo casuale con `aux_weight=0.25`. I pesi che contengono tensori PGI, e le esecuzioni da zero, addestrano ancora con PGI. Passa `aux_weight=0.25` per collegare comunque un nuovo ramo, oppure `aux_weight=0` per tenerlo spento.

- Quando PGI è collegato, YOLO9-m e YOLO9-c costruiscono il ramo ausiliario di MultimediaTechLab: 12.8M (m) e 25.7M (c) di parametri usati solo nell'addestramento a 80 classi, contro 7.5M e 10.2M, quindi l'addestramento PGI di queste taglie richiede più memoria e più tempo per passo. Un checkpoint di addestramento m/c di 1.6.0 riprende e fa fine-tuning con il ramo con cui è stato addestrato. L'inferenza e YOLO9-t/s non cambiano.

- YOLO9, YOLO9-E2E e YOLO9-P2 limitano la norma L2 del gradiente a 10, e il loro assegnatore di addestramento valuta gli anchor in fp32, quindi i positivi scelti con AMP fp16 possono differire. Passa `clip_max_norm=0` per il clipping di 1.6.0; un'esecuzione iniziata in 1.6.0 e ripresa in 1.7.0 applica comunque il clipping a meno che tu non lo passi. L'assegnazione in fp32 non ha nessun interruttore.

- Un `perspective` diverso da zero deforma per spostamento degli angoli in ogni famiglia che usa la deformazione affine condivisa (YOLO9 e le sue varianti E2E e P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR e RT-DETRv2), comprese le esecuzioni riprese da un checkpoint di 1.6.0 che lo impostava. Valori da `0.002` in su danno la stessa deformazione massima, e i valori negativi sollevano `ValueError`. Nessuna impostazione riproduce la deformazione di 1.6.0; il valore predefinito `perspective=0` non cambia.

- La pipeline mosaic di YOLO9 estrae il rapporto di mixup da Beta(1, 1) invece che da Beta(32, 32), senza nessuna impostazione per il vecchio valore. L'addestramento predefinito di YOLOv7 cambia, perché YOLOv7 attiva il mixup di default. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR e RT-DETRv2 cambiano solo quando il mixup è attivo.

- `libreyolo train` senza `name=` scrive le esecuzioni EC, D-FINE, SegFormer e LingBot-Vision in `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` e `lingbotvision_exp` invece di `runs/train/exp`, le sonde V-JEPA 2 in `runs/train/train`, e le esecuzioni YOLO9-P2 in `runs/train/yolo9_exp`. Passa `name=exp` per il vecchio percorso, oppure `name=yolo9_p2_exp` per YOLO9-P2.

- `libreyolo train` fa partire D-FINE, SegFormer e LingBot-Vision dalle loro ricette. Per D-FINE sono 132 epoche, `lr0=2e-4`, `weight_decay=1e-4`, 2 epoche di warmup e nessuna augmentation con mosaic, mixup, HSV o affine. Per inviare i valori che inviava 1.6.0, passa `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` indicizza `matrix[predicted, true]` (era `[actual, predicted]`), conta i rilevamenti a `conf_thres=0.25` (era 0.15), prende `names=` e `task=` solo come argomenti keyword, e il suo `matrix` non può più essere riassegnato. Trasponi il codice che indicizzava la vecchia matrice, e passa `conf_thres=0.15` per i vecchi conteggi. `plots=True` scrive i conteggi in `confusion_matrix.png` e il grafico normalizzato in `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` sopprime i box sovrapposti tra classi invece di emettere un avviso. I task senza box sollevano `ValueError`, `LibreEnsemble` e HRNet sollevano `NotImplementedError`, e il tier SAM rifiuta la keyword con `TypeError`; togli l'argomento da quelle chiamate.

- Un `interpolation` di classificatore sconosciuto solleva `ValueError`, comprese le modalità torchvision `LANCZOS`, `BOX`, `HAMMING` e `NEAREST_EXACT`. Usa `bilinear`, `bicubic`, `nearest` o `bilinear_no_antialias`.

- Riesegui la validazione prima di confrontare con i numeri di 1.6.0 per `val(single_cls=True)` su un modello addestrato su più classi, i dataset di segmentazione che mescolano righe con solo box e righe con poligoni, e RTMDet-Ins, che mantiene fino a `max_det=300` maschere per immagine di default; passa `max_det=100` per l'impostazione di MMDetection.

- Senza `libreyolo[rfdetr]`, le chiamate della CLI con un nome di modello RF-DETR o DINOv2 escono con il codice di errore JSON `model_load_failed` invece di `model_not_found`, e il messaggio nomina `pip install libreyolo[rfdetr]`.

Vedi il [changelog](/docs/changelog) per la release completa.

## Da 1.5.0 a 1.6.0

- Aggiorna gli ambienti vincolati a versioni precedenti a ONNX Runtime 1.18 a `onnxruntime>=1.18.0` prima di installare l'extra ONNX.

- SAM 3D Body accetta l'insieme verificato di risorse dello snapshot. Installa `libreyolo[hf]`, ottieni l'accesso controllato e usa l'acquisizione automatica, oppure fornisci la directory invariata dello snapshot e la risorsa MHR fissata su un filesystem locale affidabile.

- Riesegui la validazione dopo le correzioni al ridimensionamento RF-DETR per i task diversi dalla posa e alla normalizzazione dei classificatori. Ricontrolla le soglie di confidenza in produzione prima di confrontare i risultati con 1.5.0; non esiste un flag per il preprocessing precedente.

- D-FINE, DEIM, RT-DETRv4 e il rilevamento YOLO-NAS attivano FP16 AMP di default. Passa `amp=False` per mantenere FP32.

- Per le precedenti impostazioni di fine-tuning YOLO9, imposta `aux_weight=0`, `max_labels=100` e `warmup_momentum=0.937`. Imposta `letterbox_pad="topleft"` quando una nuova conversione contrassegnata come centrata deve riprodurre la vecchia geometria. I vecchi checkpoint a testa singola riprendono con il grafo originale. Le nuove run YOLO9 usano più memoria GPU, quindi riduci `batch` se una dimensione del batch della 1.5.0 esaurisce la memoria.

- RF-DETR e DINOv2 creano directory di esecuzione incrementali con il nome della famiglia. Aggiorna i sistemi che usano i percorsi degli artefatti, oppure imposta `output_dir="runs/train", exist_ok=True` per mantenere il vecchio percorso e il comportamento di riutilizzo.

- Mantieni `mixup + cutmix <= 1` nella classificazione; le combinazioni non valide ora generano un errore durante la configurazione.

- QAT disattiva EMA, SyncBatchNorm e la media dei checkpoint. Usa i checkpoint best/last di QAT senza fare affidamento su questi stati.

- I loader personalizzati con hook che modificano il dataset devono usare `persistent_workers=False` o ricreare i worker dopo la modifica. Le copie persistenti incompatibili con più worker ora generano un errore.

- Gli array di immagini NumPy vengono letti come BGR, l'ordine di OpenCV. Passa `color_format="rgb"` dove passi un array RGB come `np.asarray(pil_image)`; l'output di `cv2.imread()` e i frame video non richiedono modifiche.

- Un array NumPy o un tensore 4D è un batch: `predict()` restituisce una lista con un `Results` per immagine invece di usare solo la prima immagine.

- `train(resume=True)` e `train(resume="path/to/last.pt")` ripristinano gli argomenti di addestramento salvati dell'esecuzione e continuano a scrivere nella sua directory di esecuzione; gli argomenti che passi esplicitamente hanno la precedenza. Riprendere pesi pubblicati, o un'esecuzione che ha già raggiunto le sue `epochs`, solleva `ValueError`.

- Con la validazione attiva, l'ultima epoca viene sempre validata, quindi le esecuzioni più brevi di `eval_interval` ora riportano le metriche e scrivono `best.pt`. `val=False` disattiva la validazione, ultima epoca inclusa. La validazione durante l'addestramento scrive in `<run>/val` invece che in `runs/val/`.

- Gli errori di `train` nella CLI riportano il proprio tipo di errore, quindi gli errori di configurazione escono con codice 2 invece che con 1 e `io_error`.

Vedi il [changelog](/docs/changelog) per la release completa e [importazione dei pesi](/docs/migrate) per la conversione dei checkpoint.

## Da 1.4.0 a 1.5.0

Questa pagina parla dell'aggiornamento di LibreYOLO stesso. Se cerchi come
caricare un checkpoint da un progetto upstream, quello è
[importare pesi esistenti](/docs/migrate), un argomento diverso.

La voce completa della release è il [changelog](/docs/changelog). Quello che
segue è solo la parte che richiede qualcosa da te.

### Modifiche al codice che devi fare

#### `allow_experimental=True` non esiste più

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
riceve un avviso `Unknown training config keys (ignored)`, e l'argomento non ha
alcun effetto. Insieme a esso è stato rimosso
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`. L'hook `get_download_notice()`
sopravvive, ed è ancora sovrascritto da MiDaS, SegFormer e YOLO9-P2.

I livelli di supporto vengono ancora pubblicati, semplicemente non sono più un
argomento: vedi [livelli di stabilità](/docs/reference/stability-tiers).

#### Il livello di esportazione `"experimental"` non esiste più

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Il codice che si dirama in base alla stringa del livello dovrebbe leggere
`"available"` dove leggeva `"experimental"`. `BaseExporter` non emette più un
`RuntimeWarning` per quei formati. Lo stato di ogni singolo formato è elencato
nella [matrice di esportazione](/docs/reference/export-matrix).

#### `pretrained=False` insieme a `resume` ora viene rifiutato

Prima la combinazione proseguiva in modo incoerente. Ora solleva:

```
ValueError: pretrained=False cannot be combined with resume.
```

Scegline uno. `pretrained=False` parte da un'inizializzazione nuova con seed,
che nella 1.5.0 funziona per ogni famiglia addestrabile invece che per tre
soltanto, mentre `resume` riprende un'esecuzione interrotta a partire dal suo
checkpoint. Entrambi sono documentati in [addestramento](/docs/train).

#### `--imgsz` della CLI è una stringa, non un int

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

### Numeri che cambiano

Tre modifiche spostano le metriche con le impostazioni predefinite. Se segui i
risultati da una versione all'altra, leggile prima di confrontare
un'esecuzione della 1.5.0 con una della 1.4.0.

#### faster-coco-eval è il backend predefinito per le metriche COCO

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

#### I checkpoint YOLOX addestrati prima della 1.5.0 richiedono un override di eps

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
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

oppure incorporare una volta per tutte `sqrt((var + 1e-3) / (var + 1e-5))` nei
pesi di BN e salvare il risultato. I checkpoint addestrati con la 1.5.0 e
successive non richiedono né l'una né l'altra cosa.

#### L'addestramento multi-scala di D-FINE usa la ricetta upstream per ogni taglia

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

### Da sapere, senza azioni da fare

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
  di sollevare `TypeError`. Un nome di flag sconosciuto viene comunque
  rifiutato con `No such option`.
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

### Checkpoint e dataset

I checkpoint scritti dalla 1.4.0 si caricano senza modifiche. Lo
[schema](/docs/reference/checkpoint-schema) ha aggiunto `imgsz_h` e
`imgsz_w` per i modelli rettangolari, e scrive ancora lo scalare
`imgsz = max(h, w)` per i lettori più vecchi. Le esportazioni
[ExecuTorch](/docs/export/executorch) e [MNN](/docs/export/mnn) ora richiedono
un file sidecar, rispettivamente `<program>.pte.json` e `<model>.mnn.json`,
e le esportazioni HRNet portano `pose_input: "person_crop"`. I formati dei
dataset non cambiano.
