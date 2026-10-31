---
title: Validazione e metriche
seo_title: Validazione e metriche in LibreYOLO
description: >-
  Esegui val() su qualsiasi modello, leggi le chiavi delle metriche che ogni
  task restituisce, scegli un backend di valutazione e attiva una loss di
  validazione accanto alla metrica di accuratezza.
lead: >-
  La validazione fa passare un modello su uno split di un dataset tramite val()
  e restituisce un dizionario piatto di chiavi di metriche e valori float. Le
  chiavi sono stringhe letterali, e quali ottieni dipende dal task, non dalla
  famiglia.
keywords:
  - map50-95
  - valutazione coco
  - metriche di validazione
  - faster-coco-eval
  - pycocotools
  - validation loss
  - miou
  - panoptic quality
  - top1 accuracy
  - confusion matrix yolo
  - class-agnostic nms
last_verified: 1.6.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Su un altro split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Scrivere predizioni in formato COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
  confusion:
    - label: Matrice di confusione
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")

        metrics = model.val(data="coco8.yaml")


        cm = metrics.confusion_matrix

        print(cm.matrix.shape)      # (81, 81): 80 classi più background

        print(cm.class_accuracy())  # nome della classe: quota di oggetti
        trovati a cui è data quella classe

        tp, fp = cm.tp_fp()
  curves:
    - label: Curve per classe
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        box = metrics.box
        print(box.px.shape, box.f1_curve.shape)  # (1000,) e (classi, 1000)
        for index, f1 in zip(box.ap_class_index, box.f1):
            print(model.names[int(index)], round(float(f1), 3))
source_hash: a5704354163337cf
---

## Eseguire una validazione

`val()` prende il dataset e restituisce le metriche.

<code-tabs name="val" />

Il valore restituito è un `dict` piatto di chiavi di metriche e numeri. Ogni
chiave è letterale, quindi leggila per nome e non per posizione. Per detect,
segment e classify è una sottoclasse di `dict` che porta anche i risultati per
classe come attributi, descritti più sotto.

Gli argomenti principali sono `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` e `verbose`. `conf` vale `0.001` di
default e `iou` vale `0.6`, entrambi molto più permissivi dei default della
predizione, perché uno sweep di mAP ha bisogno della coda a bassa confidenza.
`imgsz` di default vale la dimensione di input del modello stesso invece di un
numero fisso. `split` accetta `val`, `test` o `train` e nient'altro. Senza
`data`, un checkpoint addestrato viene validato sul dataset su cui è stato
addestrato; i pesi pubblicati non ne portano nessuno e richiedono `data=`.

Qualsiasi altro campo della configurazione di validazione passa come argomento
keyword, inclusi `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache`,
`save_plots`, `single_cls` e `agnostic_nms`. Con `half=True` su CUDA, la
validazione gira sotto autocast in `amp_dtype`, e gli output bfloat16 vengono
convertiti in float32 prima del postprocessing; anche l'addestramento con
`amp_dtype="bfloat16"` valida allo stesso modo.

`agnostic_nms=True` mantiene solo il box con il punteggio più alto tra i box che
si sovrappongono oltre `iou`, qualunque sia la classe. Agisce sui rilevamenti
finiti della famiglia, dopo `classes=`, quindi mAP, matrice di confusione, curve e
conteggi per immagine valutano tutti gli stessi box. Copre detect e segment, e
qualsiasi altro task solleva `ValueError`. Il lato predizione è in
[Soglie e filtraggio](/docs/predict/thresholds).

`single_cls=True` valuta il rilevamento come una sola classe. La ground truth
viene ridotta alla classe 0, e i rilevamenti del modello sono mantenuti secondo
`classes=` nello spazio di etichette del modello, poi rietichettati 0, così un
checkpoint addestrato su più classi viene valutato come un rilevatore a classe
singola.

## Chiavi delle metriche per task

Il rilevamento di oggetti restituisce la famiglia di numeri COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Due di queste sono trappole. `metrics/precision` e `metrics/recall` sono alias
tenuti per retrocompatibilità: contengono i valori di mAP 50-95 e AR@100, non una
coppia di precisione e recall. Usa le chiavi con il nome esplicito.

La segmentazione di istanze restituisce i valori di mAP e AR visti sopra come
numeri sulle maschere sotto le chiavi senza suffisso, con le versioni sui box
sotto un suffisso `(B)` e le versioni sulle maschere ripetute sotto `(M)`.
Precisione e recall esistono solo in forma suffissata per questo task, come
`metrics/precision(B)`/`metrics/recall(B)` e
`metrics/precision(M)`/`metrics/recall(M)`, ed entrambe le coppie contengono gli
stessi valori alias di detect: la coppia `(B)` è mAP50-95 sui box e AR@100 sui
box, la coppia `(M)` è mAP50-95 sulle maschere e AR@100 sulle maschere.

| Task | Chiavi |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, più le suddivisioni per dimensione e per recall viste sopra |
| segment | versioni sulle maschere delle chiavi di detect viste sopra (le chiavi senza suffisso sono quelle sulle maschere); `precision`/`recall` esistono solo come `(B)`/`(M)`, entrambe con lo stesso schema di alias |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, e le corrispondenti chiavi `keypoints_AR` |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, più le copie con suffisso `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, più una chiave di sweep mAP |

`metrics/precision` e `metrics/recall` di OBB non sono alias: sono la precisione
e il recall veri a IoU 0.50, presi nel punto di lavoro più permissivo (ogni
predizione che sopravvive a `conf`, di default `0.001`). Le copie con suffisso
`(OBB)` ripetono gli stessi quattro valori sotto un nome specifico del task, con
la stessa convenzione vista sopra per `(B)` e `(M)`.

`accuracy_top5` è in realtà top-`min(5, num_classes)`, quindi su un dataset a tre
classi è top-3, che ogni campione soddisfa e che perciò vale 1.0.
I risultati di classificazione espongono i due valori anche come attributi `top1`
e `top5`, accanto a `confusion_matrix`.

La chiave di sweep del task point è costruita a partire dalle soglie di distanza,
quindi con i default vale `metrics/mAP@[0.01:0.10]` e la chiave a soglia singola
vale `metrics/mAP@0.01`. Passare `dist_thresholds` cambia entrambe le stringhe.

Quasi tutti i task restituiscono anche una chiave `fitness`, il singolo numero su
cui la selezione del checkpoint migliore si basa di default. Rilevamento,
segmentazione e OBB non ne hanno una; le loro famiglie sono selezionate su
`metrics/mAP50-95`, che i loro dizionari restituiscono. Pose non restituisce né
`fitness` né `metrics/mAP50-95`; i suoi trainer impostano invece `best_metric_key`
a `metrics/keypoints_mAP50-95`.

La classificazione ImageFolder aggiunge le metriche macro `metrics/precision`, `metrics/recall` e `metrics/f1`, mediate sulle classi presenti nei target di validazione. Top-1 resta la fitness predefinita. Il rilevamento restituisce anche `metrics/best_conf` e `metrics/best_conf_f1`, scegliendo la soglia ottimale per micro-F1 a IoU 0.50, e mette le soglie con nomi di classe come chiavi su `metrics.box.best_conf_per_class`. I rilevamenti con lo stesso punteggio restano raggruppati; in caso di parità viene scelta la soglia più alta. Se nessun F1 è positivo, viene restituito 0.0. La segmentazione non espone queste chiavi di soglia.

## Matrice di confusione

I risultati di detect, segment e classify portano `results.confusion_matrix`. Viene
calcolata a ogni validazione, che i plot vengano scritti o no, e la sua `matrix` è
un array di interi indicizzato `matrix[predicted, true]`.

<code-tabs name="confusion" />

Per detect e segment, le predizioni con confidenza 0.25, o `conf` se più alta,
vengono accoppiate uno a uno con i box di ground truth di qualsiasi classe a IoU
0.5, partendo dall'IoU più alto. Una coppia conta alla sua classe predetta e alla
sua classe vera, quindi un box ben posizionato con la classe sbagliata finisce
fuori dalla diagonale. La matrice ha una riga e una colonna `background` in più:
una ground truth non accoppiata conta nella riga `background`, una predizione non
accoppiata nella colonna `background`. La segmentazione accoppia i box, non le
maschere. Con `classes=`, la ground truth delle altre classi viene esclusa.

Le regioni COCO crowd (`iscrowd=1`) seguono la regola che applica mAP. Una
predizione che nessuna ground truth della sua classe ha accoppiato, e che si trova
per almeno metà dentro una regione crowd della stessa classe (intersezione sull'area
della predizione stessa), viene ignorata invece che contata. `box.image_metrics` e
le immagini di `visualize=True` usano le stesse soglie e la stessa regola crowd. Le
etichette in formato YOLO non hanno regioni crowd.

Per classify, ogni immagine conta una volta, alla sua classe top-1 rispetto alla
sua etichetta, e non c'è nessun `background`.

| Metodo | Restituisce |
|---|---|
| `summary(normalize=False)` | Un dict per classe predetta, `{"Predicted": nome, <classe vera>: valore}`; con `normalize`, le quote di ogni classe vera |
| `to_csv()`, `to_json()` | Il riepilogo come testo CSV o JSON |
| `to_df()` | Il riepilogo come DataFrame Polars; richiede `pip install polars` |
| `tp_fp()` | Veri positivi per classe (la diagonale) e falsi positivi (il resto della riga predetta) |
| `nonzero()` | Le celle riempite come array allineati `(predicted, true, count)` |
| `class_accuracy()` | Per classe, la quota delle sue istanze a cui è data quella classe; per il rilevamento, solo dei suoi oggetti trovati |
| `plot(normalize=True, save_dir="")` | Scrive `confusion_matrix_normalized.png`, o `confusion_matrix.png` con `normalize=False`, e restituisce il percorso |

Sopra le 2.000 classi la matrice densa di classificazione non viene costruita:
`matrix`, `summary()`, le esportazioni `to_*()` e `plot()` sollevano un errore,
mentre `tp_fp()`, `nonzero()` e `class_accuracy()` funzionano ancora.

La classe è `libreyolo.validation.ConfusionMatrix`, costruita come
`ConfusionMatrix(nc, iou_thres=0.5, conf_thres=0.25, *, names=None, task="detect")`,
dove `task` è `"detect"` o `"classify"`.

## Precisione, recall e F1 per classe

I risultati di detect e segment portano precisione, recall e F1 di ogni classe al
variare delle soglie di confidenza, con abbinamento a IoU 0.5.

<code-tabs name="curves" />

`box.px` contiene le 1000 confidenze, equispaziate su `[0, 1]`. `box.p_curve`,
`box.r_curve` e `box.f1_curve` sono array `(classi, 1000)`, una riga per ogni voce
di `box.ap_class_index`, gli indici di classe che hanno ground truth nello split,
in ordine crescente. `box.p`, `box.r` e `box.f1` sono i valori per classe alla
confidenza che massimizza l'F1 medio su quelle classi. La segmentazione li calcola
tutti dai box. Sono array vuoti quando l'esecuzione non ha prodotto curve.

## Metriche dei box per immagine

I risultati di rilevamento e segmentazione restano compatibili con i dizionari ed espongono anche `results.box.image_metrics`. Ogni nome file corrisponde a `precision`, `recall`, `f1`, `tp`, `fp` e `fn`, usando la regola di abbinamento della visualizzazione anche quando è disattivata. Qui la segmentazione conta i box. Per nomi di base duplicati, dopo la prima occorrenza vengono usati i percorsi completi. I denominatori nulli producono 0.0. Questi record non vengono raccolti tra rank distribuiti.

## Chiavi di velocità

Ogni validatore aggiunge i tempi:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Sono millisecondi per immagine mediati sull'intera esecuzione. Descrivono la
macchina e le impostazioni con cui hai eseguito, quindi un numero preso da lì ha
senso solo se riportato insieme al suo hardware, alla dimensione del batch e alla
precisione.

## Backend di valutazione

Le metriche di rilevamento e di segmentazione vengono calcolate tramite un
evaluator COCO, e `faster_coco_eval=True`, il default, seleziona il backend C++
quando il pacchetto `faster-coco-eval` è installato. Quando non lo è,
l'esecuzione ripiega su pycocotools con un solo avviso per processo:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Il backend effettivamente usato viene registrato sul modello come
`last_eval_backend`, e la CLI lo riporta nel proprio output per i task in stile
rilevamento. Imposta `LIBREYOLO_FASTER_COCO_EVAL` per sovrascrivere dall'ambiente
il valore della configurazione.

`iou_thresholds` è rispettato solo sul percorso OBB. Il percorso COCO valuta con
il proprio sweep fisso da 0.50 a 0.95 e ignora il valore.

## Loss di validazione

Di default la validazione riporta solo l'accuratezza. `val_loss=True` calcola
anche l'obiettivo di addestramento della famiglia sui batch di validazione.

<code-tabs name="valloss" />

Emette `metrics/loss` più un `metrics/loss/<component>` per ogni termine, pesato
esattamente come lo pesa l'addestramento, così le componenti sommate danno il
totale. Tramite un logger compaiono come `val/loss` e `val/loss/<component>`, e
`libreyolo monitor` sovrappone `metrics/loss` a `train/loss`.

Le componenti sono quelle proprie di ogni famiglia:

| Task | Famiglie | Componenti |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

È disattivata di default perché l'assegnazione dei target aggiunge tempo e memoria
alla validazione. Il validatore riusa l'output del modello già prodotto per la
metrica di accuratezza invece di eseguire un secondo forward pass, opera sotto
`no_grad` sul modello di valutazione o EMA, e nell'addestramento multi-GPU la
calcola localmente sul rank 0, senza operazioni collettive. La selezione del
checkpoint migliore resta sulla metrica di accuratezza.

Tre cose che deliberatamente non fa. Non include mai i termini di contrastive
denoising, perché hanno bisogno del ground truth al momento del forward e la
validazione fa il forward senza. Riporta i numeri del modello in modalità
valutazione,
quindi dove il forward di train e quello di eval di una famiglia differiscono
davvero, nelle statistiche di BatchNorm o nella stochastic depth, il numero
riflette la modalità eval; è questo il confronto voluto. E un task per cui una
famiglia non l'ha implementata solleva un errore di configurazione al setup invece
di saltarla in silenzio:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO è l'eccezione che non cambia nulla: il suo validatore ha sempre calcolato
questa loss, e `val_loss=True` incide solo sulle chiavi sotto cui viene
pubblicata.

La validazione con augmentation e la loss di validazione non si possono
combinare, e chiederle entrambe solleva un errore.

## File che una validazione scrive

`val()` scrive sempre `config.yaml` nella propria directory di salvataggio, che
di default è `runs/val/<model>_<size>_<timestamp>` quando `save_dir` non è
indicato. `project`, `name` e `exist_ok` la scelgono come fa l'addestramento:
`project/name`, con un suffisso incrementale a meno che `exist_ok=True`.

<code-tabs name="json" />

`save_json=True` scrive `predictions.json` per il rilevamento, e
`predictions_bbox.json` più `predictions_masks.json` per la segmentazione. OBB non
lo supporta e lo dice.

`save_plots=True`, o il suo alias `plots=True`, scrive in una sottodirectory
`plots/` e richiede matplotlib, che `pip install "libreyolo[plots]"` installa. Per
il rilevamento vengono scritti `box_metrics.png`, i grafici di AP e recall per
classe, le curve precision-recall e di confidenza, le curve F1 per classe in
`f1_conf_box.png`, la matrice di confusione come conteggi in
`confusion_matrix.png` e come quote di ogni classe vera in
`confusion_matrix_normalized.png`, e immagini di esempio annotate quando OpenCV è
installato. La segmentazione aggiunge copie lato maschera del grafico delle
metriche, dei grafici per classe e delle curve precision-recall e di confidenza;
le curve F1 e la matrice di confusione restano a livello di box. Pose ha il
proprio set di metriche e di curve. La classificazione scrive i due file della
matrice di confusione, tranne sopra le 2.000 classi. Gli altri validatori non
implementano i plot; semantic, panoptic, depth, normal, edge, restore, matte, OCR,
OBB e point non scrivono nulla lì. Un errore nel plotting genera un avviso e non
interrompe mai l'esecuzione.

`visualize=True` salva immagini con box TP/FP/FN distinti per classe per rilevamento e segmentazione, oppure immagini con etichetta confrontata con top-1 per la classificazione ImageFolder, in `visualize/errors/` e `visualize/correct/`. L'abbinamento usa IoU 0.5 e confidenza `max(0.25, conf)`. I valori predefiniti sono `visualize=False`, `show_labels=True` e `show_conf=True`. I task non supportati e la validazione delle clip V-JEPA 2 rifiutano la visualizzazione.

`plot_samples=8` limita il grafico separato dei campioni; 0 lo disattiva e -1 mantiene tutte le immagini. Questo non modifica le metriche né l'output di visualizzazione.

## Validazione durante l'addestramento

L'addestramento valida ogni `eval_interval` epoche, e sempre dopo l'ultima
epoca, sullo split `val` del dataset, e le metriche che produce sono ciò che
guida la selezione di `best.pt`, l'early stop di `patience` e le chiavi `val/`
in ogni logger. La validazione viene eseguita sui pesi EMA quando l'EMA è
attiva. I suoi file vanno in una directory `val/` dentro l'esecuzione.
`val=False` disattiva la validazione durante l'addestramento, ultima epoca
inclusa.

Vedi [Iperparametri](/docs/train/hyperparameters) per `eval_interval`, `patience`
e `save_plots`, e [Logger degli esperimenti](/docs/train/loggers) per dove finiscono
i numeri.

## Correlati

- [Dataset](/docs/train/datasets) per le chiavi degli split e i formati che i validatori leggono.
