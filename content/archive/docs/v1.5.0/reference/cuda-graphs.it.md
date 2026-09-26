---
title: Grafi CUDA
seo_title: Matrice di supporto dei grafi CUDA in LibreYOLO
description: >-
  Quali famiglie catturano il forward quando fanno predizioni e forward e
  backward quando addestrano, cosa è garantito sui numeri, dove una cattura
  viene divisa e perché una famiglia non supportata solleva un errore.
lead: >-
  Un grafo CUDA registra un'esecuzione di una sequenza fissa di kernel e la
  riproduce come un unico lancio. LibreYOLO cattura l'inferenza su 39 famiglie
  verificate e l'addestramento su 24, sempre per famiglia, sempre dopo un
  controllo di parità bit a bit e mai come fallback silenzioso.
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - grafi cuda pytorch
  - accelerare inferenza yolo cuda graph
  - addestrare yolo con cuda graph
  - capture_error_mode thread_local
last_verified: 1.5.0
verification: >-
  Elenco delle famiglie di inferenza derivato dalla matrice CAPTURABLE in
  tests/e2e/test_cuda_graph_families.py alla v1.5.0. Elenco delle famiglie di
  addestramento, classi di parità e tempi da docs/training_cuda_graphs.md. API e
  NotImplementedError da BaseModel._require_cuda_graph_support, cuda_graph_scope
  e capture_graph in libreyolo/models/base/model.py, con la variabile di classe
  SUPPORTS_CUDA_GRAPH. Divisioni sulle giunzioni lette dagli override di
  _get_graph_runner nelle famiglie depth_anything3, birefnet, ppocr, sam e
  sensenova e da libreyolo/models/base/detr_cuda_graph.py. capture_error_mode da
  libreyolo/models/base/cuda_graph.py e libreyolo/training/cuda_graph.py.
  Fallback di addestramento da libreyolo/training/trainer.py e flag --cuda-graph
  da libreyolo/cli/commands/train.py.
meta:
  - label: Famiglie di inferenza
    value: '39'
  - label: Famiglie di addestramento
    value: '24'
  - label: Flag di inferenza
    value: predict(cuda_graph=True)
    mono: true
  - label: Flag di addestramento
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predizione
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # True cattura al primo uso per ogni forma di input.

        # "auto" aspetta che una forma si ripeta prima di pagare il costo della
        cattura.

        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Addestramento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Addestramento da CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Cosa viene catturato

Un grafo registra una sequenza fissa di kernel e gli indirizzi di memoria che
leggono e scrivono. Non registra valori, forme o flusso di controllo. La
riproduzione è un unico lancio invece di centinaia, ed è per questo che il
guadagno è massimo su reti piccole con batch piccoli, dove uno step è dominato
dall'overhead di lancio più che dall'aritmetica.

I due punti di ingresso catturano quantità di lavoro diverse.

| | Nel grafo | Eager |
|---|---|---|
| Inferenza | Il forward della rete, `model._forward(x)` | Preprocessing, NMS, tutto il postprocessing |
| Addestramento | Il forward e il backward della rete | Loss, step dell'optimizer, gradient clipping, EMA, schedule del learning rate |

Né l'NMS né la loss di detection sono candidati. Entrambi selezionano con
maschere booleane, eseguono il matching ungherese o un assigner e si diramano sul
risultato, che è esattamente ciò che un grafo non può registrare. Tenerli fuori è
ciò che rende la cattura sicura, invece di un limite da aggirare.

<code-tabs name="usage" />

`cuda_graph` accetta tre valori quando fai predizioni. `False` è il valore
predefinito. `True` cattura la prima volta che vede ciascuna forma di input.
`"auto"` aspetta che una forma si ripeta, così il lavoro one-shot o a forma
variabile non paga mai una cattura che non riutilizzerà.
`capture_graph(imgsz=None, batch=1, dtype=None)` sposta il costo fuori dalla
prima richiesta, `graph_info()` riporta i grafi catturati e il numero di
riproduzioni e `release_graphs()` li libera.

Durante l'addestramento il flag è un semplice booleano, `--cuda-graph` da CLI.
Vedi [prestazioni in predizione](/docs/predict/performance) e
[prestazioni in addestramento](/docs/train/performance) per i controlli che
stanno intorno.

## Supporto in inferenza

Il supporto è per famiglia, dichiarato attraverso la variabile di classe
`SUPPORTS_CUDA_GRAPH`, e una famiglia viene marcata solo dopo che cattura e
riproduce in modo bit-identico su due input di prova presi da distribuzioni
diverse. Questa matrice di parità condivisa copre 39 famiglie su nove task.

| Task | Famiglie |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Diverse famiglie compaiono sotto più di un task, quindi la matrice ha più righe
che famiglie distinte. Altre tre famiglie catturano attraverso percorsi di codice
specifici, con test dedicati propri invece che attraverso la matrice condivisa, e
non fanno parte delle 39: PP-OCR, SAM e SenseNova.

La verifica è bit a bit, non approssimata. Una versione precedente del protocollo
giudicava la parità in base alla magnitudine relativa e ha declassato per errore
tre famiglie sane, YOLOX, EfficientNetV2 e YOLOv7, la cui differenza tra eager e
grafo misura circa 1e-7 pur restando bit-identica sulla prova che conta.

## Supporto in addestramento

La cattura in addestramento è passata da due famiglie a 24 in questa release, su
cinque task.

| Task | Famiglie |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Tutto il resto si addestra in eager: gli altri task sulle stesse famiglie, le
famiglie non elencate, le esecuzioni distribuite e quelle di distillazione. La
cattura viene saltata anche finché una forma è ancora nuova, perché il percorso
di addestramento aspetta che una forma di input si ripeta tre volte prima di
catturare, il che significa che con `multi_scale=True` potrebbe non catturare
mai.

## Due risposte diverse per una famiglia non supportata

Il percorso di inferenza solleva un errore. `predict(cuda_graph=True)` su una
famiglia che non ha aderito solleva `NotImplementedError` con il nome della
famiglia, invece di eseguire in eager lasciandoti credere di aver ottenuto uno
speedup che non hai ottenuto. Il motivo è che una cattura sbagliata non fallisce
in modo rumoroso: la riproduzione di un forward che fa qualcosa di non
catturabile restituisce numeri sbagliati in silenzio, quindi il supporto deve
essere un'affermazione esplicita per famiglia e non un tentativo con fallback.

Il percorso di addestramento scrive un log. `train(cuda_graph=True)` si può
passare sempre senza rischi, e una famiglia, un task o una configurazione che non
si può catturare scrive una riga e addestra in eager, senza cambiare nulla. Anche
una cattura che fallisce a metà di un'esecuzione fa scendere in eager il resto
dell'esecuzione invece di interromperla. L'asimmetria è voluta: una predizione è
una chiamata che puoi correggere sul posto, mentre un'esecuzione di addestramento
non deve morire alla sesta ora per un'ottimizzazione facoltativa.

## Divisione sulle giunzioni

Alcune famiglie non si possono catturare intere perché una fase fa davvero
qualcosa che un grafo non può registrare. Invece di rinunciare alla famiglia, la
cattura viene divisa su una giunzione verificata: la parte catturabile viene
riprodotta, il resto gira in eager e l'output combinato è lo stesso che si
ottiene eseguendo tutto in eager.

| Famiglia | Catturato | Eager, e perché |
|---|---|---|
| Depth Anything 3 | La rete | Lo step del cielo, che è lavoro visibile all'host dopo il forward |
| BiRefNet | L'encoder, `forward_enc` | Il decoder, il cui `deform_conv2d` sotto cattura riproduce un risultato diverso |
| PP-OCR | La fase di detection, `forward_det` | Il riconoscimento, perché la larghezza dei ritagli varia da riga a riga |
| SAM | L'encoder di immagini | Il percorso dei prompt, che gira molte volte per ogni encode |
| SenseNova | La vision tower | La generazione autoregressiva, con una cache KV che cresce a ogni step |
| Rilevatori encoder-decoder | Backbone ed encoder | Decoder e criterio ungherese |

La divisione di BiRefNet merita una seconda lettura: il comportamento anomalo di
`deform_conv2d` sotto cattura si riproduce con una chiamata isolata, fuori da
qualsiasi modello. Sostituirlo con un equivalente in puro PyTorch è stato
scartato perché avrebbe spostato anche le predizioni eager, e i numeri eager sono
il contratto.

Il caso encoder-decoder copre D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 ed EC. Il loro decoder costruisce query di contrastive denoising a
partire dal ground truth, e il numero di queste query dipende dal conteggio di
ground truth più grande nel batch, quindi il numero di token del decoder cambia
da un batch all'altro. Questa è l'unica cosa che un grafo non può tollerare.
Backbone più encoder sono all'incirca un quinto o un quarto di uno step per
queste famiglie, ed è per questo che stanno in fondo alla tabella degli speedup.

PP-OCR cattura un grafo per ogni forma di input della detection, entro il limite
di cache del runner, e restituisce il risultato eager quando non c'è nessuno
scope di cattura attivo.

## Numerica

La maggior parte delle famiglie è bit-identica e, dove non lo è, il motivo viene
detto invece che accennato. Allo step zero dell'addestramento la loss è
bit-identica per tutte e 24 le famiglie e nessun buffer di BatchNorm differisce;
è il confronto dei gradienti a separare le categorie.

| Classe | Famiglie | Significato |
|---|---|---|
| Esatta | La maggior parte delle 24 | Ogni gradiente bit-identico |
| 1 ULP | fomo, lingbotvision | L'ultimo bit del float32, circa 1e-7 relativo, da un ordine di somma diverso |
| Rumore eager | La linea DETR | Il grafo differisce dall'eager non più di quanto due esecuzioni eager differiscano tra loro |
| Arrotondamento float | rtmdet | 137 gradienti su 139 bit-identici, due differiscono di circa 3e-4 |
| Stream RNG proprio | segformer | La stochastic depth sta dentro la regione catturata |

La classe del rumore eager è quella da leggere nel modo giusto. Per quelle
famiglie due esecuzioni eager con lo stesso seed già non concordano, quindi
bit-identico non è un'asticella che l'esecuzione con grafo ha mancato: è
un'asticella che nessuno supera. Vale più in generale con `amp=False`, dove un
non determinismo relativo misurato di 3.2e-7 su un gradiente dei pesi in fp32 si
accumula: due esecuzioni eager di YOLOv9-t con lo stesso seed divergono del 36
percento in 20 step, e disattivare TF32 non risolve.

## Pin memory

La cattura gira con `capture_error_mode="thread_local"`. Con la modalità
`"global"` predefinita di PyTorch, il thread di pin memory del DataLoader che
prepara il batch successivo chiama `cudaHostAlloc`, che invalida la cattura in
corso e allo stesso tempo ne viene avvelenato, così l'esecuzione muore al
prelievo del batch successivo con un errore sollevato dall'interno del thread di
pin memory. Questa combinazione è stata osservata due volte in una campagna di
addestramento reale prima di essere diagnosticata.

La modalità thread-local limita solo il thread che cattura. Il thread di pin non
tocca mai lo stream di cattura, quindi niente di ciò che fa dovrebbe finire nel
grafo. L'addestramento va oltre e sostituisce temporaneamente una sottoclasse di
`torch.cuda.CUDAGraph` che forza la modalità, perché `make_graphed_callables` non
espone nessun argomento per farlo, sotto un lock in modo che due catture
concorrenti non possano lasciare installata la sostituzione.

## Quanto vale

Misurato su una RTX 5070 Ti con AMP, un processo per braccio, riproducendo un
batch reale così il dataloader resta fuori dal giro, il più veloce di 24 step
dopo il warm-up. Detection a 640 px, classificazione a 224 px.

| Famiglia | Batch | Speedup |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Tutto il resto | variabile | da 1.04x a 1.26x |

Un'esecuzione intera guadagna meno, perché un grafo non può accelerare il
dataloader né la validazione. Un fine-tuning di YOLOv9-t di 20 epoche su 406
immagini è passato da 428.4 s a 367.7 s, un guadagno end-to-end di 1.16x, con un
mAP50-95 identico di 0.6394 nei due bracci e loss per epoca identiche.

Il tetto è fissato da quanta parte di uno step è rete. Sullo stesso hardware a
640 px e batch 8 è l'84 percento per YOLOv9-t ma solo il 26 percento per
RTMDet-t, che passa la maggior parte di uno step nel suo label assigner.
L'overhead di lancio è massimo su Windows, quindi su Linux i guadagni si
attestano all'incirca su un terzo o metà di questa tabella, e un'esecuzione
limitata dal dataloader non vede nessun cambiamento di tempo reale. La memoria di
picco si muove tra il 5 percento in meno e il 19 percento in più.

## Avvertenze

Un grafo registra indirizzi, non valori, quindi qualsiasi cosa sposti i parametri
lo fa cadere. Cambiare dispositivo con `predict(device=...)`, quantizzare e
dequantizzare invalidano tutti i grafi catturati.

La dimensione del batch conta più della famiglia: RT-DETR-r18 guadagna 1.19x con
batch 2 e 1.04x con batch 8, perché un batch grande è limitato dal calcolo e ha
meno overhead di lancio da togliere.

La suite di parità in inferenza è stata eseguita senza il pacchetto opzionale
`kernels` installato, quindi non copre la sicurezza della cattura con i kernel
compilati dell'Hub attivi. Imposta `LIBREYOLO_HUB_KERNELS=0` per toglierli di
mezzo mentre isoli un problema di cattura. Vedi
[kernels](/docs/reference/kernels).
