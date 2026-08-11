---
title: Prestazioni dell'addestramento
seo_title: 'Addestramento più veloce: CUDA graph, AMP, profiler'
description: >-
  Rendi più veloce un addestramento: cattura lo step nei CUDA graph, scegli un
  dtype per l'AMP e usa il profiler integrato per scoprire dove va davvero il
  tempo.
lead: >-
  Tre leve cambiano la velocità di uno step di addestramento: la precisione
  mista, la cattura del forward e del backward della rete nei CUDA graph, e
  qualunque cosa il profiler dica stia davvero frenando lo step.
keywords:
  - cuda graphs training
  - velocizzare addestramento yolo
  - addestramento precisione mista
  - training bfloat16
  - pytorch profiler
  - training limitato dal dataloader
  - kernel launch overhead
  - utilizzo gpu addestramento
last_verified: 1.5.0
snippets:
  profile:
    - label: Profilare e continuare l'addestramento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Profila una breve finestra di step reali, stampa un verdetto e poi
        # continua l'esecuzione con gli hook rimossi.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 'Solo misurare, poi fermarsi'
      language: bash
      code: >
        # Imposta no_aug_epochs=0 ed esegue le epoche necessarie a riempire la
        finestra.

        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Approfondire il risultato
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Misura prima di cambiare qualsiasi cosa

Le tre leve qui sotto risolvono problemi diversi, e applicare quella sbagliata
non cambia nulla. Il profiler dice quale problema hai.

<code-tabs name="profile" />

`profile=True` misura una finestra di step reali di addestramento, per
impostazione predefinita cinque scartati e poi venti misurati, stampa un report,
scrive i suoi artefatti e poi prosegue l'addestramento con gli hook rimossi. Non
costa nulla quando è disattivato, e viene ignorato nell'addestramento distribuito.

Il report si chiude con uno di quattro verdetti:

| Verdetto | Significato | Leve |
|---|---|---|
| `dataloader` | la GPU aspetta i dati in ingresso | più `workers`, `cache="ram"` o `"disk"`, augmentation più leggera, batch più grande |
| `host / launch` | la GPU viene alimentata troppo lentamente, molti kernel minuscoli | batch più grande, CUDA graph, meno sincronizzazioni con l'host per step |
| `compute` | la GPU è satura | AMP o bfloat16, oppure accettarlo |
| `memory-pressure` | thrashing dell'allocatore, VRAM al limite | abbassa il batch; qui le cifre di utilizzo non sono affidabili |

Il numero di utilizzo è il tempo di occupazione dei kernel diviso il tempo di
step non sincronizzato. La finestra è divisa di proposito: la prima metà gira
senza sincronizzazioni aggiuntive, così il verdetto riflette la sovrapposizione
reale, e solo la seconda metà racchiude ogni fase tra due sincronizzazioni per
attribuire il tempo GPU. Sincronizzare ogni fase concede respiro ai worker del
dataloader e nasconde la starvation, quindi i numeri di composizione non vengono
mai usati per scegliere il verdetto.

Nella directory dell'esecuzione finiscono quattro file: `timeline.html`, che si
apre da solo in un browser, `profile_trace.json` per Perfetto o Nsight,
`profile_summary.json` e `profile.json`, quello autonomo da copiare in giro e da
dare in pasto ai sottocomandi `libreyolo profile`.

Su `profile run` vale la pena sapere due cose. Imposta `no_aug_epochs=0`, perché
il profiler misura l'epoca 0 e un'esecuzione breve con il valore predefinito di
`no_aug_epochs` profilerebbe il dataloader più leggero senza augmentation invece
di quello che l'addestramento usa davvero. E `--repeat N` riporta media e
deviazione standard, il che conta perché uno step limitato dai lanci è abbastanza
rumoroso da rendere fuorviante una singola esecuzione; scrive una directory per
prova, `prof_1`, `prof_2` e così via, più un `profile_repeat.json` aggregato.

## Precisione mista

`amp=True` è il valore predefinito per la maggior parte delle famiglie ed esegue
il forward sotto l'autocast CUDA. `amp_dtype` sceglie tra `float16` e `bfloat16`.

<code-tabs name="amp" />

Float16 ha bisogno del loss scaling dinamico e riceve uno scaler dei gradienti
attivo; l'intervallo di esponenti più ampio di bfloat16 non ne ha bisogno, quindi
il suo scaler è disattivato. Quattro famiglie arrivano con `amp=False`, D-FINE,
DEIM, YOLO-NAS e FOMO, e l'impostazione di DEIM si propaga a RT-DETRv4 per
ereditarietà. D-FINE ne dichiara il motivo: il suo decoder limita le attivazioni
a 65504, il più grande valore finito di float16.

La semantica degli argomenti, incluso cosa fa una richiesta di bfloat16 su
hardware che non supporta bfloat16, è su
[Iperparametri](/docs/train/hyperparameters).

## CUDA graph

`cuda_graph=True` cattura il forward e il backward di addestramento della rete in
un CUDA graph, eliminando l'overhead di lancio dei kernel a ogni step.

<code-tabs name="graph" />

Passare il flag è sempre sicuro. Una famiglia, un task o una configurazione che
non si possono catturare scrivono una riga di log e addestrano in eager, senza
cambiare nulla.

Viene catturata solo la rete. La loss resta eager per scelta, perché le loss di
rilevamento selezionano con maschere booleane, eseguono il matching ungherese e
si ramificano sui risultati dell'assegnazione, e un graph non può registrare
nulla di tutto ciò. Anche lo step dell'ottimizzatore, il gradient clipping,
l'aggiornamento dell'EMA e lo schedule del learning rate restano eager.

Questo limita il guadagno a quanta parte di uno step è rete, e la quota varia
parecchio. Misurato su una RTX 5070 Ti a 640 px, batch 8: l'84 percento di uno
step di YOLOv9-t è rete, il 44 percento di uno step di YOLOv7-b, il 31 percento
di uno step di YOLOX-t e il 26 percento di uno step di RTMDet-t. Gli ultimi due
passano la maggior parte dello step dentro i loro assegnatori di etichette,
quindi catturare la rete è ciò che li aiuta meno.

### Quanto vale

Condizioni valide per ogni cifra qui sotto: RTX 5070 Ti, Windows, AMP, un
processo per ciascun ramo a partire da uno stato salvato condiviso, riproducendo
un batch reale così da togliere di mezzo il dataloader, il più veloce di 24 step
dopo il warm-up. Rilevamento a 640 px, classificazione a 224 px. La dimensione
del batch è per riga.

| Famiglia | Dimensione | Batch | Eager | Con graph | Speedup |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Queste cifre isolano lo step su GPU. Un fine-tuning completo paga anche il
dataloader e la validazione. YOLOv9-t su un dataset di rilevamento da 406
immagini, 20 epoche, batch 8, 640 px, 4 worker del dataloader, sulla stessa
macchina: 428.4 s di tempo reale in eager contro 367.7 s con il graph, un
guadagno di 1.16x, con mAP50-95 di 0.6394 in entrambi i rami.

Tre cose spostano questi numeri. I batch piccoli sono limitati dai lanci e quelli
grandi dal calcolo, quindi RT-DETR-r18 guadagna 1.19x con batch 2 e 1.04x con
batch 8. L'overhead di lancio è massimo su Windows, e su Linux i guadagni sono
circa da un terzo a metà di quelli in tabella. E un'esecuzione limitata dal
dataloader non vede alcun cambiamento nel tempo reale, ed è per questo che il
profiler viene prima.

La cattura si attiva allo stesso modo con `amp=False`, ma i kernel fp32 durano di
più, quindi uno step è meno limitato dai lanci e la maggior parte delle famiglie
guadagna meno. Sullo stesso hardware, MobileNetV4-s con batch 16 passa da 2.74x
sotto AMP a 3.61x in fp32, mentre YOLOv9-t con batch 8 passa da 1.99x a 1.69x e
RT-DETR-r18 con batch 4 da 1.12x a 0.99x.

### Dove si applica la cattura

| Task | Famiglie |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Tutto il resto ricade su eager con una riga di log: gli altri task su quelle
famiglie, le famiglie non elencate, le esecuzioni distribuite e le esecuzioni con
distillazione. Anche una cattura che fallisce a runtime fa scendere in eager il
resto dell'esecuzione invece di generare un errore.

Per i detector encoder-decoder, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4, ed EC,
vengono catturati solo il backbone e l'encoder. Il loro decoder legge il ground
truth per costruire le query di contrastive denoising, e il numero di quelle
query segue il conteggio di ground truth più alto del batch, quindi il suo numero
di token cambia da un batch all'altro.

### Shape

Un graph è valido esattamente per la shape di input con cui è stato catturato. Il
trainer conta le shape dei batch e cattura quando una shape si è ripetuta tre
volte. I batch con qualsiasi altra shape girano in eager: i batch multi-scala e
l'ultimo batch parziale di un'epoca.

Questa è la trappola per le famiglie DETR, che per impostazione predefinita
ridimensionano ogni batch. Con `multi_scale=True` un'esecuzione breve può non
vedere mai una shape abbastanza spesso da arrivare a catturare. Passa
`multi_scale=False` quando quello che ti interessa è lo speedup.

YOLOX cambia ciò che la regione catturata calcola a metà esecuzione, attivando il
suo ramo di regressione L1 quando il mosaic si chiude a `no_aug_epochs`. Lì il
trainer invalida la cattura e ricattura una volta che la nuova shape si è
stabilizzata.

### Numerica e memoria

La maggior parte delle famiglie riproduce bit per bit la traiettoria della loss
in eager sotto AMP. FOMO e LingBot-Vision differiscono nell'ultimo bit del float32
per via di un diverso ordine di somma. Neanche i detector con deformable
attention, D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR ed EC, riproducono le proprie
esecuzioni in eager, perché quel backward accumula con operazioni atomiche e le
convoluzioni TF32 scelgono un ordine di riduzione a ogni lancio; l'esecuzione con
il graph resta dentro quella dispersione. RTMDet differisce di circa 3e-4 in
valore relativo su due dei 139 gradienti, perché condivide le convoluzioni della
testa tra i livelli della piramide e i due percorsi di backward sommano tre
contributi in ordine diverso. SegFormer ha la stochastic depth dentro la regione
catturata, quindi un graph riprodotto attinge al proprio flusso casuale ed è
statisticamente equivalente all'eager anziché identico; il manager lo registra una
volta al momento della cattura.

Con `amp=False` il risultato bit-identico non è ottenibile da nulla su questo
hardware, con o senza cattura. Due esecuzioni eager di YOLOv9-t identiche e con lo
stesso seed divergono del 36 percento in valore relativo su 20 step, e YOLOX-t del
2.6 percento, perché cuDNN sceglie un algoritmo non deterministico per il
gradiente dei pesi con alcune shape di convoluzione fp32.

Un graph catturato fissa buffer statici di input, output e workspace, quindi il
picco di VRAM sale di circa un set di attivazioni in più. Sulle famiglie qui
sopra, il picco di allocazione si è spostato tra il -5 e il +19 percento. Il costo
relativo è massimo per i piccoli modelli di classificazione, le cui attivazioni
sono già piccole di partenza: ResNet-18 a 224 px, batch 16, è passato da 0.48 GB
in eager a 0.57 GB con il graph. Se questo porta un'esecuzione oltre il limite,
abbassa il batch o lascia il flag disattivato.

## Correlati

- [Iperparametri](/docs/train/hyperparameters) per `batch`, `nbs`, `cache` e
  `workers`.
- [Addestramento multi-GPU](/docs/train/multi-gpu), dove non sono disponibili né i
  CUDA graph né il profiler.
- [CUDA graph](/docs/reference/cuda-graphs) per la matrice di supporto combinata
  di inferenza e addestramento, i punti di taglio e il contratto sulla numerica.
