---
title: NVIDIA DeepStream
seo_title: Eseguire modelli YOLO su NVIDIA DeepStream
description: >-
  Esporta un modello LibreYOLO per NVIDIA DeepStream: un grafo ONNX più una
  configurazione nvinfer generata. Comandi esatti per la build del parser e per
  la pipeline.
lead: >-
  NVIDIA DeepStream esegue l'inferenza attraverso il suo elemento nvinfer, che
  ha bisogno di un grafo ONNX, di un file di configurazione corrispondente e di
  un parser dei bounding box. Impostare deepstream=True sull'esportazione ONNX
  scrive i primi due e li collega al terzo.
keywords:
  - NVIDIA DeepStream
  - DeepStream YOLO
  - nvinfer
  - parser bounding box DeepStream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - TensorRT engine
  - yolo su Jetson
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Scrive
    value: 'Un grafo ONNX, config_infer_primary_<stem>.txt e <stem>_labels.txt'
  - label: Copertura
    value: 43 combinazioni di famiglia e task su nove task
  - label: Parser
    value: >-
      NvDsInferParseYolo, dal progetto DeepStream-Yolo con licenza MIT di Marcos
      Luciano. Da compilare una volta per dispositivo.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Disponibilità
    value: >-
      Incluso nella v1.5.0. Confluito in dev il 2026-08-08 nella pull request
      728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Runtime validato
    value: 'DeepStream 8.0.0 su una RTX 5070 Ti, solo rilevamento, 2026-08-08'
verification: >-
  Scritto a partire dalla validazione a runtime del 2026-08-08. Elenchi di
  famiglie, chiavi di configurazione e valori predefiniti letti da
  libreyolo/export/deepstream.py e libreyolo/export/exporter.py al commit
  5f81e11e, confluito in dev lo stesso giorno nella pull request 728.
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Scrive libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # e libreyolo9s_labels.txt nella directory di lavoro.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Tieni ogni modello di rilevamento nella sua directory: ogni config di

        # rilevamento indica lo stesso file di cache dell'engine. Vedi "Trappole
        note".

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argomenti
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True viene rifiutato per ogni altro formato
            deepstream=True,
            conf=0.25,         # imposta pre-cluster-threshold (e classifier-threshold,
                               # segmentation-threshold su quei task)
            iou=0.45,          # imposta nms-iou-threshold, omesso con cluster-mode=4
            batch=1,           # imposta batch-size e il nome del file di cache dell'engine
            half=False,        # True segna nella config network-mode=2 (build fp16)
            int8=False,        # True segna nella config network-mode=1
            dynamic=True,      # asse batch dinamico nel grafo ONNX
            imgsz=640,         # imposta infer-dims=3;H;W
        )


        # deepstream=True e nms=True si escludono a vicenda: DeepStream esegue
        la

        # soppressione nella sua fase di clustering, quindi nel grafo non viene
        incorporato nulla.
    - label: Scarica prima i pesi di D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Verifica il passthrough della GPU prima di ogni altra cosa
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, da eseguire dentro il container DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 su questa immagine è uno stub e la build muore con

        # "fatal error: crt/host_defines.h: No such file or directory".
        Individua un

        # toolkit che contenga davvero l'header; sull'immagine 8.0 è cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # L'immagine include libcublas.so.12 e libcublas.so.12.8.4.1 ma non la

        # libcublas.so senza versione di cui -lcublas ha bisogno, quindi il link

        # fallisce con "/usr/bin/ld: cannot find -lcublas". Dai al linker i nomi
        che vuole.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: La segmentazione di istanze usa un parser diverso
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Eseguilo
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Entrambi i passaggi in un solo container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Disponibilità

L'esportazione DeepStream è inclusa nella v1.5.0. È confluita in `dev` il
2026-08-08 nella pull request 728, quindi un'installazione aggiornata la
contiene e non serve fissare alcun branch.

<code-tabs name="install" />

Se hai clonato il branch `deepstream-export` prima del 2026-08-08, sostituiscilo.
Quel branch è stato rebasato e force-pushato, e nella cronologia più vecchia
manca la correzione che permette a queste esportazioni di funzionare su una
macchina CUDA.

## Cosa scrive l'esportazione

`model.export(format="onnx", deepstream=True)` scrive tre file affiancati.
Per `libreyolo9s.pt`:

- `libreyolo9s.onnx`, il grafo di rilevamento, un solo tensore di output di forma
  `(batch, num_detections, 6)`, dove ogni riga è `[x1, y1, x2, y2, score, class_id]`
  in coordinate pixel dell'input della rete.
- `config_infer_primary_libreyolo9s.txt`, una configurazione `nvinfer` che porta
  con sé le costanti di preprocessing della famiglia, il numero di classi, le
  soglie e il collegamento al parser.
- `libreyolo9s_labels.txt`, un nome di classe per riga.

Il file delle etichette compare ogni volta che il checkpoint porta con sé i nomi
delle classi. I modelli di profondità non ne hanno, quindi non ricevono né il
file né la chiave `labelfile-path`.

LibreYOLO non produce nessun `.so`. Il `.so` che DeepStream carica è il parser
dei bounding box di `marcoslucianops/DeepStream-Yolo`, compilato una volta per
dispositivo, ed è lo stesso binario qualunque rilevatore LibreYOLO gli metti
davanti. Il modello è l'ONNX. La classificazione e la segmentazione semantica non
hanno bisogno di alcun parser, perché `nvinfer` le post-processa da sé.

## Esportare il modello

<code-tabs name="export" />

`LibreDFINE._load_weights` solleva `FileNotFoundError` quando il file non è già
su disco, senza tentare alcun download, quindi scarica prima `LibreDFINEs.pt` a
mano. La lacuna è tracciata nella
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). I pesi di YOLO9
si scaricano al primo utilizzo.

Il flag esiste solo in Python. `libreyolo export` su questo branch non ha
l'opzione `deepstream`, e la CLI costruisce i suoi argomenti di esportazione da
un elenco fisso invece di far passare chiavi sconosciute.

## Compilare il parser dei bounding box

Il rilevamento ha bisogno della libreria parser, la segmentazione di istanze ne
richiede una diversa, e i task restanti non ne richiedono nessuna. Due cose
sull'immagine DeepStream 8.0 rompono il comando di build documentato, ed entrambe
sono problemi dell'ambiente, non di LibreYOLO.

L'immagine include `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` e `cuda-12.9`
sotto `/usr/local`. Solo `cuda-12.5` ha un toolkit completo. Include anche
`libcublas.so.12` e `libcublas.so.12.8.4.1` ma non la `libcublas.so` senza
versione contro cui `-lcublas` si risolve. Lo script qui sotto aggira entrambi i
problemi.

<code-tabs name="parser" />

Poi fai puntare `custom-lib-path` nella config generata al
`libnvdsinfer_custom_impl_Yolo.so` appena compilato. Il valore generato è il
percorso relativo `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`,
che si risolve quando `deepstream-app` viene eseguito dalla checkout di
`DeepStream-Yolo` e altrimenti va modificato.

## Eseguire la pipeline

Verifica che il container veda la GPU prima di spendere tempo su qualsiasi altra
cosa. È il controllo che la sessione di validazione ha fatto per primo, su una
scheda Blackwell sotto WSL2.

<code-tabs name="gpu" />

La sessione di validazione ha guidato `deepstream-app` con una sola sorgente da
file, nessun sink di visualizzazione, l'on-screen display attivo e
`gie-kitti-output-dir` impostato in modo che i rilevamenti di ogni frame
finissero su disco come testo KITTI. Una config con queste impostazioni:

<code-tabs name="run" />

`nvinfer` costruisce l'engine TensorRT dall'ONNX alla prima esecuzione e lo mette
in cache accanto al modello, quindi la prima esecuzione paga la build dell'engine
e quelle successive caricano la cache.

## La config generata

Entrambe le config qui sotto sono state scritte dall'esportatore per la sessione
di validazione, senza modifiche successive.

| Chiave | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Le due config differiscono in tre punti: `maintain-aspect-ratio`, `cluster-mode`
e la presenza o meno di `nms-iou-threshold`. La config di D-FINE omette del tutto
quella chiave, che è quanto richiede `cluster-mode=4`.

Le teste che emettono al massimo una predizione per oggetto ricevono
`cluster-mode=4`, così DeepStream non esegue alcun clustering su di esse; il
clustering unirebbe rilevamenti realmente distinti. Questo riguarda `rfdetr`,
`dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` e `yolo9_e2e`.
Le teste a griglia e ad anchor ricevono `cluster-mode=2` più `nms-iou-threshold`.

Le config di rilevamento portano anche
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, che affida la costruzione
dell'engine alla libreria parser. È questo a fissare il nome del file di cache
dell'engine, ed è l'origine della collisione descritta nelle trappole note.

## Task e famiglie supportati

Sono esportabili quarantatré combinazioni di famiglia e task.
`deepstream_supported_tasks()` e `deepstream_supported_families(task)` in
`libreyolo/export/deepstream.py` restituiscono gli stessi elenchi a runtime.

| Task | `network-type` | Libreria parser | Famiglie |
|---|---|---|---|
| Rilevamento | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Classificazione | 1 | Nessuna | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentazione semantica | 2 | Nessuna | pidnet, eomt, dinov2, lingbotvision |
| Segmentazione di istanze | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Posa | 100 | Nessuna | yolo9, yolonas, rfdetr, ec |
| Profondità | 100 | Nessuna | depth_anything, zipdepth |
| Restauro | 100 | Nessuna | nafnet, realesrgan, swinir |
| Matting | 100 | Nessuna | birefnet |
| Sguardo | 100 | Nessuna | l2cs |

`network-type=100` significa che DeepStream non ha un post-processore per quel
task. Quelle config impostano `output-tensor-meta=1`, gli output nativi del grafo
passano inalterati e l'applicazione li decodifica dai metadati del tensore. I
grafi con più output non sono un problema: ogni layer di output arriva ai
metadati con gli stessi nomi di output e gli stessi assi dinamici di una normale
esportazione ONNX.

Nella segmentazione di istanze ogni riga è la riga di rilevamento seguita dalla
maschera di quell'istanza, appiattita a `(netH / 4, netW / 4)`, che è la
risoluzione fissata nel codice del parser seg, come probabilità per
`segmentation-threshold`.

La classificazione e la stima dello sguardo funzionano come inferenza secondaria.
Imposta `process-mode=2` e `operate-on-gie-id` nella config generata per mettere
un classificatore dietro a un rilevatore. Lo sguardo è un contratto di sola
testa, un ritaglio di volto per input, quindi ha bisogno di un rilevatore di
volti davanti.

Tre famiglie mancano di proposito. `segformer` non è collegato al contratto
condiviso di esportazione semantica e non può esportare in ONNX in nessun
formato. RTMDet-Ins e YOLO9 hanno l'esportazione della segmentazione di istanze
bloccata dentro LibreYOLO stesso. `depth_anything3` non ha
un'implementazione di esportazione.

Dietro due righe della tabella ci sono lacune nei checkpoint. È pubblicato solo
il checkpoint semantico `l` di EoMT, e la classificazione DINOv2 non ha alcun
checkpoint pubblicato, quindi quella combinazione richiede pesi affinati da te.

## Differenze di preprocessing

`nvinfer` calcola `net-scale-factor * (x - offsets)` per canale con una scala
scalare, che non può esprimere una deviazione standard per canale. Le famiglie
che ne hanno bisogno (`rfdetr`, `ec`, le taglie di `deimv2` con backbone DINO,
`rtmdet`, `picodet` e ogni famiglia di classificazione) hanno la normalizzazione
incorporata nel grafo esportato, e la config generata passa al grafo lo spazio di
input grezzo corrispondente.

La geometria è il punto in cui le pipeline Python di LibreYOLO e `nvinfer`
divergono ancora:

- Le famiglie letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) nativamente riempiono i bordi di grigio. `nvinfer` li riempie
  di nero.
- Il rilevamento `yolonas` nativamente ridimensiona il lato più lungo a 636 dentro
  il suo canvas da 640. Il `maintain-aspect-ratio` di `nvinfer` usa tutti i 640.
- La classificazione nativamente ridimensiona il lato più corto e poi ritaglia al
  centro. `nvinfer` deforma il frame o la ROI dell'oggetto fino all'input della
  rete, quindi i soggetti ritagliati stretti risultano diversi.
- EoMT nativamente esegue tasselli a finestra scorrevole per la segmentazione
  semantica. Il grafo esportato è un unico canvas deformato, più veloce e meno
  accurato.
- `pidnet` emette una mappa di classi a 1/8 della risoluzione di input e
  `lingbotvision` a 1/16. DeepStream fa l'upsampling della mappa di classi per la
  visualizzazione.

Il controllo di parità ONNX riceve tensori già preprocessati, quindi verifica gli
output del grafo e non può accorgersi di un ordine dei colori o di una politica
di padding sbagliati nella config. Valida sui tuoi dati prima di mettere in
produzione un carico che richiede parità esatta.

## Trappole note

### Due modelli di rilevamento nella stessa directory caricano l'engine l'uno dell'altro

Ogni config di rilevamento contiene la stessa riga:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Il costruttore di engine del parser richiede quel basename e non varia da modello
a modello. Esporta un secondo modello di rilevamento nella stessa directory e la
seconda esecuzione carica l'engine in cache del primo modello. Non va in crash
nulla; i box sono semplicemente sbagliati. Dai a ogni modello di rilevamento una
directory tutta sua. La sessione di validazione ha dovuto isolare D-FINE in una
directory dedicata prima di poterlo testare.

### Un box può portare una sola classe

Il formato di riga di `nvinfer` è `[x1, y1, x2, y2, score, class_id]`, una classe
per box, quindi l'esportazione riduce i punteggi di classe al loro argmax. Un box
che `predict` riporta sotto due classi sopravvive sotto una sola. Caso misurato:
LibreYOLO riporta `vase 0.773` e `bottle 0.383` sullo stesso box, e il grafo
DeepStream mantiene `vase`. Discende dal formato di riga del parser e non si può
cambiare senza uscire da quel contratto, quindi è un comportamento atteso e non
una regressione.

## Validato

`deepstream-app` è arrivato a EOS con `App run successful` su entrambi i tipi di
testa del rilevatore, sul `sample_1080p_h264.mp4` fornito da NVIDIA (1443 frame),
con i dump KITTI per frame attivi.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Tipo di testa | a griglia | uno-a-uno |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frame con rilevamenti | 1443 | 1443 |
| Rilevamenti totali | 18031 | 71105 |

Gli istogrammi delle classi su tutti i 1443 frame mettono le auto al primo posto
e le persone al secondo per entrambi i modelli, il che è corretto per una scena
di strada. Il divario di quattro volte nel numero di rilevamenti è la differenza
di `cluster-mode` che fa il suo lavoro: D-FINE con `cluster-mode=4` non esegue
alcun clustering, quindi ogni query sopra soglia sopravvive, quasi-duplicati
inclusi.

Due modelli addestrati in modo indipendente collocano l'oggetto dominante nello
stesso punto:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Quella esecuzione stabilisce cinque cose: TensorRT costruisce un engine dall'ONNX
esportato su sm_120, `nvinfer` accetta ogni chiave della config generata,
`NvDsInferParseYolo` legge correttamente il layout del tensore, i box finiscono
in coordinate 1920x1080 alla risoluzione della sorgente, e le etichette si
risolvono contro il file di etichette generato.

L'ambiente in cui è stata eseguita:

| Componente | Valore |
|---|---|
| Sistema operativo host | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Runtime dei container | Docker Desktop 29.4.3, backend WSL2 |
| Immagine DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Versione di DeepStream | 8.0.0 |
| CUDA nel container | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` a HEAD |

Oltre all'esecuzione della pipeline, `tests/unit/test_deepstream_export.py` copre
gli adattatori del grafo e le chiavi della config generata, e i suoi 35 test
passano su questo commit.

## Non validato

Elencato perché l'ambito qui sopra non venga letto più ampio di quanto sia.

- Jetson e aarch64. Il contratto di esportazione non dipende dall'architettura,
  ma la pipeline è stata eseguita solo su una GPU discreta x86.
- Quarantuno delle 43 combinazioni. Solo il rilevamento con `yolo9` e il
  rilevamento con `dfine` sono passati per DeepStream. La classificazione, la
  segmentazione semantica, la segmentazione di istanze e i task a tensore grezzo
  sono coperti da test unitari e controlli di parità ONNX, non da un'esecuzione
  della pipeline.
- FP16 e INT8. È stato provato solo `network-mode=0`.
- Multi-stream e batching. Una sola sorgente, `batch-size=1`.
- L'accuratezza rispetto a un dataset di ground truth. I rilevamenti sono stati
  controllati per plausibilità semantica e accordo tra modelli, non misurati come
  mAP attraverso DeepStream.
