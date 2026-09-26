---
title: NVIDIA Jetson
seo_title: Installare LibreYOLO e PyTorch su NVIDIA Jetson
description: >-
  Installa LibreYOLO su un NVIDIA Jetson: le quattro librerie CUDA che JetPack
  non include, il passaggio --no-deps che serve a PyTorch e i numeri misurati
  sull'Orin Nano.
lead: >-
  Le schede NVIDIA Jetson eseguono LibreYOLO sui normali wheel aarch64 di
  PyTorch. Non è coinvolta nessuna build di torch specifica per Jetson, ma
  JetPack omette quattro librerie a cui torch si collega, e l'installazione deve
  fornirle.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - installare pytorch su jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available false jetson
  - no kernel image is available for execution on the device
  - tensorrt su jetson
  - wheel aarch64 pytorch
last_verified: 1.4.0
meta:
  - label: Scheda
    value: >-
      Jetson Orin Nano Super Developer Kit, 8 GB, compute capability della GPU
      8.7
  - label: Piattaforma
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Stack testato
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, il 2026-07-27
  - label: Assenti in JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Benchmark
    value: >-
      223 esecuzioni verificate su questa scheda, 58 modelli in 12 famiglie, in
      PyTorch, ONNX Runtime e TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Tracciato in
    value: La metà Jetson della issue 648
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Ricetta di installazione e output atteso presi dall'installazione del
  2026-07-27 su un Jetson Orin Nano Super. Le righe di latenza e accuratezza
  vengono dallo snapshot dei risultati verificati dietro visionanalysis.org,
  filtrato sull'hardware jetson_orin, misurato a giugno 2026 su libreyolo
  1.2.0.dev0. Comportamento dell'esportazione e del loader letti da
  libreyolo/export/exporter.py, libreyolo/export/tensorrt.py e
  libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Pacchetti di sistema e un ambiente virtuale
      language: bash
      code: |
        # JetPack non preinstalla pip né il modulo venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: 'PyTorch, dall''indice dei wheel per CUDA 13'
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: Le quattro librerie che JetPack non include
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Se pip pretende cuda-toolkit 13.0.3, installa con --no-deps'
      language: bash
      code: >
        # --no-deps significa che anche le dipendenze Python di torch vanno
        elencate a mano.

        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: 'Fatti dire qual è la prossima libreria mancante, invece di indovinare'
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Tutto quello che manca ancora in tutte le librerie di torch, in un
        colpo solo:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Installa LibreYOLO dopo torch, non prima'
      language: bash
      code: >
        # torch è già soddisfatto, quindi pip lascia al suo posto la build CUDA.

        pip install libreyolo


        # L'extra ONNX serve solo per esportare. Un'esportazione TensorRT passa

        # per ONNX, quindi installalo prima della sezione sull'esportazione qui
        sotto.

        pip install "libreyolo[onnx]"
  verify:
    - label: Versioni e dispositivo
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Poi esegui un kernel vero
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Scarica il checkpoint al primo utilizzo.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Scrive libreyolo9s.onnx, poi ne costruisce libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # L'engine si ricarica dallo stesso punto di ingresso.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Modalità di alimentazione e clock
      language: bash
      code: >
        sudo nvpmodel -q      # quali modalità espone questa scheda, e quella
        attiva

        sudo nvpmodel -m 0    # modalità più alta sulla scheda testata qui

        sudo jetson_clocks


        tegrastats            # carico in tempo reale; nvidia-smi è limitato su
        Tegra
source_hash: c07ff908503e89b5
---

## Cosa documenta questa pagina

Questa pagina documenta una configurazione verificata da capo a fondo, non una
matrice di compatibilità. La scheda era un Jetson Orin Nano Super Developer Kit
con 8 GB di memoria e JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python
3.12.3), e lo stack che ci è girato sopra era `libreyolo 1.4.0` con `torch
2.13.0+cu130`, OpenCV 5.0.0 e NumPy 2.5.1. `torch.cuda.is_available()` ha
restituito `True` e la GPU si è dichiarata `Orin`.

Altre release di JetPack, altre schede Jetson e altre versioni di CUDA non sono
state testate. La ricetta qui sotto è quella che ha funzionato su quella
combinazione.

Quell'esecuzione è del 2026-07-27 su LibreYOLO 1.4.0 e non è stata ripetuta su
hardware 1.5.0: questa è l'unica pagina dell'albero 1.5.0 che porta ancora una
verifica 1.4.0, ed è il motivo per cui il suo front matter dice
`last_verified: "1.4.0"`. Nulla nelle modifiche di 1.5.0 tocca il percorso di
installazione, le
quattro librerie mancanti o i flag di esportazione descritti qui, quindi ci si
aspetta che i comandi reggano, ma i numeri di versione negli output qui sotto
sono quelli stampati da 1.4.0, non una misura su 1.5.0.

Due cose vanno contro quello che dice la maggior parte delle guide per Jetson. I
wheel sono le normali build aarch64 pubblicate per CUDA 13, quindi non serve
nessuna build di torch specifica per Jetson. E JetPack non include quattro
librerie a cui quei wheel si collegano, quindi `import torch` fallisce una
libreria alla volta finché non sono installate tutte e quattro.

## Installazione

Le immagini di JetPack arrivano senza pip e senza il modulo `venv`, quindi si
comincia da lì.

<code-tabs name="prep" />

Una scheda da 8 GB è al limite con i checkpoint più grandi. Aggiungere swap
sull'NVMe prima di caricarli evita che il processo venga ucciso per esaurimento
della memoria a metà esecuzione.

Poi PyTorch. L'indice CUDA 13 contiene i wheel aarch64; l'indice extra fornisce
le dipendenze pure-Python da PyPI.

<code-tabs name="torch" />

I quattro wheel `nvidia-*-cu13` sono la parte che sfugge facilmente. JetPack
fornisce il driver della GPU, non cuDNN, NCCL, cuSPARSELt o NVSHMEM, e torch si
rifiuta di essere importato senza di loro. Installarli tutti e quattro in una
volta è più rapido che scoprirli un'eccezione alla volta.

Il terzo snippet copre un errore specifico: i metadati delle dipendenze di torch
per la build CUDA 13 chiedono `cuda-toolkit==13.0.3`, che su PyPI non ha nessun
wheel aarch64, quindi la risoluzione fallisce prima di scaricare qualsiasi cosa.
`--no-deps` salta il resolver, il che significa che ogni dipendenza va indicata
sulla riga di comando.

LibreYOLO va per ultimo. Installarlo per primo lascia scegliere a pip il proprio
torch, che su questa piattaforma non è la build CUDA.

<code-tabs name="install" />

Ogni dipendenza rimanente si risolve in un wheel aarch64 precompilato, compresi
OpenCV, NumPy, SciPy, pycocotools e safetensors. Non si compila niente dai
sorgenti.

## Verifica che CUDA funzioni

<code-tabs name="verify" />

Il secondo snippet conta quanto il primo. Un wheel compilato per l'architettura
GPU sbagliata riporta comunque `torch.cuda.is_available() == True` e poi
fallisce alla prima operazione vera con `CUDA error: no kernel image is
available for execution on the device`. Una moltiplicazione di matrici sul
dispositivo è il controllo che lo smaschera.

## Esegui una predizione

<code-tabs name="predict" />

`predict` restituisce lo stesso oggetto `Results` di qualsiasi altra
piattaforma, quindi le pagine dei modelli valgono senza modifiche.

## Esportare in TensorRT

Su questa scheda TensorRT è stato più veloce sia di PyTorch sia di ONNX Runtime
per tutti i 55 modelli misurati in ogni runtime.

<code-tabs name="export" />

`format="tensorrt"` scrive prima un grafo ONNX e da quello costruisce l'engine,
quindi l'extra `onnx` deve essere installato. `LibreYOLO()` decide in base al
suffisso del file, quindi un file `.engine` si carica con la stessa chiamata di
un checkpoint `.pt`.

Non usare l'extra pip `tensorrt` su un Jetson. Fissa `tensorrt-cu12`, una build
CUDA 12, su una piattaforma CUDA 13. Usa invece il TensorRT che installa
JetPack. Se `import tensorrt` fallisce dentro l'ambiente virtuale mentre
funziona fuori, ricrea l'ambiente con `--system-site-packages` in modo che il
modulo di sistema sia visibile.

Gli engine TensorRT serializzati sono legati al dispositivo, all'architettura
della GPU e alla versione di TensorRT che li ha costruiti. Un engine costruito
su una workstation non si carica su un Jetson, quindi il passo di build va
eseguito sulla scheda.

## Misurato su questa scheda

Latenza per immagine, dimensione del batch 1, da capo a fondo compresi
preprocessing e postprocessing, su COCO val2017 (sottoinsieme di 500 immagini)
con `conf=0.001` e `max_det=300`. Cinque modelli sui 58 misurati:

| Modello | Input (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

La colonna mAP è il punteggio ottenuto dall'esecuzione TensorRT FP16 stessa. Sui
55 modelli misurati in tutti e quattro i runtime, il divario massimo tra il
punteggio PyTorch FP32 e quello TensorRT FP16 è stato di 0.59 punti, su
DEIMv2-X. I runtime differiscono nella velocità, non nell'accuratezza.

TensorRT FP32 è stato più veloce sia di PyTorch sia di ONNX Runtime per tutti e
55 quei modelli. Anche TensorRT FP16 è stato più veloce di PyTorch FP32 per
tutti e 55, da 1.68x a 6.22x, con una mediana di 3.39x. Quello che varia è ONNX
Runtime: è stato più lento di PyTorch su 23 dei 55, tra cui la riga RT-DETR-r18.

Le condizioni dietro ogni numero: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, misurati a giugno
2026. Su un Jetson la latenza dipende anche dalla modalità di alimentazione
attiva, che i record dei benchmark non riportano.

<code-tabs name="power" />

Tutte le 223 esecuzioni, compresi gli altri 53 modelli e le colonne complete
dell'accuratezza, sono pubblicate su
[la pagina Jetson Orin di Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Risoluzione dei problemi

### import torch fallisce indicando una libreria condivisa

Manca una delle quattro librerie qui sopra. Invece di indovinare quale,
leggila direttamente dal binario:

<code-tabs name="ldd" />

Ogni voce mancante corrisponde a un wheel:

| Libreria mancante | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch avvisa che nessuna build supporta questa GPU

La prima chiamata CUDA sulla configurazione funzionante stampa questo:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

Su questa scheda l'avviso è puramente estetico. Il wheel contiene kernel `sm_80`
e l'Orin li esegue. Lo stesso avviso compariva sul wheel precedente di quello
stesso indice, quello che ha prodotto tutte le righe di benchmark qui sopra.
Conferma con la moltiplicazione di matrici del controllo CUDA, invece di fidarti
o diffidare del messaggio.

### CUDA error: no kernel image is available for execution on the device

Il wheel installato è stato compilato per un'architettura GPU diversa. È quello
che succede con i wheel dell'indice `sbsa` di NVIDIA, che puntano alle GPU ARM
da server e non al silicio Jetson. Reinstalla dall'indice CUDA 13 della sezione
sull'installazione.

### pip non trova cuda-toolkit 13.0.3

Non esiste un wheel aarch64 per quel pacchetto. Usa la forma `--no-deps` della
sezione sull'installazione ed elenca esplicitamente le dipendenze di torch.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Il wheel aarch64 di torch si collega alle NVIDIA Performance Libraries per i
calcoli su CPU. Installale e mettile nel percorso delle librerie:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Quell'indice va bene per queste due librerie CPU. Sono le sue build di torch a
produrre l'errore "no kernel image" qui sopra.

### Sorgenti di wheel che non vanno bene per JetPack 7.2

| Sorgente | Risultato sull'Orin Nano Super |
|---|---|
| torch da `pypi.jetson-ai-lab.io/sbsa/cu130` | Compilato per GPU ARM da server. Si importa, dichiara CUDA disponibile, poi fallisce con "no kernel image is available for execution on the device". |
| torch da `pypi.jetson-ai-lab.io/jp6/*` | Build per CUDA 12 e Python 3.10. Non si installano sul Python 3.12 di questa immagine. |
| Container PyTorch per JetPack 6 | L'inizializzazione di CUDA fallisce con errore 801 su un host JetPack 7. |
| Compilare torch dai sorgenti | Funziona, ma richiede ore su una scheda da 8 GB ed è superfluo una volta installati i wheel CUDA 13. |

## DeepStream

Per una pipeline video completa invece di un loop Python, esporta con
`deepstream=True` ed esegui il grafo attraverso `nvinfer`. Quel percorso ha una
pagina tutta sua, con la configurazione `nvinfer` generata, la build del parser
dei bounding box e le trappole note: [DeepStream](/docs/export/deepstream).

La pipeline DeepStream in sé è stata validata su una GPU discreta x86, non su un
Jetson. Il contratto di esportazione non dipende dall'architettura, ma
l'esecuzione della pipeline su aarch64 resta in sospeso.

## Non verificato

- Release di JetPack diverse dalla 7.2, e release di L4T diverse dalla R39.2.
- Schede Jetson diverse dall'Orin Nano Super 8 GB.
- L'addestramento sulla scheda. Sono stati esercitati inferenza ed
  esportazione; un addestramento no.
- Gli engine INT8. Per questa scheda esistono solo righe FP32 e FP16.
- Dimensioni del batch superiori a 1. Ogni misura qui sopra è a batch 1.
