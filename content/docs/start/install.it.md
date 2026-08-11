---
title: Installazione
seo_title: Installare LibreYOLO
description: >-
  Installa LibreYOLO da PyPI, scegli gli extra opzionali richiesti da una
  famiglia di modelli o da un target di esportazione, e verifica che PyTorch
  veda la tua GPU.
lead: >-
  LibreYOLO è pubblicato su PyPI come libreyolo. Il pacchetto base copre
  predizione, addestramento, validazione e le famiglie di modelli che non
  richiedono nulla oltre a PyTorch; gli extra opzionali aggiungono il resto.
keywords:
  - installare libreyolo
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo gpu
  - requisiti libreyolo
last_verified: 1.5.0
meta:
  - label: Pacchetto
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 o superiore
  - label: Licenza del codice
    value: MIT
  - label: Dipendenza principale
    value: PyTorch 2.4 o superiore
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Con extra
      language: bash
      code: >
        # Separali con la virgola per combinarne più di uno in una sola
        installazione.

        pip install "libreyolo[rfdetr,onnx]"
    - label: Tutto
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Dal codice sorgente
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, ogni GPU visibile e quali
        # pacchetti opzionali sono installati.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inventario dei modelli
      language: bash
      code: |
        # Ogni famiglia registrata con i suoi task, le dimensioni e le
        # risoluzioni di input. Le famiglie a cui manca l'extra sono
        # elencate con il comando pip che le abilita.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Installazione

<code-tabs name="install" />

È richiesto Python 3.10 o superiore. L'installazione base tira dentro PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors e SciPy, così YOLOv9 e le altre famiglie che non
richiedono altro funzionano subito dopo `pip install libreyolo`.

Un clone lascia attivo `release`, il branch stabile il cui codice corrisponde a
questa documentazione. Il branch di integrazione, che porta il lavoro non ancora
rilasciato, è `dev`.

## Extra opzionali

Un extra è un nome tra parentesi quadre che aggiunge le dipendenze richieste da
una famiglia di modelli o da un target di esportazione. Non cambia nient'altro:
l'API è la stessa con o senza l'extra.

### Famiglie di modelli

| Extra | Aggiunge |
|---|---|
| `rfdetr` | `transformers`, che fornisce il backbone di RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, che fornisce gli encoder ViT-L/16 e EfficientNet-Lite3 di MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` e `bitsandbytes` fuori da macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` e `ftfy`, necessari al tokenizer di testo CLIP incluso nella libreria |
| `siglip2` | `sentencepiece`, necessario al tokenizer multilingue di SigLIP 2 |
| `gaze` | `gdown`, che attiva il download automatico del checkpoint L2CS |
| `rtdetr` | Niente. RT-DETR non ha bisogno di dipendenze extra; il nome viene mantenuto per stabilità |

### Esportazione e runtime

| Extra | Aggiunge |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 e `pycuda`, fuori da macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, solo macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` più `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` e `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` più `MNN` |
| `ncnn` | `pnnx` e `ncnn` |
| `paddle` | `libreyolo[onnx]` più `paddlepaddle` 2.6.2 e `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` per l'inferenza V2 su HTTP e HTTPS |

### Addestramento, valutazione e logging

| Extra | Aggiunge |
|---|---|
| `lora` | `libreyolo[rfdetr]` più `peft`, per il fine-tuning con `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, il backend C++ per la valutazione COCO |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` è opzionale invece che una dipendenza obbligatoria, così una
piattaforma senza wheel precompilata non può rompere un'installazione semplice.
Quando il pacchetto manca, la valutazione COCO ricade su pycocotools e
l'esecuzione continua.

### Strumenti

| Extra | Aggiunge |
|---|---|
| `stream` | `yt-dlp`, necessario solo per risolvere gli URL delle pagine di YouTube |
| `tracking` | Niente. Ogni dipendenza del tracking è già una dipendenza core |
| `label` | `libreyolo[sam]`, che abilita l'assistenza click-to-mask in `libreyolo label` |
| `hub-kernels` | `kernels`, il loader opzionale per i kernel compilati dell'Hub. Vedi [kernels](/docs/reference/kernels), dove si nota che installarlo può spostare le predizioni di RF-DETR entro la tolleranza float |
| `clip-convert` | `libreyolo[clip]` più `open_clip_torch`, per la conversione dei pesi e i controlli di parità |
| `siglip2-convert` | `libreyolo[siglip2]` più `transformers`, per lo stesso motivo |

Le webcam, RTSP, RTMP, TCP, UDP, HLS e le liste multi-stream locali non
richiedono alcun extra. Solo gli URL delle pagine di YouTube lo richiedono.

### L'extra aggregato

`libreyolo[all]` installa in un solo comando gli extra dei modelli,
dell'esportazione, del tracking e del logging. Alcuni ne restano deliberatamente
fuori. `neptune` è escluso perché la versione stabile di `neptune-scale`
richiede protobuf sotto la 7 mentre il percorso TFLite richiede protobuf 7.
`executorch` è escluso perché ExecuTorch vincola la versione di PyTorch con cui
si accoppia, e `coreai` perché `coreai-torch` fissa PyTorch a 2.11.x e
trascinerebbe l'intero ambiente su quella versione. Anche `fast-eval`,
`hub-kernels`, `clip-convert` e `siglip2-convert` restano fuori. Installa uno
qualsiasi di questi indicandolo per nome.

## Vincoli di piattaforma

Tre extra sono limitati per piattaforma dai loro marker di dipendenza, così
l'installazione riesce ovunque e installa semplicemente meno dove una wheel non
esiste.

| Extra | Vincolo |
|---|---|
| `coreai` | Solo macOS. La toolchain di Core AI non converte né esegue altrove |
| `tensorrt` | Saltato su macOS, che non ha CUDA |
| `tflite`, `litert` | `onnx2tf` e `ai-edge-litert` richiedono Python 3.12 o superiore |

`sensenova` salta `bitsandbytes` su macOS, dove non viene pubblicata alcuna
wheel; il resto dell'extra si installa normalmente.

Se il vincolo è il disco, la maggior parte è PyTorch, e la maggior parte di
PyTorch è il payload CUDA che la sua wheel predefinita include. Una wheel solo
CPU lo elimina senza rinunciare a nulla. Per il rilevamento con ONNX su una
macchina che non deve portarsi dietro torch,
vedi l'[installazione leggera](/docs/lightweight-install).

## GPU e CUDA

La selezione del dispositivo avviene quando il modello viene costruito. Il
valore predefinito, `device="auto"`, usa CUDA quando `torch.cuda.is_available()`
è vero, poi Metal Performance Shaders quando `torch.backends.mps.is_available()`
è vero, e altrimenti la CPU. Nient'altro nella libreria ispeziona l'hardware,
quindi se PyTorch non vede una GPU, non la vede nemmeno LibreYOLO.

Per fissare invece il dispositivo, passa `device` al modello oppure a `predict`,
`train`, `val` ed `export`. Accetta `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, un
intero semplice come `0`, o una stringa di cifre come `"0"`; gli ultimi due
vengono espansi in `cuda:<n>`.

Parti da `libreyolo checks`, che stampa la versione di Torch, le versioni di
CUDA e cuDNN con cui Torch è stato compilato, e ogni GPU visibile con la sua
memoria. Quando segnala che non c'è CUDA su una macchina che ha una scheda
NVIDIA, la wheel di PyTorch risolta da pip è una build per CPU. Installa prima
una build con CUDA dall'indice di PyTorch, poi installa LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

È lo stesso indice che il repository fissa per il proprio ambiente gestito con
uv su Linux e Windows. Richiede il driver NVIDIA 555 o superiore, che è il
requisito del runtime CUDA 12.8. macOS resta sulla wheel di PyPI, dato che
l'host di download di PyTorch non pubblica build per Darwin.

## Verificare l'installazione

<code-tabs name="verify" />

`libreyolo models` è il modo più rapido per vedere se un extra ha avuto effetto:
una famiglia a cui manca la dipendenza viene stampata con il comando pip esatto
che la abilita. Entrambi i comandi accettano anche `--json`, che stampa gli
stessi dati come oggetto leggibile da una macchina su stdout.
