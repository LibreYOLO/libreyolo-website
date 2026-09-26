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
last_verified: 1.6.0
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
      code: |
        # Usa la virgola per combinarne più di uno in una sola installazione.
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
source_hash: 531023c2092fd751
---

## Installazione

<code-tabs name="install" />

È richiesto Python 3.10 o superiore. L'installazione base porta con sé PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors e SciPy, così YOLOv9 e le altre famiglie che non
richiedono altro funzionano subito dopo `pip install libreyolo`.

Un clone fa il checkout di `release`, il branch stabile il cui codice corrisponde
a questa documentazione. Il branch di integrazione, che contiene il lavoro non
ancora rilasciato, è `dev`.

## Extra opzionali

Un extra è un nome tra parentesi quadre che aggiunge le dipendenze richieste da
una famiglia di modelli o da un target di esportazione. Non cambia nient'altro:
l'API è la stessa con o senza l'extra.

### Famiglie di modelli

| Extra | Aggiunge |
|---|---|
| `ground` | Le dipendenze VLM per il grounding da istruzioni a punti |
| `vlm-train` | Lo stack VLM più `peft>=0.17.0` per il fine-tuning Qwen3-VL |
| `vla` | `lerobot[smolvla,diffusion,dataset]>=0.6.1`, Python 3.12 o superiore |
| `marigold` | Dipendenze di diffusione, accelerazione e Transformers a versioni fissate |
| `molmo2` | `transformers==4.57.1`, `einops` e `accelerate` |
| `rfdetr` | `transformers`, che fornisce il backbone RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, che fornisce gli encoder ViT-L/16 ed EfficientNet-Lite3 di MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` e `bitsandbytes` fuori da macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` e `ftfy`, necessari al tokenizer testuale CLIP incluso nel codice |
| `siglip2` | `sentencepiece`, necessario al tokenizer multilingue SigLIP 2 |
| `gaze` | `gdown`, che abilita il download automatico del checkpoint L2CS |
| `rtdetr` | Nulla. RT-DETR non richiede dipendenze extra; il nome viene mantenuto stabile |

### Esportazione e runtime

| Extra | Aggiunge |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime>=1.18.0` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 e `pycuda`, fuori da macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, solo macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` più `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` e `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` più `MNN` |
| `ncnn` | `pnnx` e `ncnn` |
| `paddle` | `libreyolo[onnx]` più `paddlepaddle` 2.6.2 e `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` per inferenza HTTP e HTTPS V2 |

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
piattaforma senza wheel precompilata non può far fallire un'installazione
semplice. Quando il pacchetto manca, la valutazione COCO ripiega su pycocotools e
l'esecuzione continua.

### Strumenti

| Extra | Aggiunge |
|---|---|
| `hf` | `huggingface_hub>=1.0.0` per caricamento, pubblicazione e logger Hub |
| `llm` | `openai>=1.66.0` per endpoint API compatibili |
| `fiftyone` | `fiftyone>=1.0.0` per la cura dei dataset |
| `stream` | `yt-dlp`, necessario solo per risolvere URL di pagine YouTube |
| `tracking` | Nulla. Ogni dipendenza del tracking è già una dipendenza del core |
| `label` | `libreyolo[sam]`, che abilita l'assistenza da clic a maschera in `libreyolo label` |
| `hub-kernels` | `kernels`, il loader opzionale per i kernel Hub compilati. Vedi [kernel](/docs/reference/kernels), che spiega come l'installazione possa cambiare le predizioni RF-DETR entro la tolleranza floating-point |
| `clip-convert` | `libreyolo[clip]` più `open_clip_torch`, per conversione dei pesi e verifiche di parità |
| `siglip2-convert` | `libreyolo[siglip2]` più `transformers`, per lo stesso motivo |

Le webcam, RTSP, RTMP, TCP, UDP, HLS e le liste multi-stream locali non
richiedono alcun extra. Solo gli URL delle pagine di YouTube ne hanno bisogno.

### L'extra aggregato

`libreyolo[all]` installa gli extra per modelli, esportazione, tracking e logging con un solo comando. Alcuni sono volutamente esclusi. `neptune` è escluso perché la versione stabile di `neptune-scale` richiede protobuf inferiore a 7, mentre il percorso TFLite richiede protobuf 7. `executorch` è escluso perché ExecuTorch vincola la versione di PyTorch con cui si abbina e `coreai` perché `coreai-torch` fissa PyTorch a 2.11.x e porterebbe l'intero ambiente a quella versione. Sono esclusi anche `fast-eval`, `hub-kernels`, `clip-convert` e `siglip2-convert`. Installali per nome. `all` include `hf` e `llm`; `fiftyone`, `vla`, `marigold` e `molmo2` restano separati. FiftyOne include OpenCV headless, che si sovrappone al pacchetto `cv2` del core.

L'installazione del core richiede Python 3.10 o superiore e aggiunge `cloudpickle>=3.0.0` per DDP gestito da un coordinatore.

Usa un ambiente separato per Molmo2: il vincolo a Transformers 4.57.1 è in conflitto con gli stack VLM, Hub e Marigold più recenti. Marigold fissa diffusers 0.38.0, peft 0.18.1, accelerate 1.13.0 e Transformers 5.4.0, oltre a bitsandbytes 0.49.2 su Linux/Windows. L'inferenza predefinita Marigold a quattro bit richiede CUDA. North Micro Vision richiede Transformers 5.16 o superiore; Gemma 4 richiede 5.10 o superiore, oltre il minimo condiviso dei VLM.

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
semplice intero come `0`, o una stringa di cifre come `"0"`; gli ultimi due
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
