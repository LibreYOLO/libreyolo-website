---
title: Instalação
seo_title: Instalar o LibreYOLO
description: >-
  Instale o LibreYOLO a partir do PyPI, escolha os extras opcionais de que uma
  família de modelos ou um destino de exportação precisa, e confirme que o
  PyTorch enxerga sua GPU.
lead: >-
  O LibreYOLO é publicado no PyPI como libreyolo. O pacote base cobre predição,
  treinamento, validação e as famílias de modelos que não precisam de nada além
  do PyTorch; os extras opcionais adicionam o resto.
keywords:
  - instalar libreyolo
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo gpu
  - requisitos libreyolo
last_verified: 1.5.0
meta:
  - label: Pacote
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 ou superior
  - label: Licença do código
    value: MIT
  - label: Dependência principal
    value: PyTorch 2.4 ou superior
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Com extras
      language: bash
      code: |
        # Separe por vírgulas para combinar vários em uma só instalação.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Tudo
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: A partir do código-fonte
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, cada GPU visível e quais
        # pacotes opcionais estão instalados.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inventário de modelos
      language: bash
      code: |
        # Cada família registrada com suas tarefas, tamanhos e
        # resoluções de entrada. Famílias cujo extra está faltando são
        # listadas com o comando pip que as habilita.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Instalação

<code-tabs name="install" />

É necessário Python 3.10 ou superior. A instalação base traz PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors e SciPy, então o YOLOv9 e as demais famílias que não
precisam de mais nada funcionam logo após `pip install libreyolo`.

Um clone faz checkout de `release`, o branch estável cujo código corresponde a
esta documentação. O branch de integração, que carrega o trabalho ainda não
lançado, é `dev`.

## Extras opcionais

Um extra é um nome entre colchetes que adiciona as dependências de que uma
família de modelos ou um destino de exportação precisa. Nada mais muda: a API é
a mesma com ou sem o extra.

### Famílias de modelos

| Extra | Adiciona |
|---|---|
| `rfdetr` | `transformers`, que fornece o backbone do RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, que fornece os encoders ViT-L/16 e EfficientNet-Lite3 do MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` e `bitsandbytes` fora do macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` e `ftfy`, necessários para o tokenizador de texto do CLIP embutido |
| `siglip2` | `sentencepiece`, necessário para o tokenizador multilíngue do SigLIP 2 |
| `gaze` | `gdown`, que liga o download automático do checkpoint L2CS |
| `rtdetr` | Nada. O RT-DETR não precisa de dependência extra; o nome é mantido por estabilidade |

### Exportação e runtimes

| Extra | Adiciona |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 e `pycuda`, fora do macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, apenas no macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` mais `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` e `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` mais `MNN` |
| `ncnn` | `pnnx` e `ncnn` |
| `paddle` | `libreyolo[onnx]` mais `paddlepaddle` 2.6.2 e `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` para inferência V2 por HTTP e HTTPS |

### Treinamento, avaliação e logging

| Extra | Adiciona |
|---|---|
| `lora` | `libreyolo[rfdetr]` mais `peft`, para fine-tuning com `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, o backend C++ de avaliação COCO |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` é opcional em vez de dependência obrigatória para que uma plataforma
sem wheel pré-compilada não quebre uma instalação simples. Quando o pacote está
ausente, a avaliação COCO recorre ao pycocotools e a execução continua.

### Ferramentas

| Extra | Adiciona |
|---|---|
| `stream` | `yt-dlp`, necessário apenas para resolver URLs de páginas do YouTube |
| `tracking` | Nada. Toda dependência de tracking já é uma dependência principal |
| `label` | `libreyolo[sam]`, que habilita o auxílio de clique-para-máscara em `libreyolo label` |
| `hub-kernels` | `kernels`, o carregador opcional dos kernels compilados do Hub. Veja [kernels](/docs/reference/kernels), que observa que instalá-lo pode deslocar as predições do RF-DETR na tolerância de float |
| `clip-convert` | `libreyolo[clip]` mais `open_clip_torch`, para conversão de pesos e verificações de paridade |
| `siglip2-convert` | `libreyolo[siglip2]` mais `transformers`, pelo mesmo motivo |

Webcams, RTSP, RTMP, TCP, UDP, HLS e listas locais de múltiplos streams não
precisam de extra. Só as URLs de páginas do YouTube precisam.

### O extra agregado

`libreyolo[all]` instala os extras de modelos, exportação, tracking e logging em
um único comando. Alguns ficam deliberadamente de fora. `neptune` é excluído
porque o `neptune-scale` estável exige protobuf abaixo de 7, enquanto o caminho
do TFLite exige protobuf 7. `executorch` é excluído porque o ExecuTorch limita
com qual versão do PyTorch ele se combina, e `coreai` porque `coreai-torch` fixa
o PyTorch em 2.11.x e arrastaria todo o ambiente para essa versão. `fast-eval`,
`hub-kernels`, `clip-convert` e `siglip2-convert` também ficam de fora. Instale
qualquer um deles pelo nome.

## Restrições de plataforma

Três extras são delimitados por plataforma pelos seus marcadores de dependência,
então a instalação funciona em todo lugar e simplesmente instala menos onde não
existe wheel.

| Extra | Restrição |
|---|---|
| `coreai` | Apenas macOS. A toolchain do Core AI não converte nem roda em outro lugar |
| `tensorrt` | Ignorado no macOS, que não tem CUDA |
| `tflite`, `litert` | `onnx2tf` e `ai-edge-litert` exigem Python 3.12 ou superior |

`sensenova` pula `bitsandbytes` no macOS, onde nenhuma wheel é publicada; o
resto do extra instala normalmente.

Se a restrição for o disco, a maior parte dele é o PyTorch, e a maior parte do
PyTorch é a carga CUDA que a wheel padrão embute. Uma wheel só de CPU remove
isso sem abrir mão de nada. Para detecção com ONNX em uma máquina que não deve
carregar torch nenhum, veja a [instalação leve](/docs/lightweight-install).

## GPU e CUDA

A seleção de dispositivo acontece quando o modelo é construído. O padrão,
`device="auto"`, usa CUDA quando `torch.cuda.is_available()` é verdadeiro,
depois Metal Performance Shaders quando `torch.backends.mps.is_available()` é
verdadeiro, e CPU caso contrário. Nada mais na biblioteca inspeciona o hardware,
então, se o PyTorch não enxerga uma GPU, o LibreYOLO também não.

Para fixar o dispositivo, passe `device` ao modelo ou a `predict`, `train`,
`val` e `export`. Ele aceita `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, um inteiro
puro como `0`, ou uma string de dígito como `"0"`; os dois últimos são
expandidos para `cuda:<n>`.

Comece por `libreyolo checks`, que imprime a versão do Torch, as versões de CUDA
e cuDNN contra as quais o Torch foi compilado, e cada GPU visível com sua
memória. Quando ele informa que não há CUDA em uma máquina que tem placa NVIDIA,
a wheel do PyTorch que o pip resolveu é uma build de CPU. Instale uma build com
CUDA a partir do índice do PyTorch primeiro e só então instale o LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Esse é o mesmo índice que o repositório fixa para o próprio ambiente gerenciado
por uv no Linux e no Windows. Ele precisa do driver NVIDIA 555 ou superior, que
é o requisito do runtime CUDA 12.8. O macOS fica com a wheel do PyPI, já que o
host de download do PyTorch não publica builds para Darwin.

## Conferir a instalação

<code-tabs name="verify" />

`libreyolo models` é a forma mais rápida de ver se um extra fez efeito: uma
família cuja dependência está faltando é impressa com o comando pip exato que a
habilita. Os dois comandos também aceitam `--json`, que imprime os mesmos dados
como um objeto legível por máquina no stdout.
