---
title: NVIDIA Jetson
seo_title: "Instalar LibreYOLO e PyTorch na NVIDIA Jetson"
description: "Instale o LibreYOLO em uma NVIDIA Jetson: as quatro bibliotecas CUDA que o JetPack deixa de fora, o passo --no-deps que o PyTorch exige e números medidos na Orin Nano."
lead: "As placas NVIDIA Jetson rodam o LibreYOLO com os wheels padrão do PyTorch para aarch64. Nenhuma build de torch específica para Jetson entra em jogo, mas o JetPack omite quatro bibliotecas contra as quais o torch linka, e a instalação precisa fornecê-las."
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - instalar pytorch na jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - tensorrt na jetson
  - wheels aarch64
last_verified: "1.4.0"
meta:
  - label: Placa
    value: "Jetson Orin Nano Super Developer Kit, 8 GB, compute capability de GPU 8.7"
  - label: Plataforma
    value: "JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64"
  - label: Stack testado
    value: "libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv 5.0.0, numpy 2.5.1, em 2026-07-27"
  - label: Ausentes no JetPack
    value: "nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13, nvidia-nvshmem-cu13"
    mono: true
  - label: Benchmarks
    value: "223 execuções verificadas nesta placa, 58 modelos de 12 famílias, em PyTorch, ONNX Runtime e TensorRT"
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: https://www.visionanalysis.org/hardware/jetson_orin
  - label: Acompanhado em
    value: "A metade Jetson da issue 648"
    links:
      - label: issue 648
        href: https://github.com/LibreYOLO/libreyolo/issues/648
verification: "Receita de instalação e saída esperada tiradas da instalação de 2026-07-27 em uma Jetson Orin Nano Super. As linhas de latência e acurácia vêm do snapshot de resultados verificados por trás do visionanalysis.org, filtrado pelo hardware jetson_orin, medido em junho de 2026 sobre libreyolo 1.2.0.dev0. Comportamento de exportação e do carregador lido de libreyolo/export/exporter.py, libreyolo/export/tensorrt.py e libreyolo/models/__init__.py."
snippets:
  prep:
    - label: Pacotes do sistema e um ambiente virtual
      language: bash
      code: |
        # O JetPack não vem com o pip nem com o módulo venv pré-instalados.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch, a partir do índice de wheels para CUDA 13
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: As quatro bibliotecas que o JetPack não entrega
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: Se o pip exigir o cuda-toolkit 13.0.3, instale com --no-deps
      language: bash
      code: |
        # --no-deps significa que as dependências Python do torch também são nomeadas na mão.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Descubra a próxima biblioteca faltante em vez de adivinhar
      language: bash
      code: |
        ldd "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Tudo que ainda falta em todas as bibliotecas do torch, de uma só vez:
        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so 2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: Instale o LibreYOLO depois do torch, não antes
      language: bash
      code: |
        # o torch já está satisfeito, então o pip deixa a build CUDA no lugar.
        pip install libreyolo

        # O extra ONNX só é necessário para exportar. Uma exportação para TensorRT
        # passa pelo ONNX, então instale-o antes da seção de exportação abaixo.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versões e dispositivo
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
    - label: Depois rode um kernel de verdade
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

        # Baixa o checkpoint no primeiro uso.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: |
        libreyolo predict --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE

        # Escreve libreyolo9s.onnx e, a partir dele, constrói libreyolo9s.engine.
        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt", half=True)

        # A engine é carregada de volta pelo mesmo ponto de entrada.
        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Modo de energia e clocks
      language: bash
      code: |
        sudo nvpmodel -q      # quais modos esta placa expõe, e qual está ativo
        sudo nvpmodel -m 0    # modo mais alto na placa testada aqui
        sudo jetson_clocks

        tegrastats            # carga ao vivo; o nvidia-smi é limitado no Tegra
---

## O que esta página registra

Esta página registra uma configuração que foi verificada de ponta a ponta, não uma
matriz de suporte. A placa foi uma Jetson Orin Nano Super Developer Kit com 8 GB de
memória rodando JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python 3.12.3), e o
stack que subiu nela foi `libreyolo 1.4.0` com `torch 2.13.0+cu130`, OpenCV
5.0.0 e NumPy 2.5.1. `torch.cuda.is_available()` retornou `True` e a GPU
se identificou como `Orin`.

Outras versões do JetPack, outras placas Jetson e outras versões do CUDA não foram
testadas. A receita abaixo é a que funcionou nessa combinação.

Essa execução foi em 2026-07-27 contra o LibreYOLO 1.4.0, e não foi repetida
em hardware com 1.5.0: esta é a única página da árvore 1.5.0 que ainda carrega uma
verificação de 1.4.0, e é por isso que o front matter diz `last_verified: "1.4.0"`.
Nada nas mudanças de 1.5.0 mexe no caminho de instalação, nas quatro bibliotecas
faltantes ou nas flags de exportação descritas aqui, então espera-se que os comandos
continuem valendo, mas os números de versão nas saídas abaixo são os que a 1.4.0
imprimiu, não uma medição de 1.5.0.

Duas coisas nisso contrariam o que a maioria dos guias de Jetson diz. Os wheels são
as builds aarch64 comuns publicadas para CUDA 13, então nenhuma build de torch
específica para Jetson é necessária. E o JetPack não entrega quatro bibliotecas
contra as quais esses wheels linkam, então o `import torch` falha uma biblioteca de
cada vez até que as quatro estejam instaladas.

## Instalação

As imagens do JetPack chegam sem pip e sem o módulo `venv`, então os dois vêm
primeiro.

<code-tabs name="prep" />

Uma placa de 8 GB é apertada para checkpoints maiores. Adicionar swap no NVMe antes
de carregá-los evita um kill por falta de memória no meio da execução.

Depois o PyTorch. O índice de CUDA 13 carrega os wheels aarch64; o índice extra
fornece as dependências puramente Python do PyPI.

<code-tabs name="torch" />

Os quatro wheels `nvidia-*-cu13` são a parte fácil de esquecer. O JetPack
fornece o driver da GPU, não o cuDNN, o NCCL, o cuSPARSELt nem o NVSHMEM, e o torch
se recusa a importar sem eles. Instalar os quatro de uma vez é mais rápido do que
descobri-los uma exceção por vez.

O terceiro snippet cobre uma falha específica: os metadados de dependência do torch
para a build CUDA 13 pedem `cuda-toolkit==13.0.3`, que não tem wheel aarch64 no
PyPI, então a resolução falha antes de qualquer download. `--no-deps` pula o
resolvedor, o que significa que toda dependência tem que ser nomeada na linha de comando.

O LibreYOLO entra por último. Instalá-lo primeiro deixa o pip escolher o próprio
torch, que nesta plataforma não é a build CUDA.

<code-tabs name="install" />

Todas as demais dependências resolvem para um wheel aarch64 pré-compilado, incluindo
OpenCV, NumPy, SciPy, pycocotools e safetensors. Nada compila a partir do código-fonte.

## Confira que o CUDA funciona

<code-tabs name="verify" />

O segundo snippet importa tanto quanto o primeiro. Um wheel compilado para a
arquitetura de GPU errada ainda reporta `torch.cuda.is_available() == True` e então
falha na primeira operação de verdade com `CUDA error: no kernel image is available for
execution on the device`. Uma multiplicação de matrizes no dispositivo é a checagem que
pega isso.

## Rodar uma predição

<code-tabs name="predict" />

O `predict` retorna o mesmo objeto `Results` que em qualquer outra plataforma, então as
páginas dos modelos valem sem mudanças.

## Exportar para TensorRT

Nesta placa, o TensorRT foi mais rápido que o PyTorch e o ONNX Runtime nos 55
modelos que foram medidos em todos os runtimes.

<code-tabs name="export" />

`format="tensorrt"` escreve primeiro um grafo ONNX e constrói a engine a partir dele,
então o extra `onnx` precisa estar instalado. `LibreYOLO()` despacha pelo sufixo do
arquivo, então um arquivo `.engine` carrega pela mesma chamada que um checkpoint `.pt`.

Não use o extra pip `tensorrt` em uma Jetson. Ele fixa o `tensorrt-cu12`, uma build
de CUDA 12, contra uma plataforma CUDA 13. Use o TensorRT que o JetPack instala
no lugar dele. Se o `import tensorrt` falhar dentro do ambiente virtual mas
funcionar fora, recrie o ambiente com `--system-site-packages` para que o
módulo do sistema fique visível.

Engines serializadas do TensorRT ficam presas ao dispositivo, à arquitetura da GPU e à
versão do TensorRT que as construiu. Uma engine construída em uma workstation não vai
carregar em uma Jetson, então o passo de build roda na própria placa.

## Medido nesta placa

Latência por imagem, tamanho de batch 1, de ponta a ponta incluindo pré-processamento e
pós-processamento, no COCO val2017 (subconjunto de 500 imagens) com `conf=0.001` e
`max_det=300`. Cinco modelos dos 58 medidos:

| Modelo | Entrada (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

A coluna mAP é a pontuação da própria execução TensorRT FP16. Entre os 55 modelos
medidos nos quatro runtimes, a maior diferença entre a pontuação PyTorch FP32 e a
pontuação TensorRT FP16 foi de 0.59 pontos, no DEIMv2-X. Os runtimes diferem em
velocidade, não em acurácia.

O TensorRT FP32 foi mais rápido que o PyTorch e o ONNX Runtime nos 55 modelos.
O TensorRT FP16 também foi mais rápido que o PyTorch FP32 nos 55, de 1.68x
a 6.22x, com mediana de 3.39x. O ONNX Runtime é o que varia: foi
mais lento que o PyTorch em 23 dos 55, a linha do RT-DETR-r18 entre eles.

Condições por trás de cada número: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, medido em junho de 2026.
A latência em uma Jetson também depende do modo de energia ativo, que os registros do
benchmark não carregam.

<code-tabs name="power" />

Todas as 223 execuções, incluindo os outros 53 modelos e as colunas completas de acurácia, estão
publicadas em
[a página da Jetson Orin no Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Solução de problemas

### import torch falha citando uma biblioteca compartilhada

Uma das quatro bibliotecas acima está faltando. Em vez de adivinhar qual, leia isso
direto do binário:

<code-tabs name="ldd" />

Cada entrada faltante corresponde a um wheel:

| Biblioteca faltante | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### o torch avisa que nenhuma build suporta esta GPU

A primeira chamada CUDA na configuração que funciona imprime isto:

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

O aviso é cosmético nesta placa. O wheel carrega kernels `sm_80` e a
Orin os executa. O mesmo aviso apareceu no wheel anterior daquele
índice, o que produziu todas as linhas de benchmark acima. Confirme com a multiplicação
de matrizes da checagem do CUDA em vez de confiar ou desconfiar da mensagem.

### CUDA error: no kernel image is available for execution on the device

O wheel instalado foi compilado para outra arquitetura de GPU. É isso que
acontece com wheels do índice `sbsa` da NVIDIA, que miram GPUs ARM de servidor
em vez do silício da Jetson. Reinstale a partir do índice de CUDA 13 da seção de
instalação.

### o pip não encontra o cuda-toolkit 13.0.3

Não existe wheel aarch64 para ele. Use a forma `--no-deps` da seção de
instalação e nomeie as dependências do torch explicitamente.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

O wheel de torch para aarch64 linka as NVIDIA Performance Libraries para matemática de CPU. Instale-as
e coloque-as no library path:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Esse índice serve bem para essas duas bibliotecas de CPU. As builds de torch dele são as que
produzem a falha de "no kernel image" acima.

### Fontes de wheels que não servem para o JetPack 7.2

| Fonte | Resultado na Orin Nano Super |
|---|---|
| torch de `pypi.jetson-ai-lab.io/sbsa/cu130` | Compilado para GPUs ARM de servidor. Importa, reporta CUDA disponível e então falha com "no kernel image is available for execution on the device". |
| torch de `pypi.jetson-ai-lab.io/jp6/*` | Builds de CUDA 12 e Python 3.10. Não instalam no Python 3.12 desta imagem. |
| Contêineres PyTorch do JetPack 6 | A inicialização do CUDA falha com erro 801 em um host JetPack 7. |
| Compilar o torch a partir do código-fonte | Funciona, mas leva horas em uma placa de 8 GB e é desnecessário depois que os wheels de CUDA 13 estão instalados. |

## DeepStream

Para um pipeline de vídeo completo em vez de um laço em Python, exporte com
`deepstream=True` e rode o grafo pelo `nvinfer`. Esse caminho tem sua própria
página, incluindo o config gerado do `nvinfer`, a build do parser de bounding box e
as armadilhas conhecidas: [DeepStream](/docs/export/deepstream).

O pipeline do DeepStream em si foi validado em uma GPU discreta x86, não em uma
Jetson. O contrato de exportação não depende da arquitetura, mas a execução do pipeline
em aarch64 continua pendente.

## Não verificado

- Versões do JetPack diferentes da 7.2, e versões do L4T diferentes da R39.2.
- Placas Jetson diferentes da Orin Nano Super 8 GB.
- Treinamento na placa. Inferência e exportação foram exercitadas; uma execução de
  treinamento não.
- Engines INT8. Só existem linhas FP32 e FP16 para esta placa.
- Tamanhos de batch acima de 1. Toda medição acima é batch 1.
