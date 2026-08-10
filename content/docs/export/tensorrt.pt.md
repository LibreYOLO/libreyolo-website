---
title: TensorRT
seo_title: "Exportar para TensorRT a partir do LibreYOLO"
description: "Construa um engine TensorRT a partir de um modelo LibreYOLO: o intermediário ONNX, as builds FP16 e INT8, os perfis de batch dinâmico e os limites de portabilidade do engine."
lead: "O TensorRT compila um grafo em um engine ajustado para uma GPU específica. O LibreYOLO exporta primeiro um intermediário ONNX, faz o parse dele com o parser ONNX do TensorRT, constrói o engine e escreve os metadados do modelo ao lado dele como um sidecar JSON."
keywords:
  - exportar yolo tensorrt
  - engine tensorrt
  - trt fp16
  - calibração int8 tensorrt
  - perfil de otimização
  - batch dinâmico tensorrt
  - hardware compatibility level
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="tensorrt")'
    mono: true
  - label: Escreve
    value: "Um arquivo .engine mais um sidecar de metadados .engine.json"
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.engine")'
    mono: true
  - label: Formas
    value: "Estáticas por padrão; dynamic=True adiciona um perfil de otimização no eixo de batch"
  - label: Precisão
    value: "FP32, FP16 (half=True), INT8 (int8=True com data=)"
  - label: Requer
    value: "Uma GPU NVIDIA na hora de construir e na hora de rodar. Engines não migram entre arquiteturas de GPU."
verification: "Lido de libreyolo/export/tensorrt.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/tensorrt.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        # O engine é construído a partir de um intermediário ONNX, então os dois extras são necessários.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Conferir o toolchain antes de construir
      language: bash
      code: |
        python -c "import tensorrt, torch; print(tensorrt.__version__, torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t_fp16.engine e weights/LibreYOLO9t_fp16.engine.json
        path = model.export(format="tensorrt", half=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # obrigatório quando int8=True
            dynamic=False,
            workspace=4.0,                  # GiB de memória temporária de build
            min_batch=1,                    # limites do perfil dinâmico
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # ou "ampere_plus"
            gpu_device=0,                   # dispositivo de build em uma máquina com várias GPUs
            verbose=False,
        )
  dynamic:
    - label: Engine com batch dinâmico
      language: python
      code: |
        from libreyolo import LibreYOLO

        # O intermediário ONNX precisa do eixo de batch dinâmico para o perfil
        # ter em que se apoiar.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 com dados de calibração
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # obrigatório: não existe padrão para este formato
            fraction=1.0,
        )
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT puro
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Os nomes das classes, a tarefa e o tamanho de entrada ficam no sidecar, não no engine.
        # A alocação de buffers, o pré-processamento e o pós-processamento ficam por sua conta aqui.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Conferir uma família e uma tarefa antes de construir
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

Tanto a build quanto a execução precisam de uma GPU NVIDIA com uma stack CUDA
funcionando. Não há fallback de CPU para este formato.

<code-tabs name="install" />

O extra `tensorrt` fixa `tensorrt-cu12` e `pycuda`, e o marker descarta os dois
no macOS. Em um Jetson, não use esse extra: ele fixa uma build de CUDA 12 contra
uma plataforma CUDA 13. Use o TensorRT que o JetPack instala, como descrito em
[NVIDIA Jetson](/docs/export/jetson).

## Exportação

<code-tabs name="export" />

A exportação roda em duas etapas. A primeira escreve um intermediário ONNX em um
caminho temporário, a segunda faz o parse dele e constrói o engine, e o
intermediário é removido depois. `workspace` é a memória temporária de build em
GiB; um valor maior deixa o builder testar mais kernels e não afeta a memória de
inferência.

O sidecar de metadados é escrito ao lado do engine como `<engine>.json` e
registra a precisão que a build de fato conseguiu. Quando a GPU não tem FP16
rápido ou INT8 rápido, o builder avisa e cai para outra precisão, e o sidecar
informa a precisão que saiu, não a que foi pedida.

Em FP16, um backbone ViT no grafo é detectado e suas camadas float são fixadas em
FP32. Backbones no estilo DINOv2 estouram em FP16 e produzem NaN, então a build
ativa `OBEY_PRECISION_CONSTRAINTS` e informa `FP16 (FP32 ViT backbone)`. O passo
não faz nada em backbones CNN.

### Batch dinâmico

<code-tabs name="dynamic" />

`dynamic=True` adiciona um perfil de otimização que vai de `min_batch` até
`max_batch`, otimizado em `opt_batch`, e registra esses três valores no sidecar.
O perfil só é adicionado quando o intermediário ONNX de fato carrega uma dimensão
de batch dinâmica; caso contrário, a build registra no log que está usando
otimização estática e segue em frente.

### INT8

<code-tabs name="int8" />

O INT8 usa o calibrador de entropia do TensorRT sobre um loader de calibração do
LibreYOLO, e `data` é obrigatório: este formato não tem fallback de oito imagens.
A calibração precisa de `cuda-python` ou `pycuda` para o buffer no dispositivo. O
cache de calibração é indexado por um hash dos bytes do ONNX, então as escalas de
um modelo nunca são reaproveitadas por outro que por acaso escreva no mesmo
caminho de saída.

`half=True` e `int8=True` juntos geram um aviso e constroem em INT8, que mantém
um fallback FP16 para as camadas que o TensorRT não consegue quantizar.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` decide pelo sufixo `.engine`, lê no sidecar os nomes das classes, a
tarefa e o schema de pose, e devolve o mesmo objeto `Results` que o checkpoint.
Ele levanta erro na hora quando não há nenhum dispositivo CUDA presente.

O segundo snippet é o caminho do runtime puro. A alocação de buffers no host e no
dispositivo, o pré-processamento, a decodificação, o NMS e o reescalonamento de
coordenadas passam todos a ser sua responsabilidade, e o engine em si não carrega
nomes de classes, então o sidecar precisa viajar junto com ele.

## Restrições

Um engine serializado está preso à arquitetura da GPU, à stack de drivers e à
versão do TensorRT que o construiu. Um engine construído em uma workstation não
vai carregar em outra arquitetura, e é por isso que a etapa de build roda na
máquina de deploy. `hardware_compatibility="ampere_plus"` troca um pouco de
desempenho por portabilidade entre Ampere e arquiteturas mais novas. O valor
`"same_compute_capability"` mapeia para `NONE` e gera um aviso: o engine é
otimizado só para a GPU atual, e a exportação diz isso em vez de alegar uma
portabilidade que não aplicou.

Só o eixo de batch entra no perfil. Uma build com dimensões espaciais dinâmicas
não faz parte deste contrato, e é por isso que o FCOS está bloqueado: ele precisa
de altura e largura com padding dinâmico para preservar sua transformação de
proporção 800 por 1333.

Bloqueados antes do tracing: segmentação com YOLO9, segmentação com RTMDet-Ins,
detecção com SSD, Faster R-CNN e RetinaNet, e matting com BiRefNet ou FeyNobg,
casos em que o TensorRT 10.16 chega ao nó ONNX `DeformConv` compartilhado e não
consegue fazer o parse dele porque `ModulatedDeformConv2d` não está no registro
de plugins.

Quando uma combinação não está nem validada nem bloqueada, o caminho do conversor
está disponível e o projeto não registrou paridade de runtime TensorRT para ela.
Isso é uma afirmação sobre evidência, não sobre se a build funciona ou não.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação
específica:

<code-tabs name="support" />
