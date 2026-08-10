---
title: ncnn
seo_title: "Exportar para ncnn a partir do LibreYOLO"
description: "Exporte um modelo LibreYOLO para ncnn através do PNNX: o par param e bin, o canvas de exportação fixo, a reescrita do Focus do YOLOX e quais famílias convertem."
lead: "ncnn é a biblioteca de inferência em CPU da Tencent para alvos móveis. O LibreYOLO converte através do PNNX, escrevendo um grafo model.ncnn.param ao lado de um arquivo de pesos model.ncnn.bin e de um metadata.yaml que carrega a família, a tarefa e os nomes das classes."
keywords:
  - exportar yolo ncnn
  - pnnx
  - model.ncnn.param
  - inferência cpu mobile
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="ncnn")'
    mono: true
  - label: Escreve
    value: "Um diretório com model.ncnn.param, model.ncnn.bin e metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t_ncnn")'
    mono: true
  - label: Formas
    value: "Fixas. Os metadados registram dynamic=False independentemente da flag."
  - label: Precisão
    value: "Somente FP32. half=True e int8=True são rejeitados."
verification: "Lido de libreyolo/export/ncnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/ncnn.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        # O pnnx converte, o ncnn roda o resultado.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve o diretório weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, ou (altura, largura)
            batch=1,
            simplify=True,    # vale só para o caminho de fallback via ONNX
            opset=None,       # automático; vale só para o caminho de fallback via ONNX
            output_path=None, # None escreve weights/<stem>_ncnn
        )

        # half=True e int8=True são rejeitados durante a validação.
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn puro
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # O ncnn recebe uma única imagem CHW, não um batch.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # O pré-processamento e o pós-processamento são por sua conta neste caminho.
  support:
    - label: Conferir uma família e tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

O extra instala as duas metades do toolchain: o `pnnx` faz a conversão e o `ncnn`
executa o resultado. Nenhum dos dois passa por ONNX no caminho principal.

## Exportação

<code-tabs name="export" />

O artefato é um diretório. `weights/LibreYOLO9t_ncnn` contém
`model.ncnn.param`, `model.ncnn.bin` e `metadata.yaml`; os três são um único
artefato e andam juntos.

A conversão tenta o PNNX direto a partir do PyTorch primeiro. Se isso falhar, ela
exporta um grafo ONNX estático para um diretório temporário e chama a ferramenta de
linha de comando `pnnx` sobre ele, e a exportação só levanta erro quando os dois
caminhos falham, reportando os dois erros. Portanto, `opset` e `simplify` afetam
apenas o fallback.

O YOLOX precisa de uma reescrita para sequer converter. Sua camada Focus usa
fatiamento com stride, que o PNNX não consegue rebaixar, então a exportação a troca
por `pixel_unshuffle` e permuta os canais de entrada da convolução seguinte para
compensar a ordenação diferente dos canais. A saída é numericamente idêntica, e os
pesos originais são restaurados depois da exportação.

## Rodar o artefato

<code-tabs name="run" />

O `LibreYOLO()` reconhece qualquer diretório que contenha `model.ncnn.param` e
`model.ncnn.bin`, lê o `metadata.yaml` e devolve o mesmo objeto `Results` que o
checkpoint.

O segundo snippet é o caminho de runtime puro, e dois detalhes diferem de todos os
outros formatos aqui. O ncnn trabalha sobre uma única imagem CHW em vez de um batch,
então não há eixo de batch na frente. Os nomes dos blobs vêm do arquivo `.param`; o
PNNX escreve `in0` e `out0` por convenção, e o backend faz o parse do arquivo em vez
de assumi-los. Nesse caminho, o pré-processamento, a decodificação, o NMS e o
reescalonamento de coordenadas são por sua conta.

## Restrições

FP32 em um canvas fixo. `half=True` e `int8=True` são ambos rejeitados durante a
validação, e os metadados exportados registram `dynamic=False` independentemente do
que a flag dizia, para que nenhum backend assuma um eixo que o grafo não tem.

Toda família no estilo DETR é recusada no preflight: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr` e `ec`. A mensagem é a mesma para todas elas, dizendo que o modelo precisa
de operações de decoder ou de amostragem indisponíveis no ncnn, e apontando para
ONNX, OpenVINO, TorchScript ou TensorRT no lugar.

O que converte é amplo do lado convolucional: YOLO9 e YOLO9-E2E, YOLOX, PicoDet,
YOLO-NAS detecção e pose, os detectores mais antigos YOLO1, YOLO3, YOLO4 e YOLO7, as
quatro famílias de classificação CNN, a segmentação semântica PIDNet, a detecção de
pontos FOMO em 96 por 96 fixos, ZipDepth, NAFNet e Real-ESRGAN.

As entradas bloqueadas nomeiam a falha concreta. Grafos de transformer costumam
deixar para trás nós `pnnx.Expression` não suportados, o que produz uma rede sem
blob de entrada executável, e é isso que barra DINOv2, CLIP, SigLIP2 e SegFormer. O
BiRefNet precisa da convolução deformável do torchvision, que o PNNX não consegue
rebaixar. O grafo convertido do YOLO2 encerra o runtime do ncnn no Windows com uma
divisão inteira por zero nativa durante a extração da saída.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
