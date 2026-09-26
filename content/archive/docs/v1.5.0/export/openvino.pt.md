---
title: OpenVINO
seo_title: Exportar para OpenVINO IR a partir do LibreYOLO
description: >-
  Converta um modelo LibreYOLO para OpenVINO IR: o par model.xml e model.bin, a
  compressão de pesos FP16, o INT8 do NNCF e a inferência em CPU, GPU ou NPU.
lead: >-
  O OpenVINO IR é o formato de runtime da Intel, um grafo model.xml ao lado de
  um blob de pesos model.bin. O LibreYOLO exporta um intermediário ONNX,
  converte com ov.convert_model e escreve um metadata.yaml no mesmo diretório.
keywords:
  - exportar yolo openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - quantização int8 nncf
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="openvino")
    mono: true
  - label: Escreve
    value: 'Um diretório com model.xml, model.bin e metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Recarrega com
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Formas
    value: 'Seguem o intermediário ONNX: batch dinâmico quando dynamic=True'
  - label: Precisão
    value: >-
      FP32, compressão de pesos FP16 (half=True), INT8 via NNCF (int8=True com
      data=)
verification: >-
  Lido de libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py e pyproject.toml
  no branch dev.
snippets:
  install:
    - label: Instalação
      language: bash
      code: >
        # O IR é convertido a partir de um intermediário ONNX, então os dois
        extras são necessários.

        pip install "libreyolo[onnx,openvino]"
    - label: O INT8 ainda exige o NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve o diretório weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True mantém um eixo de batch dinâmico através do IR
            half=False,       # True armazena pesos FP16
            int8=False,       # True roda a quantização pós-treinamento do NNCF
            data=None,        # obrigatório quando int8=True
            output_path=None, # None escreve weights/<stem>_openvino
        )
  int8:
    - label: INT8 com dados de calibração
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # obrigatório: não há padrão para este formato
            fraction=1.0,
        )
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Selecionar o dispositivo
      language: python
      code: >
        from libreyolo import LibreYOLO


        # "auto" e "cpu" mapeiam para CPU, "gpu" e "cuda" mapeiam para GPU,

        # qualquer outra coisa é repassada em maiúsculas, por exemplo "npu" ->
        NPU.

        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO puro
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Os nomes das classes, a tarefa e o tamanho de entrada vivem no
        metadata.yaml ao lado do IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # O pré-processamento e o pós-processamento ficam por sua conta neste
        caminho.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Instalação

<code-tabs name="install" />

A conversão passa por um intermediário ONNX, então o extra `onnx` faz parte do
requisito em vez de ser um acompanhante opcional. O NNCF é uma instalação separada
e só é necessário para `int8=True`.

## Exportação

<code-tabs name="export" />

O artefato é um diretório, não um arquivo. `weights/LibreYOLO9t_openvino` contém
`model.xml`, `model.bin` e `metadata.yaml`, e `_fp16` é inserido antes do sufixo
quando `half=True`. Mova ou copie o diretório inteiro; os três arquivos são um só
artefato.

`half=True` define `compress_to_fp16` no salvamento. Isso é compressão de pesos no IR,
não uma mudança na precisão de inferência que o dispositivo escolhe em tempo de execução.

### INT8

<code-tabs name="int8" />

`int8=True` roda a quantização pós-treinamento do NNCF sobre um loader de calibração
do LibreYOLO com o preset mixed, e `data` é obrigatório: este formato não tem
fallback de oito imagens. A ausência do NNCF levanta um `ImportError` nomeando o
comando de instalação.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` reconhece qualquer diretório que contenha `model.xml` e devolve o mesmo
objeto `Results` que o checkpoint, lendo os nomes das classes, a tarefa, o tamanho de
entrada e o esquema de pose do `metadata.yaml`.

A string de dispositivo é mapeada em vez de repassada direto. `auto` e `cpu` compilam
os dois para CPU, `gpu` e `cuda` compilam os dois para GPU, e qualquer outro valor é
convertido para maiúsculas e entregue ao OpenVINO, que é como se chega a um alvo NPU.

O terceiro snippet é para leitores sem o LibreYOLO instalado. O pré-processamento, a
decodificação, o NMS e o reescalonamento de coordenadas ficam por sua conta ali, e os
nomes das classes só existem no `metadata.yaml`.

## Restrições

Um IR sem o seu `metadata.yaml` ainda carrega, mas o backend então recorre a 80 classes
e à tarefa de detecção, o que está errado para qualquer outra coisa. Mantenha o
diretório intacto.

Bloqueados antes do traçado: segmentação com YOLO9, segmentação com RTMDet-Ins,
detecção com SSD, Faster R-CNN e RetinaNet, e matting com BiRefNet ou FeyNobg, onde
o OpenVINO 2026.2 não consegue rebaixar a operação padrão do ONNX `DeformConv-19` do
decodificador de matte compartilhado.

Onde uma combinação não é nem validada nem bloqueada, o caminho do conversor está
disponível e o projeto não registrou paridade de runtime do OpenVINO para ela. Várias
combinações são validadas com um contexto explícito anexado, por exemplo a segmentação
semântica com DeepLabV3 em uma entrada fixa de 520 por 520 no OpenVINO 2026.2 com a
precisão de inferência padrão da CPU, e o olhar (gaze) com L2CS em um recorte de rosto
fixo de 448 por 448. O `libreyolo formats` imprime esse contexto por combinação.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
