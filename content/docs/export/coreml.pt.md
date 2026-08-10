---
title: Core ML
seo_title: "Exportar para Core ML a partir do LibreYOLO"
description: "Exporte um detector LibreYOLO para um .mlpackage do Core ML: o contrato de entrada ImageType, FP16, as compute units, o NMS embutido e as quatro famílias suportadas."
lead: "Core ML é o formato de modelos on-device da Apple. O LibreYOLO traça o detector por trás de um wrapper de pré-processamento específico de cada família, de modo que o grafo convertido sempre recebe uma entrada de imagem RGB canônica, e então escreve um .mlpackage em formato ML Program com os metadados do modelo anexados."
keywords:
  - exportar yolo coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - nms embutido coreml
  - yolo no ios
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="coreml")'
    mono: true
  - label: Escreve
    value: "Um bundle .mlpackage (um diretório) em formato ML Program"
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.mlpackage") no macOS'
    mono: true
  - label: Formas
    value: "Fixas. A entrada é um ct.ImageType de forma rígida."
  - label: Precisão
    value: "FP32, FP16 (half=True). Sem INT8."
  - label: Famílias
    value: "Somente detecção, para yolox, yolo9, rtdetr e rfdetr"
verification: "Lido de libreyolo/export/coreml.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/coreml.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve o bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True converte com precisão de cálculo FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None escreve weights/<stem>.mlpackage
        )

        # dynamic é aceito, mas a entrada é um ct.ImageType de forma fixa,
        # e os metadados embutidos registram dynamic=False de qualquer jeito.
  nms:
    - label: Embutir a camada de NMS da Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Somente detecção com YOLOX e YOLO9, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: Pelo LibreYOLO, no macOS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # ou cpu_and_ne para fixar o Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools puro
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # A entrada é uma imagem chamada "image" no tamanho fixo da exportação.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # O letterboxing e o pós-processamento ficam por sua conta neste caminho.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

A predição precisa de macOS. `LibreYOLO()` recusa um `.mlpackage` em qualquer outra
plataforma com uma mensagem nomeando a plataforma atual, e a matriz de suporte registra
essas combinações como disponíveis porque a paridade de runtime exige um runner macOS.

## Exportação

<code-tabs name="export" />

O bundle é escrito em `weights/` com o nome-base do checkpoint, com `_fp16`
acrescentado quando `half=True`. Um `.mlpackage` é um diretório, então copie a árvore inteira.

Toda família é traçada por trás de um wrapper de pré-processamento, de modo que o grafo
convertido recebe uma única entrada canônica: RGB, `scale=1/255`, sem bias, declarada como
`ct.ImageType`. O wrapper absorve a convenção própria da família, que é BGR na
faixa de 0 a 255 para o YOLOX, média e desvio padrão do ImageNet para o RF-DETR,
e identidade para o YOLO9 e o RT-DETR. É por isso que um consumidor Core ML recebe uma
imagem comum em vez de um tensor específico da família.

A conversão tem como alvo o ML Program com um deployment target mínimo de iOS 15.
`compute_units` fica armazenado no modelo convertido e pode ser sobrescrito de novo quando
o artefato é carregado.

Os metadados do modelo vão para `user_defined_metadata` como strings, que é de onde o
backend lê a família, a tarefa, os nomes das classes, o tamanho de entrada e o esquema de pose.

### NMS embutido

<code-tabs name="nms" />

`nms=True` envolve o modelo em um pipeline do Core ML que termina na camada
`NonMaximumSuppression` da Apple. O resultado tem duas saídas: `confidence`, com formato
`N` pela contagem de classes, e `coordinates`, com formato `N` por 4 como `xywh` normalizado.

Vale somente para detecção com YOLOX e YOLO9, e exige batch 1. As
famílias no estilo DETR são recusadas pelo nome, porque a predição de conjunto faz um top-k sobre
queries e classes sem nenhuma etapa de IoU e não consegue usar essa camada. `max_det` também não é
exposto aqui; quando o limite de detecções importa, use o
[NMS embutido do ONNX](/docs/export/onnx) em vez disso.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` reconhece um diretório com o sufixo `.mlpackage` e devolve o
mesmo objeto `Results` que o checkpoint. `compute_units` é o único argumento que a
factory repassa para este formato, e ele aceita `all`, `cpu_and_gpu`,
`cpu_and_ne` e `cpu_only`. O argumento `device` é ignorado, porque o Core ML
roteia por compute units.

O segundo snippet é o caminho do runtime puro. O letterboxing, a decodificação, o NMS e o
reescalonamento de coordenadas ficam por sua conta ali, e os nomes das classes vivem em
`user_defined_metadata`.

## Restrições

Quatro famílias, somente detecção: `yolox`, `yolo9`, `rtdetr` e `rfdetr`. Qualquer outra
coisa é recusada no preflight, porque o wrapper de pré-processamento ciente da família é
o que torna correto o contrato de entrada de imagem fixa, e uma família fora dele
converteria com a normalização errada. O erro nomeia ONNX e TorchScript como
alternativas.

O formato de entrada é fixado de forma rígida por `ct.ImageType`, então `dynamic=True` não muda nada
e os metadados registram `dynamic=False`. Exporte um segundo bundle para uma segunda
resolução.

`half=True` converte com precisão de cálculo FP16. Não existe caminho para INT8 a partir deste
exportador.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para o formato on-device mais recente
da Apple, veja [Core AI](/docs/export/coreai). Para uma combinação:

<code-tabs name="support" />
