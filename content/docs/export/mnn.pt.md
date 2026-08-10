---
title: MNN
seo_title: "Exportar para MNN a partir do LibreYOLO"
description: "Exporte um detector LibreYOLO para MNN passando por ONNX e mnnconvert: uma forma NCHW fixa, FP32 na CPU e um sidecar de metadados que o contrato do runtime exige."
lead: "MNN é o motor de inferência leve da Alibaba. O LibreYOLO exporta um grafo ONNX estático, converte esse grafo com a ferramenta mnnconvert que vem no pacote MNN e escreve um sidecar JSON registrando os nomes de entrada e de saída, a forma de entrada fixa e os nomes das classes."
keywords:
  - exportar yolo mnn
  - mnnconvert
  - inferência mnn
  - inferência de detector no celular
  - forma nchw fixa
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="mnn")'
    mono: true
  - label: Escreve
    value: "Um arquivo .mnn mais um sidecar de metadados .mnn.json"
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.mnn")'
    mono: true
  - label: Formas
    value: "NCHW fixa. dynamic=True é rejeitado."
  - label: Precisão
    value: "Somente FP32, somente CPU."
  - label: Tarefas
    value: "Somente detecção nesta versão"
verification: "Lido de libreyolo/export/mnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/mnn.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        # O extra inclui libreyolo[onnx]: o MNN converte a partir de um intermediário ONNX.
        pip install "libreyolo[mnn]"
    - label: Confirmar que o conversor está no path
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.mnn e weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, ou (altura, largura)
            batch=1,          # embutido no artefato
            simplify=True,    # onnxsim sobre o intermediário ONNX
            output_path=None, # None escreve weights/<stem>.mnn
            verbose=False,    # True exibe o log do mnnconvert
        )

        # dynamic=True levanta ValueError. half=True e int8=True são rejeitados.
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN puro
      language: python
      code: |
        import json

        import MNN
        import numpy as np

        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))
        print(meta["mnn_input_names"], meta["mnn_output_names"], meta["mnn_input_shape"])

        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )
        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )

        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)
        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )
        outputs = module.forward([input_var])
        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # O pré-processamento e o pós-processamento são por sua conta nesse caminho.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

O extra inclui `libreyolo[onnx]`, porque a conversão roda sobre um intermediário
ONNX. Ele também traz o executável `mnnconvert`, que o exportador procura primeiro
ao lado do interpretador Python ativo e depois no `PATH`. Um conversor ausente
levanta um `ImportError` que nomeia o comando de instalação, em vez de falhar no
meio da conversão.

## Exportação

<code-tabs name="export" />

Antes de entregar o grafo, o exportador lê o contrato de entrada do ONNX e recusa
qualquer coisa que não consiga expressar: mais de uma entrada de imagem, ou uma
forma de entrada com uma dimensão simbólica. O MNN nesta versão exige uma forma
NCHW totalmente fixa, e o `batch` fica embutido no artefato em vez de ser negociado
na hora do carregamento.

O sidecar não é burocracia opcional. `weights/LibreYOLO9t.mnn.json` registra os
nomes de entrada e de saída, a forma de entrada fixa, o batch, os nomes das classes,
a versão do MNN usada e o backend para o qual o artefato foi construído, e o runtime
valida cada um desses campos no carregamento.

No Windows, o MNN 3.6.1 às vezes conclui a conversão e depois encerra durante o
desmonte do processo com uma violação de acesso ou um status de fail-fast. O
exportador reconhece esses códigos de saída específicos e trata a conversão como
bem-sucedida quando o arquivo de saída está presente.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` despacha pelo sufixo `.mnn` e devolve o mesmo objeto `Results` que o
checkpoint. O carregamento é estrito por design: o sidecar tem que declarar
`format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, um tamanho, uma
tarefa de detecção, uma forma NCHW fixa e positiva que concorde com o tamanho de
imagem registrado, e nomes de classes cobrindo todo índice de 0 a `nc - 1`. Qualquer
divergência levanta um erro em vez de adivinhar.

Predizer com um `imgsz` diferente daquele para o qual o artefato foi construído
também levanta erro, e `device` é ignorado com um aviso, porque as exportações MNN
rodam na CPU aqui.

O segundo snippet é o caminho do runtime puro. O pré-processamento, a decodificação,
o NMS e o reescalonamento de coordenadas passam a ser por sua conta ali, e os nomes
de entrada e de saída vêm do sidecar porque o carregador de módulos do MNN os quer
explicitamente.

## Restrições

Somente detecção. O backend recusa qualquer outra tarefa no carregamento, e o lado
da exportação acompanha: fora das combinações registradas, a verificação prévia
levanta um erro com "MNN v1 has no implemented runtime contract for this family and
task."

FP32, CPU, forma fixa. `dynamic=True` levanta `ValueError`, e `half=True` e
`int8=True` são rejeitados durante a validação.

As famílias de detecção validadas são YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM e YOLO-NAS, cada uma coberta por
conversão, recarga de um artefato novo, execução em CPU pelo MNN, checagem de
metadados e paridade de detecções pós-NMS batendo com o modelo PyTorch. O DEIMv2
converte, recarrega, executa e preserva as detecções pós-NMS, mas sua rota
intermediária em ONNX tem paridade de scores em nível de query incompleta, então ele
é registrado como disponível e não como validado.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
