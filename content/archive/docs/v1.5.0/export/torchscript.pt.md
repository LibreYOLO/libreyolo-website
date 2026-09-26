---
title: TorchScript
seo_title: Exportar para TorchScript a partir do LibreYOLO
description: >-
  Exporte um modelo LibreYOLO para TorchScript: um arquivo .torchscript traçado
  com os metadados do LibreYOLO dentro, carregável pelo Python ou pelo libtorch.
lead: >-
  TorchScript é o formato de grafo serializado do próprio PyTorch. O LibreYOLO
  traça o modelo com torch.jit.trace e salva o resultado junto com um arquivo
  extra libreyolo_metadata.json, de modo que o arquivo carrega a família, a
  tarefa, os nomes das classes e o tamanho de entrada.
keywords:
  - exportar yolo torchscript
  - torch.jit.trace
  - torch.jit.load
  - deploy libtorch
  - metadados torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Escreve
    value: Um arquivo .torchscript com um arquivo extra libreyolo_metadata.json
  - label: Extra
    value: Nenhum. O TorchScript vem junto com o PyTorch.
  - label: Recarrega com
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Formas
    value: Fixas. O grafo é traçado em uma única forma de entrada.
  - label: Precisão
    value: 'FP32, FP16 (half=True). Sem INT8.'
verification: >-
  Lido de libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py e libreyolo/backends/torchscript.py no branch dev.
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int, ou (altura, largura)
            batch=1,
            half=False,       # pesos e ativações em FP16
            device=None,      # None faz o trace na CPU para este formato
            output_path=None, # None escreve weights/<stem>.torchscript
        )

        # dynamic é aceito, mas o arquivo é sempre um trace de forma fixa,
        # e os metadados embutidos registram dynamic=False de qualquer jeito.
  run:
    - label: Através do LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch puro
      language: python
      code: >
        import json


        import torch


        extra_files = {"libreyolo_metadata.json": ""}

        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )

        module.eval()


        metadata = json.loads(extra_files["libreyolo_metadata.json"])

        print(metadata["model_family"], metadata["task"], metadata["imgsz"])


        # O pré-processamento e o pós-processamento são sua responsabilidade
        aqui.

        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Instalação

<code-tabs name="install" />

O TorchScript não precisa de nada além da instalação base, porque o `torch.jit` vem
junto com o PyTorch. É o único alvo de exportação sem dependência opcional e sem
conversor externo, o que o torna uma primeira conferida útil quando um toolchain
mais longo falha.

## Exportação

<code-tabs name="export" />

O tracing roda na CPU a menos que um dispositivo seja indicado, e o arquivo é
escrito em `weights/` sob o stem do checkpoint quando `output_path` é omitido.

A verificação de retrace que o `torch.jit.trace` normalmente faz está desligada.
Vários wrappers de exportação guardam em cache âncoras dependentes da forma durante
o primeiro forward pass, então um segundo trace observa um caminho Python diferente
mesmo com o grafo de forma fixa registrado estando correto. Os testes de paridade
validam o módulo salvo diretamente, em vez disso.

Os metadados não ficam em um sidecar. O `torch.jit.save` guarda o
`libreyolo_metadata.json` dentro do arquivo, e o `torch.jit.load` o devolve através
de `_extra_files`.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` roteia pelo sufixo `.torchscript` e devolve o mesmo objeto `Results`
que o checkpoint de origem. Com `device="auto"` o módulo é mapeado para CUDA quando
disponível, depois MPS, depois CPU.

O segundo snippet é o caminho para quem lê isto sem ter o LibreYOLO instalado, e
para deploy em C++ através do libtorch, onde o mesmo arquivo carrega com
`torch::jit::load`. Pré-processamento, decodificação, NMS e reescalonamento de
coordenadas passam a ser sua responsabilidade ali. O arquivo extra de metadados
continua legível, e é o único lugar onde os nomes das classes existem.

## Restrições

O grafo é um trace em uma única forma de entrada. `dynamic=True` é aceito por
simetria de interface, mas não muda nada, e os metadados embutidos reportam
`dynamic=False` para que um backend nunca assuma um eixo que não pode usar. Exporte
um segundo arquivo para uma segunda resolução.

`half=True` converte o modelo e a entrada do trace para FP16. Não existe caminho
INT8: `int8=True` levanta `NotImplementedError` durante a validação.

`imgsz` retangular funciona para as famílias YOLO9, HRNet, NAFNet e Real-ESRGAN, e
é recusado para famílias com contrato quadrado fixo.

Cinco combinações são recusadas antes do tracing. Segmentação com YOLO9, porque no
LibreYOLO o YOLO9 é somente detecção. Segmentação com RTMDet-Ins, cujo decode de
máscara com kernel dinâmico não tem contrato de runtime exportado. Detecção com
SSD, Faster R-CNN e RetinaNet, cujos grafos de comprimento variável ou de âncoras
dinâmicas têm evidência de paridade apenas através do contrato do ONNX Runtime.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
