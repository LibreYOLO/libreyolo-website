---
title: ExecuTorch
seo_title: "Exportar para ExecuTorch a partir do LibreYOLO"
description: "Exporte um modelo LibreYOLO para um programa .pte do ExecuTorch com delegação para XNNPACK: forma fixa, batch 1, FP32 e o sidecar de metadados de que ele precisa."
lead: "O ExecuTorch roda programas do PyTorch em alvos de borda (edge). O LibreYOLO captura o modelo com torch.export em modo estrito, faz o lowering para XNNPACK e grava o programa .pte junto com um sidecar de metadados JSON como uma unidade só."
keywords:
  - exportar yolo executorch
  - programa .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inferência pytorch no edge
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="executorch")'
    mono: true
  - label: Escreve
    value: "Um programa .pte mais um sidecar de metadados .pte.json"
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.pte")'
    mono: true
  - label: Formas
    value: "Fixas. dynamic=True e batch != 1 são recusados."
  - label: Precisão
    value: "Somente FP32. half=True e int8=True são recusados."
  - label: Delegado
    value: "XNNPACK, CPU. delegate='xnnpack' é o único valor aceito."
verification: "Lido de libreyolo/export/executorch.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/executorch.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        # Fora de libreyolo[all] de propósito: o ExecuTorch restringe com qual
        # versão do Torch ele pode ser combinado.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.pte e weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, ou (altura, largura)
            batch=1,               # qualquer outro valor levanta ValueError
            dynamic=False,         # True levanta ValueError
            delegate="xnnpack",    # o único valor aceito
            device="cpu",          # qualquer outro device levanta ValueError
            output_path=None,      # None escreve weights/<stem>.pte
        )
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Runtime puro do ExecuTorch
      language: python
      code: |
        import json
        from pathlib import Path

        import torch
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        print(runtime.backend_registry.is_available("XnnpackBackend"))

        program = runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())
        method = program.load_method("forward")

        # O pré-processamento e o pós-processamento ficam por sua conta neste caminho.
        outputs = method.execute((torch.zeros(1, 3, 640, 640),))
        print([tensor.shape for tensor in outputs])

        meta = json.load(open("weights/LibreYOLO9t.pte.json"))
        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

Este extra fica de propósito fora de `libreyolo[all]`, porque o ExecuTorch fixa com
qual versão do Torch ele funciona e instalá-lo arrastaria o ambiente inteiro para
esse par. Instale esse extra em um ambiente que você esteja disposto a restringir.

No Windows, a etapa de lowering chama o executável `flatc` que vem junto com o
ExecuTorch. Se ele não estiver no `PATH`, a exportação levanta um `RuntimeError`
dizendo isso, e a solução é rodar a partir de um Developer PowerShell do Visual Studio 2022.

## Exportação

<code-tabs name="export" />

A captura é `torch.export.export(..., strict=True)`, que é uma captura de grafo de
verdade, com guards, e não um trace gravado. Leituras de escalares no host e controle
de fluxo dependente de dados são recusados em vez de ficarem embutidos em silêncio,
então várias famílias falham aqui apesar de traçarem sem problema em outros formatos;
os motivos ficam registrados por combinação na matriz de suporte.

O lowering roda `to_edge_transform_and_lower` com o partitioner do XNNPACK. Se o
resultado não tiver nenhuma partição delegada, a exportação levanta um erro em vez de
rotular como XNNPACK um programa que só usa kernels portáteis.

O programa e o sidecar são gravados juntos. Os dois passam pela área de staging, os dois
são trocados de uma vez, e uma falha reverte para o que estava lá antes, de modo que um
par incompleto nunca chega ao disco.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` despacha pelo sufixo `.pte` e devolve o mesmo objeto `Results` que o
checkpoint. O sidecar é obrigatório no carregamento: sem `<program>.pte.json` o backend
levanta `FileNotFoundError`, porque o programa não carrega nomes de classes, tarefa nem
tamanho de entrada próprios. O backend também confere se o runtime instalado fornece
`XnnpackBackend` antes de carregar, e lê o programa a partir de bytes em vez de mapear o
arquivo, o que evita segurar um lock de arquivo do Windows por toda a vida do backend.

O segundo snippet é o caminho do runtime puro. O pré-processamento, a decodificação, o
NMS e o reescalonamento de coordenadas ficam por sua conta ali.

## Restrições

Batch 1, forma fixa, FP32, CPU. `batch != 1` e `dynamic=True` levantam `ValueError`
antes que a exportação altere qualquer coisa, `half=True` e `int8=True` são recusados
durante a validação, e um device diferente de CPU é rejeitado.

`delegate` aceita `"xnnpack"` e nada mais nesta versão.

As exportações de classificação carregam duas chaves extras de metadados, `crop_pct` e
`interpolation`, para que o runtime consiga reproduzir a política de resize e center-crop
da família.

As entradas bloqueadas nomeiam a falha concreta em vez de uma categoria. A detecção e a
segmentação com D-FINE esbarram em uma leitura de `ContextVar` não suportada na deformable
attention sob captura estrita, e forçar o caminho manual de grid-sample serializa, mas
depois falha em tempo de execução por uma ordem de dimensões inválida no tensor delegado.
DEIM e DEIMv2 capturam, fazem o lowering e serializam, e então falham durante a execução.
A segmentação semântica com EoMT falha em uma expressão simbólica dependente de dados no
caminho da máscara. O matting com BiRefNet captura em 1024 por 1024, mas não tem variante
out para `torchvision::deform_conv2d`. A restauração com SwinIR recarrega e então falha em
`aten::alias_copy.out` por ordens de dimensões incompatíveis.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
