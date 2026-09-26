---
title: Paddle
seo_title: Exportar para PaddlePaddle a partir do LibreYOLO
description: >-
  Converta um detector LibreYOLO em um modelo de inferência do PaddlePaddle
  através do X2Paddle: o toolchain fixado, os grafos estáticos FP32 com batch 1
  e a inferência em CPU.
lead: >-
  Modelos de inferência do PaddlePaddle são um grafo model.pdmodel ao lado de um
  arquivo de pesos model.pdiparams. O LibreYOLO exporta um grafo ONNX estático
  com opset 15, converte com o X2Paddle e empacota o resultado com um
  metadata.yaml para que ele seja carregado pela mesma fábrica que qualquer
  outro runtime.
keywords:
  - exportar yolo paddle
  - inferência paddlepaddle
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Escreve
    value: 'Um diretório com model.pdmodel, model.pdiparams e metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Formas
    value: 'Estáticas, batch 1, opset 15. As três são impostas.'
  - label: Precisão
    value: 'Somente FP32, somente CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 ou anterior, conferidos de
      forma exata
verification: >-
  Lido de libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md e
  pyproject.toml no branch dev.
snippets:
  install:
    - label: Instalação
      language: bash
      code: >
        # Python 3.10 a 3.12. WSL2 com Ubuntu 22.04 é o caminho validado no
        Windows.

        pip install "libreyolo[paddle]"
    - label: Confirmar as versões fixadas
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve o diretório weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; a tela quadrada desta família
            batch=1,          # qualquer outro valor levanta ValueError
            dynamic=False,    # True levanta ValueError
            simplify=True,    # False levanta ValueError
            opset=15,         # qualquer outro valor levanta ValueError
            output_path=None, # None escreve weights/<stem>_paddle
        )
  run:
    - label: Através do LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: O backend diretamente
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # O que LibreYOLO() constrói para um diretório Paddle. O mesmo objeto
        # Results, sem o roteamento da fábrica no meio.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle puro
      language: python
      code: >
        import numpy as np

        import paddle.inference as paddle_infer

        import yaml


        directory = "weights/LibreYOLO9t_paddle"

        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )

        config.disable_gpu()

        config.disable_mkldnn()

        config.switch_ir_optim(False)


        predictor = paddle_infer.create_predictor(config)

        handle = predictor.get_input_handle(predictor.get_input_names()[0])

        handle.reshape([1, 3, 640, 640])

        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))

        predictor.run()

        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # O pré-processamento e o pós-processamento ficam por sua conta neste
        caminho.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Instalação

<code-tabs name="install" />

O extra fixa exatamente a stack que o trabalho de paridade mediu: PaddlePaddle
2.6.2, X2Paddle 1.6.0 e ONNX 1.17 ou anterior. Essas versões fixadas são
conferidas na hora de exportar, não só na instalação, e uma versão diferente
levanta um `ImportError` nomeando a esperada. Versões mais novas do Paddle
rejeitam partes do código estático que o X2Paddle 1.6.0 gera, então falhar cedo é
melhor do que produzir um artefato que ninguém validou.

## Exportação

<code-tabs name="export" />

Quatro argumentos são fixos, não apenas padrões. `dynamic` precisa ser `False`,
`batch` precisa ser 1, `simplify` precisa ser `True` para um grafo de conversão
totalmente estático, e `opset` precisa ser 15, que é o teto que o X2Paddle 1.6.0
aceita. Passar qualquer outra coisa levanta um erro antes do tracing.

Uma única normalização roda sobre o grafo intermediário. O ONNX define como um a
dilatação omitida de um MaxPool, o PyTorch escreve o atributo explícito só de uns
e o X2Paddle 1.6.0 o rejeita, então o exportador remove esse padrão redundante e
deixa a operação especificada sem alteração.

O artefato é um diretório: `model.pdmodel`, `model.pdiparams` e `metadata.yaml`.
O Python que o X2Paddle gera durante a conversão não faz parte dele.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` reconhece qualquer diretório que contenha ao mesmo tempo
`model.pdmodel` e `model.pdiparams`, lê o `metadata.yaml` e devolve o mesmo objeto
`Results` que o checkpoint. Um dispositivo diferente de `auto` ou `cpu` levanta um
erro: este backend é somente CPU.

O que a fábrica constrói é o `PaddleBackend`, exportado de `libreyolo` e
importável como `libreyolo.backends.paddle.PaddleBackend`. Construa você mesmo
quando quiser o backend sem o roteamento por sufixo da fábrica, por exemplo para
passar `task=` explicitamente em um diretório cujo `metadata.yaml` não foi você
que escreveu. O `predict()` dele aceita as mesmas fontes e devolve os mesmos
resultados.

O snippet do runtime puro espelha o que o backend configura, e as três opções
desativadas são deliberadas. O pipeline de fusão em CPU do Paddle 2.6 pode
quebrar enquanto otimiza os grandes grafos de gather e scatter emitidos para a
atenção deformável, então o grafo estático portátil e sem fusão é aquele contra o
qual a paridade foi medida. O pré-processamento, a decodificação, o NMS e o
reescalonamento de coordenadas ficam por sua conta nesse caminho.

## Restrições

Sem formas dinâmicas, sem FP16, sem INT8, sem NMS embutido, sem runtime em GPU.

As combinações validadas são detecção YOLO9, detecção YOLO9-E2E e YOLO9-P2,
detecção, pose e segmentação EC, detecção RT-DETRv4, D-FINE, DEIM e DEIMv2, e
detecção e pose YOLO-NAS. Cada uma é coberta por conversão, uma recarga no runtime
de CPU, paridade de saídas em cru e resultados públicos reproduzidos.

Bloqueadas, com o motivo registrado por combinação:

| Combinação | Motivo |
|---|---|
| RF-DETR, todas as tarefas | Precisa de ONNX opset 17 e GridSample; o X2Paddle 1.6.0 aceita opset 15 ou inferior e não tem mapper de GridSample |
| Detecção RT-DETR e RT-DETRv2 | Os grafos treinados precisam de GridSample no opset 16 ou superior |
| Segmentação D-FINE | Converte e recarrega, mas o erro RMS relativo dos logits de máscara é de 3.52% e o IoU mínimo de máscaras pareadas é 0.582 |
| Segmentação YOLO9 | O YOLO9 é somente detecção no LibreYOLO |
| Segmentação RTMDet-Ins | A decodificação de máscaras com kernel dinâmico não tem contrato de runtime exportado |

Qualquer coisa que não esteja listada como validada ou bloqueada é recusada com a
observação de que não foi validada através do caminho de conversão de ONNX para
Paddle.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
