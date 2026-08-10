---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: detecção de objetos no LibreYOLO'
description: >-
  Rode o CenterNet (Objects as Points) no LibreYOLO com os backbones ResDCN-18 e
  DLA-34. Faça predições, valide e exporte para ONNX sob licença MIT. Sem
  caminho de treinamento.
lead: >-
  O CenterNet modela um objeto como o ponto central do seu bounding box e
  regride todas as demais propriedades a partir de um pico do mapa de calor,
  então não precisa de âncoras nem de uma etapa de non-maximum-suppression. O
  LibreYOLO o inclui como detector somente de inferência.
keywords:
  - CenterNet
  - Objects as Points
  - detecção de objetos python
  - detector anchor-free
  - detecção por keypoints
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # A exportação para ONNX precisa de opset 16 ou superior: a etapa de
        # upsampling por convolução deformável baixa para GridSample, que o
        # opset 16 introduziu.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A factory roteia pela extensão do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo
        # objeto Results.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Instalação

O CenterNet não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det`
filtram os picos do mapa de calor já ordenados; `iou` é aceito por paridade de
API, mas não tem efeito, porque o decode top-k de picos do CenterNet não
precisa de nenhuma etapa de supressão por IoU de caixas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Dois backbones. `resdcn18` combina um tronco ResNet-18 com upsampling por
convolução deformável; `dla34` combina um tronco DLA-34 com upsampling por
agregação profunda iterativa. Ambos alimentam as mesmas três cabeças densas
(mapa de calor, largura/altura, offset) e a mesma tela de entrada.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação para ONNX exige opset 16 ou superior: a etapa de upsampling por
convolução deformável de ambos os backbones baixa para o operador `GridSample`
do ONNX, introduzido no opset 16. Pedir um opset anterior gera um erro antes de
o traçado começar.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O grafo do ResDCN-18 também credita o human-pose-estimation.pytorch da
Microsoft, licenciado sob MIT, e o grafo do DLA-34 credita a implementação DLA
de Fisher Yu, sob BSD-3-Clause. O LibreYOLO não incorpora a extensão DCNv2
original que o projeto upstream usava; a execução nativa roda o
`deform_conv2d` do torchvision, sob BSD-3-Clause, em vez dela, e a
implementação portátil, apenas para exportação, foi escrita separadamente para
o LibreYOLO.

</provenance-box>

## Citação

<citation-block />
