---
title: FCN
families:
  - fcn
seo_title: 'FCN: predição e exportação de uma FCN ResNet sob BSD-3-Clause'
description: >-
  Use o FCN no LibreYOLO para segmentação semântica. Instale, faça predições,
  valide e exporte os checkpoints FCN com ResNet dilatada do torchvision.
lead: >-
  Um classificador denso por pixel que troca as camadas totalmente conectadas de
  um detector por convoluções, entregando um mapa de classes em resolução plena
  em vez de boxes. O LibreYOLO o inclui apenas para segmentação semântica.
keywords:
  - FCN
  - fully convolutional network
  - segmentação semântica
  - predição densa
  - ResNet
  - fcn pytorch
  - segmentação semântica python
  - classificar cada pixel da imagem
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe presentes na imagem, ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Instalação

O FCN não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

A segmentação semântica devolve um id de classe por pixel, não boxes, então
`result.semantic_mask` traz um array `(H, W)` em `.data` e a lista de ids de
classe presentes na imagem em `.classes`. `conf`, `iou` e `max_det` são aceitos
por paridade de API, mas não têm efeito: o modelo atribui uma classe a cada
pixel por argmax, sem limiar de confiança nem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Duas profundidades de ResNet, ambas com entrada fixa de 520 px. O grafo de
inferência da biblioteca é o FCN com ResNet dilatada do torchvision, não a rede
FCN-8s original do artigo, baseada em VGG e com conexões de salto.

O LibreYOLO não treina o FCN: `train()` levanta `NotImplementedError` para esta
família, que o [nível de suporte](/docs/models) acima marca como somente
inferência. Os dois checkpoints publicados são os próprios pesos do torchvision
treinados em COCO, convertidos para o loader do LibreYOLO.

## Validação

`val()` devolve `metrics/mIoU` e `metrics/pixel_accuracy`, medidos sobre
qualquer dataset no formato em que você treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
