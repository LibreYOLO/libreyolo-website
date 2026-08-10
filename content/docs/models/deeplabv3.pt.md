---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: predição e exportação de segmentação semântica com ASPP'
description: >-
  Use o DeepLabv3 no LibreYOLO para segmentação semântica. Instale, faça
  predições, valide e exporte os checkpoints ResNet e MobileNetV3 do
  torchvision.
lead: >-
  Uma rede de segmentação semântica que agrega características em várias taxas
  de dilatação em paralelo (atrous spatial pyramid pooling) antes de classificar
  cada pixel. O LibreYOLO a inclui apenas para segmentação semântica.
keywords:
  - DeepLabv3
  - segmentação semântica
  - ASPP
  - atrous spatial pyramid pooling
  - deeplabv3 pytorch
  - segmentação semântica python
  - classificar cada pixel da imagem
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe presentes na imagem, ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Instalação

O DeepLabv3 não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.
O sufixo `-sem` no nome do arquivo é obrigatório para esta família.

<code-tabs name="predict" />

A segmentação semântica devolve um id de classe por pixel, não boxes, então
`result.semantic_mask` traz um array `(H, W)` em `.data` e a lista de ids de
classe presentes na imagem em `.classes`. `conf`, `iou` e `max_det` são aceitos
por paridade de API, mas não têm efeito: o modelo atribui uma classe a cada
pixel por argmax, sem limiar de confiança nem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Três backbones: ResNet-50 dilatada, ResNet-101 dilatada e MobileNetV3-Large
dilatada. Este é o DeepLabv3, não o DeepLabv3+, então não há estágio de decoder
nem refinamento por CRF, acompanhando a implementação do torchvision em vez do
código de referência do próprio artigo.

O LibreYOLO não treina o DeepLabv3: `train()` levanta `NotImplementedError`
para esta família, que o [nível de suporte](/docs/models) acima marca como
somente inferência. Os três checkpoints publicados são os próprios pesos do
torchvision treinados em COCO com os rótulos do VOC, convertidos para o loader
do LibreYOLO.

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
