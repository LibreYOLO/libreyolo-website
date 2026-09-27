---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: predição e exportação de segmentação em tempo real sob MIT'
description: >-
  Use o PIDNet no LibreYOLO para segmentação semântica em tempo real. Instale,
  faça predições, valide e exporte os checkpoints s/m/l do Cityscapes sob
  licença MIT.
lead: >-
  Uma rede de segmentação semântica de três ramos que acrescenta um ramo
  dedicado a bordas a um projeto inspirado no controle
  proporcional-integral-derivativo, voltada para inferência em tempo real. O
  LibreYOLO a inclui apenas para segmentação semântica.
keywords:
  - PIDNet
  - segmentação semântica em tempo real
  - segmentação de bordas
  - Cityscapes
  - pidnet pytorch
  - segmentação semântica python
  - segmentação em tempo real
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe presentes na imagem, ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Instalação

O PIDNet não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

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

Três tamanhos, todos com entrada fixa de 1024 px. Os checkpoints publicados são
conversões dos pesos oficiais do PIDNet para o Cityscapes, com 19 classes.

O LibreYOLO não treina o PIDNet: `train()` levanta `NotImplementedError` para
esta família, que o [nível de suporte](/docs/models) acima marca como somente
inferência.

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

## Citação

<citation-block />
