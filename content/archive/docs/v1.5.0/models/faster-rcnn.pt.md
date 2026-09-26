---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o Faster R-CNN no LibreYOLO para detecção de objetos em quatro backbones.
  Instale, faça predição, validação e exportação do port BSD-3-Clause do
  torchvision.
lead: >-
  O Faster R-CNN detecta objetos com uma region proposal network alimentando um
  classificador de dois estágios, a arquitetura que tornou as propostas de
  região parte da mesma rede treinada em vez de uma etapa separada. O LibreYOLO
  porta a implementação do torchvision para detecção.
keywords:
  - Faster R-CNN
  - detecção de objetos python
  - region proposal network
  - detector de dois estágios
  - detecção de objetos pytorch
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Instalação

O Faster R-CNN não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `iou` definem os
limiares de confiança e de NMS; o Faster R-CNN mantém a etapa de NMS do
upstream, diferente de um detector baseado em queries. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos, cada um uma configuração diferente do torchvision em vez de uma
versão escalada da mesma: `n` é o MobileNetV3-Large com entrada de 320 px, `s` é
o mesmo backbone a 800 px, `m` é o ResNet-50 com uma pirâmide de características,
e `l` é a revisão v2, com uma cabeça de propostas de região mais profunda e uma
cabeça de box de quatro convoluções no lugar da do `m`. `n` e `s` trocam acurácia
por um backbone mais leve.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

O Faster R-CNN exporta só para ONNX, com tamanho de batch 1. O grafo exportado
mantém dentro dele a etapa de redimensionamento do upstream, então o LibreYOLO
força `dynamic=True` independentemente do que for passado, para manter o grafo
válido para fontes que não são quadradas. Um arquivo `.onnx` exportado é
recarregado por `LibreYOLO()` pelo sufixo do arquivo e retorna o mesmo
`Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
