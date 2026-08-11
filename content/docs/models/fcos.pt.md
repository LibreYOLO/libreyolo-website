---
title: FCOS
families:
  - fcos
seo_title: 'FCOS no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o FCOS no LibreYOLO para detecção de objetos sem âncoras. Instale, faça
  predições, valide e exporte o port do torchvision sob BSD-3-Clause,
  ResNet-50/FPN.
lead: >-
  O FCOS detecta objetos por pixel em vez de depender de um conjunto de caixas
  âncora predefinidas, prevendo uma caixa e um score de centerness em cada
  posição do mapa de características. O LibreYOLO porta a implementação do
  torchvision para detecção.
keywords:
  - FCOS
  - detector anchor-free
  - detecção de objetos python
  - detector one-stage
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Instalação

O FCOS não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Chamar o modelo sem
argumentos de limiar aplica os defaults publicados pelo próprio FCOS,
`conf=0.2`, `iou=0.6` e `max_det=100`; passe qualquer um dos três para
sobrescrevê-los. O FCOS mantém uma etapa final de NMS sobre suas predições por
pixel. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Variantes

Um tamanho só: ResNet-50 com uma pirâmide de características, a única variante
que esta família reconhece.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

O FCOS exporta para ONNX, TorchScript e OpenVINO. O FCOS preserva a proporção
da imagem de origem antes de o grafo rodar, então o LibreYOLO força
`dynamic=True` nos caminhos de ONNX e OpenVINO, independentemente do que for
passado, para manter o grafo válido com entradas preenchidas por padding. Um
arquivo `.onnx` exportado carrega de volta por `LibreYOLO()` pela extensão do
arquivo e retorna o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
