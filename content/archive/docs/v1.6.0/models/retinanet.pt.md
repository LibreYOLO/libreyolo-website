---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o RetinaNet no LibreYOLO para detecção de objetos one-stage com focal
  loss. Instale, faça predições, valide e exporte o port do torchvision sob
  BSD-3-Clause.
lead: >-
  O RetinaNet é um detector one-stage treinado com focal loss, que reduz o peso
  dos negativos fáceis para que uma grade densa de âncoras não precise mais de
  uma etapa separada de propostas para continuar acurada. O LibreYOLO porta a
  implementação do torchvision para detecção.
keywords:
  - RetinaNet
  - focal loss
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

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Instalação

O RetinaNet não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `iou` definem
os limiares de confiança e de NMS; o RetinaNet mantém a etapa de NMS do
upstream sobre a grade densa de âncoras. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Dois tamanhos, ambos ResNet-50 com uma pirâmide de características: `r50` é a
cabeça original, e `r50v2` a substitui por uma cabeça com GroupNorm e um bloco
P6 mais largo alimentado pelo último estágio do backbone em vez da saída da FPN.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

O RetinaNet exporta apenas para ONNX, com tamanho de batch 1. O RetinaNet
redimensiona para uma entrada de tamanho variável que preserva a proporção,
então o LibreYOLO força `dynamic=True` independentemente do que for passado,
para manter o grafo válido para fontes de formatos diferentes. Um arquivo
`.onnx` exportado carrega de volta por `LibreYOLO()` pela extensão do arquivo
e retorna o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
