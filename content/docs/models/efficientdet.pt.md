---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: detecção de objetos no LibreYOLO'
description: >-
  Rode o EfficientDet D0-D4 no LibreYOLO: detectores BiFPN para predição,
  validação e exportação para ONNX, TensorRT e OpenVINO sob licença Apache-2.0.
lead: >-
  O EfficientDet combina um backbone EfficientNet com uma rede piramidal de
  características bidirecional repetida (BiFPN) e escala profundidade, largura e
  resolução em conjunto ao longo de cinco tamanhos. O LibreYOLO o inclui como
  detector somente de inferência.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - detecção de objetos python
  - exportar efficientdet onnx
  - compound scaling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Instalação

O EfficientDet não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. O EfficientDet decodifica
candidatos baseados em âncoras e depois roda non-maximum suppression por
classe, então `conf`, `iou` e `max_det` têm efeito real aqui. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Cinco tamanhos, do D0 ao D4. Cada passo adiante combina um backbone
EfficientNet maior com uma BiFPN mais profunda e mais larga e uma cabeça de
predição mais profunda, então a contagem de parâmetros e o custo computacional
crescem juntos, seguindo a regra de compound scaling do artigo.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado carrega de volta por `LibreYOLO()` pela extensão do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

Os checkpoints D0-D4 do LibreYOLO são convertidos por meio do projeto
rwightman/efficientdet-pytorch, sob licença Apache-2.0, que por sua vez espelha
os pesos oficiais treinados em TensorFlow do google/automl sem alterar tensores
aprendidos. Nenhum código do projeto zylo117/Yet-Another-EfficientDet-Pytorch,
licenciado sob LGPL, foi consultado ou usado.

</provenance-box>
