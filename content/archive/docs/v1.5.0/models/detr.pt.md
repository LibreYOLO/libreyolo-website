---
title: DETR
families:
  - detr
seo_title: 'DETR: preveja e exporte sob Apache-2.0'
description: >-
  Use DETR, o transformer de detecção original, no LibreYOLO. Instale, preveja,
  valide e exporte quatro tamanhos baseados em ResNet, todos licenciados sob
  Apache-2.0.
lead: >-
  O DETR é o transformer de detecção original: ele prevê um conjunto fixo de
  objetos com um decoder transformer casado por matching húngaro, em vez de
  âncoras ou de uma grade densa. O LibreYOLO traz quatro tamanhos para detecção,
  apenas para inferência.
keywords:
  - DETR
  - detection transformer
  - detecção de objetos python
  - DETR ResNet-50
  - transformer para detecção de objetos
  - exportar DETR onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() devolve um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Instalação

O DETR não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` devolvido é o mesmo que todas as famílias devolvem, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram
a seleção de queries; `iou` é aceito por paridade de API, mas não tem efeito,
porque o decoder é um preditor de conjunto, sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

No LibreYOLO, o DETR é apenas para inferência. O projeto original treina por 500
épocas com matching húngaro; essa receita não está implementada aqui, então
`train()` levanta `NotImplementedError`.

## Variantes

Quatro checkpoints combinam duas profundidades de backbone, ResNet-50 ou
ResNet-101, com um estágio C5 dilatado opcional: as variantes DC5 mantêm o
último estágio do backbone em resolução plena em vez de reduzir a escala mais
uma vez, então o decoder lê um mapa de características (feature map) mais fino a
partir do mesmo tamanho de entrada. As quatro compartilham 100 object queries
aprendidas e um encoder-decoder transformer de seis camadas, e todas rodam na
mesma resolução de entrada.

## Validação

`val()` devolve um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos sobre qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` a partir da
extensão do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e devolve o mesmo `Results`. [Exportação](/docs/export) lista os
argumentos que todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
