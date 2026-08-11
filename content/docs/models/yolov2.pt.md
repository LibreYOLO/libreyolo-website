---
title: YOLOv2
families:
  - yolo2
seo_title: 'YOLOv2 no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o YOLOv2 (YOLO9000) no LibreYOLO: uma família de museu congelada e
  somente de inferência. Faça predições, valide e exporte, sob licença de
  domínio público.
lead: >-
  O YOLOv2, publicado também como YOLO9000, é o detector Darknet-19 que trouxe
  as anchor boxes e uma camada de passthrough para a linha YOLO. O LibreYOLO o
  carrega como uma peça de museu congelada e somente de inferência.
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - detecção de objetos python
  - anchor boxes
  - família de museu
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## Instalação

O YOLOv2 não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Esta família é somente de inferência: `train()` levanta `NotImplementedError`,
então esta página não tem seção de Treinamento. Predição, validação e
exportação são todas suportadas. Os pesos são baixados do Hugging Face no
primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` filtra pelo limiar
de confiança e `iou` pelo limiar de NMS, aplicados sobre as predições baseadas
em âncoras da cabeça `region`. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
valida.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pela extensão do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. Rodar o grafo em um runtime pelado, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
