---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o detector YOLOv1 original no LibreYOLO: uma família de museu congelada e
  somente de inferência. Faça predições, valide e exporte, sob uma licença de
  domínio público.
lead: >-
  O YOLOv1 é o detector original de 2016 que deu nome à família YOLO: uma única
  rede convolucional com uma cabeça totalmente conectada prediz todas as caixas
  e pontuações de classe em uma só passada, sem anchor boxes. O LibreYOLO o
  mantém como uma peça de museu congelada e somente de inferência.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - detecção de objetos python
  - Pascal VOC
  - yolo original
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Instalação

O YOLOv1 não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Esta família é somente de inferência: `train()` levanta `NotImplementedError`,
então esta página não tem seção de treinamento. Predição, validação e
exportação são todas suportadas. Os pesos são baixados do Hugging Face no
primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Duas coisas são
específicas desta família. O checkpoint publicado é treinado no Pascal VOC
(2007+2012), não no COCO, então `box.cls` indexa as 20 categorias do VOC
(aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor) em vez das 80 do COCO. E a cabeça de detecção totalmente conectada
aceita uma imagem por vez, então uma lista de fontes é percorrida em laço em vez
de rodar como um batch de verdade. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra um dataset no mesmo espaço de rótulos estilo
VOC em que o checkpoint foi treinado.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado carrega de volta por `LibreYOLO()` pela extensão do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
