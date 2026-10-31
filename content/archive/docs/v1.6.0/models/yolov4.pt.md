---
title: YOLOv4
families:
  - yolo4
seo_title: 'YOLOv4: rodar, validar e exportar no LibreYOLO'
description: >-
  Rode o YOLOv4 no LibreYOLO: uma família de museu congelada, somente de
  inferência, com backbone CSPDarknet-53. Faça predições, valide e exporte, sob
  licença de domínio público.
lead: >-
  O YOLOv4 combina um backbone CSPDarknet-53, um bloco SPP e um neck PANet com
  ativações Mish. O LibreYOLO o inclui como uma peça de museu congelada, somente
  de inferência, nos tamanhos tiny e base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - detecção de objetos python
  - ativação Mish
  - família de museu
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO4b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Instalação

O YOLOv4 não precisa de nenhum extra além do pacote base.

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
trocar por outro detector é uma mudança de uma linha. `conf` filtra o limiar de
confiança e `iou` o limiar de NMS, aplicados depois do escalonamento de centro
`scale_x_y` próprio de cada cabeça. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
validar.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado carrega de volta pelo `LibreYOLO()` a partir da extensão
do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e retorna o mesmo `Results`. Rodar o grafo em um runtime puro, sem o
LibreYOLO instalado, também é suportado, mas aí o pré-processamento e o
pós-processamento ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
