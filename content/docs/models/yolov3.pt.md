---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o YOLOv3 no LibreYOLO: uma família de museu congelada e somente de
  inferência, nos tamanhos tiny, base e SPP. Faça predições, valide e exporte,
  sob uma licença de domínio público.
lead: >-
  O YOLOv3 é o detector com Darknet-53 que trouxe predição em múltiplas escalas
  e classificadores logísticos independentes para a linha YOLO. O LibreYOLO o
  mantém como uma peça de museu congelada e somente de inferência, nos tamanhos
  tiny, base e SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - detecção de objetos python
  - detecção multiescala
  - família de museu
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tamanho SPP
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # A variante SPP adiciona um bloco de spatial pyramid pooling antes

        # das cabeças de detecção e roda no seu próprio tamanho nativo de
        entrada.

        model = LibreYOLO("LibreYOLO3spp.pt")

        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Instalação

O YOLOv3 não precisa de nada além do pacote base.

```bash
pip install libreyolo
```

## Predição

Esta família é somente de inferência: `train()` lança `NotImplementedError`, então
esta página não tem seção de treinamento. Predição, validação e exportação são
todas suportadas. Os pesos são baixados do Hugging Face no primeiro uso e ficam
em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` filtra o limiar de
confiança e `iou` o limiar de NMS, aplicados por escala antes de as caixas das
três cabeças serem juntadas. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
valida.

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
