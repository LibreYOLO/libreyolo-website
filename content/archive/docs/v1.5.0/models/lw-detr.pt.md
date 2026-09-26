---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: predição e exportação sob Apache-2.0'
description: >-
  Rode o LW-DETR no LibreYOLO para detecção de objetos em tempo real. Instale,
  faça predição, validação e exportação de cinco tamanhos baseados em ViT, todos
  sob licença Apache-2.0.
lead: >-
  Um transformer de detecção com ViT puro que a Baidu posicionou como
  alternativa em tempo real aos detectores YOLO. O LibreYOLO traz cinco tamanhos
  para detecção, apenas inferência.
keywords:
  - LW-DETR
  - detection transformer
  - detecção de objetos em tempo real
  - plain ViT
  - DETR
  - detecção de objetos python
  - transformer para detecção de objetos
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() retorna um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Instalação

O LW-DETR não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det`
filtram a seleção de queries; `iou` é aceito por paridade de API, mas não tem
efeito, porque o decoder é um preditor de conjuntos sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

No LibreYOLO, o LW-DETR é apenas para inferência. O upstream treina com a
supervisão um-para-muitos do Group-DETR em vários grupos de queries e uma loss
de classificação ciente de IoU; essa receita não está conectada aqui, então
`train()` levanta `NotImplementedError`.

## Variantes

Cinco tamanhos, todos compartilhando o encoder de ViT puro, o projetor
multiescala e o decoder do DETR deformável, e todos rodando na mesma resolução
de entrada. Os dois menores compartilham a largura do encoder e se diferenciam
pela profundidade em blocos; os dois seguintes compartilham um encoder mais
largo e se diferenciam por quantos níveis do projetor alimentam o decoder; o
maior sobe para o encoder mais largo de todos.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é recarregado por `LibreYOLO()` pelo sufixo do arquivo,
então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e retorna
o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que todo
formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
