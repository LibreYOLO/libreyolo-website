---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: prediga segmentação semântica, de instâncias e panóptica'
description: >-
  Use o EoMT no LibreYOLO para segmentação semântica, de instâncias e panóptica
  sobre um vision transformer DINOv2 simples, sem decoder. Licença MIT.
lead: >-
  Uma rede de segmentação construída sobre um vision transformer simples, sem
  decoder de pixels dedicado: queries aprendidas extras, adicionadas ao próprio
  encoder, predizem as máscaras. O LibreYOLO a suporta para segmentação
  semântica, de instâncias e panóptica.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - segmentação panóptica
  - segmentação de instâncias
  - segmentação semântica
  - segmentação panóptica python
  - eomt pytorch
last_verified: 1.5.0
snippets:
  predict:
    - label: Semântica
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe presentes na imagem, ordenados
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -seg no nome do arquivo seleciona a tarefa de instâncias,
        # então nenhum argumento de tarefa é necessário aqui.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panóptica
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ids de segmento
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Semântica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # boxes
    - label: Panóptica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Instalação

O EoMT não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.
O sufixo de tarefa no nome do arquivo (`-sem`, `-seg`, `-panoptic`) seleciona a
tarefa, e `LibreYOLO()` a infere a partir desse nome de arquivo, então nenhum
argumento `task=` é necessário.

<code-tabs name="predict" />

A segmentação semântica preenche `result.semantic_mask`, um array `(H, W)` de
ids de classe em `.data`. A segmentação de instâncias preenche `result.boxes` e
`result.masks`, no mesmo formato que todas as outras famílias de segmentação
devolvem. A segmentação panóptica preenche `result.panoptic`: um mapa `(H, W)`
de ids de segmento em `.data`, mais `.segments_info`, uma lista de dicts
`{"id", "category_id"}`, um por segmento. `conf` filtra a seleção de queries;
`iou` não tem efeito na tarefa semântica, já que ela faz argmax por pixel, sem
etapa de NMS. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

Três tamanhos de encoder, s/b/l, todos baseados em DINOv2. O checkpoint
semântico é treinado em ADE20K a 512 px; os checkpoints de instâncias e
panóptico são treinados em COCO a 640 px, com um segundo checkpoint de
instâncias treinado a 1280 px. O upstream publica pesos de segmentação de
instâncias com DINOv2 apenas no tamanho l; s e b saem só para as tarefas
semântica e panóptica. Variantes do EoMT baseadas em DINOv3 existem no
upstream, mas não são distribuídas aqui, porque dependem de pesos DINOv3
restritos e não comerciais.

O LibreYOLO não treina o EoMT: `train()` levanta `NotImplementedError` para esta
família, que o [nível de suporte](/docs/models) acima marca como somente
inferência.

## Validação

`val()` despacha por tarefa. A semântica devolve `metrics/mIoU` e
`metrics/pixel_accuracy`. A segmentação de instâncias devolve as mesmas chaves
de mAP de máscara e de box que as outras famílias de segmentação. A panóptica
devolve o Panoptic Quality como `metrics/PQ`, dividido em `metrics/SQ`
(qualidade de segmentação) e `metrics/RQ` (qualidade de reconhecimento), mais
`metrics/PQ_things` e `metrics/PQ_stuff`.

<code-tabs name="val" />

## Exportação

<export-matrix />

Hoje só a tarefa semântica exporta: a segmentação de instâncias e a panóptica
chamam `export()` e recebem `NotImplementedError`, porque a saída de máscaras
por query ainda não tem um contrato de exportação de runtime. Um artefato
semântico exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
