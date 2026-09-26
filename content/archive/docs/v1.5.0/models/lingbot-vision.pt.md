---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: segmentação semântica no LibreYOLO'
description: >-
  Use o LingBot-Vision no LibreYOLO para segmentação semântica sobre um backbone
  ViT Apache-2.0. Instale, faça predições, treine, valide e exporte, tamanhos
  s/b/l.
lead: >-
  O LingBot-Vision é uma família de backbones vision transformer
  auto-supervisionados, treinados com modelagem mascarada centrada em bordas
  para percepção espacial densa, lançada pela Robbyant. O LibreYOLO combina o
  backbone com uma cabeça densa e o suporta para uma tarefa, segmentação
  semântica.
keywords:
  - LingBot-Vision
  - segmentação semântica
  - vision transformer
  - pré-treinamento auto-supervisionado
  - Robbyant
  - segmentação semântica python
  - predição densa por pixel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone congelado por padrão, seguindo o protocolo de avaliação
        # original: só a cabeça densa 1x1 treina.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Fine-tune completo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Instalação

O LingBot-Vision não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

`result.semantic_mask` traz o mapa denso de classes: `.data` é um tensor
`(H, W)` de ids de classe no tamanho original da imagem, e `.classes` lista os
ids de classe realmente presentes. `result.boxes` é `None`, já que não há
detecções por instância. `conf` e `iou` são aceitos por paridade de API, mas não
mudam a saída, já que o modelo devolve uma classe por pixel em vez de detecções
a filtrar. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Variantes

Três tamanhos publicados, s, b e l, destilados de um modelo teacher ViT-g/16 de
1.1B parâmetros. O próprio teacher, tamanho `g`, carrega e faz fine-tuning no
LibreYOLO, mas o LibreYOLO não hospeda um checkpoint `g` próprio.

<checkpoint-table />

## Treinamento

`train()` faz fine-tuning de um checkpoint publicado. A receita padrão é o
linear probe do relatório original: o backbone ViT fica congelado e só a cabeça
densa 1x1 treina, do mesmo jeito que os pesos hospedados pelo LibreYOLO acima
foram produzidos. Passe `freeze_backbone=False` para fazer fine-tuning da rede
inteira, e espere ter que baixar o `lr0` na mesma medida.

<code-tabs name="train" />

Veja [treinamento](/docs/train) para datasets, augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário de chaves `metrics/`: mIoU e acurácia por pixel,
medidos sobre qualquer dataset no formato em que você treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

A versão original documenta seu ViT como construído sobre a arquitetura
DINOv2/DINOv3 publicada pela Meta AI. A Robbyant distribui sua implementação sob
Apache-2.0, e este port para o LibreYOLO foi feito apenas a partir do
repositório da Robbyant, nunca do código DINOv2 ou DINOv3 da Meta.

</provenance-box>

## Citação

<citation-block />
