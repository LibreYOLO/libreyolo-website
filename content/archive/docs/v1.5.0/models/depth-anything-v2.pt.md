---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: prediga e valide profundidade monocular'
description: >-
  Use o Depth Anything V2 no LibreYOLO para estimativa de profundidade
  monocular. Instale, faça predições e valide; o Small vem sob Apache-2.0, Base
  e Large são CC-BY-NC-4.0.
lead: >-
  O Depth Anything V2 é um encoder DINOv2 combinado com um decoder DPT que
  prediz um mapa denso de profundidade inversa relativa a partir de uma única
  imagem. O LibreYOLO oferece suporte a ele na tarefa de profundidade: predição
  e validação zero-shot, sem caminho de treinamento.
keywords:
  - Depth Anything V2
  - estimativa de profundidade monocular
  - mapa de profundidade python
  - profundidade a partir de uma imagem
  - depth estimation python
  - DPT
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Ler o mapa de profundidade
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: denso (H, W), maior = mais perto

        raw = depth.data                # tensor, sem unidade métrica nem escala
        entre imagens

        normalized = depth.normalized() # reescalado para [0, 1] para
        visualização
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Instalação

O Depth Anything V2 não precisa de nenhum extra opcional. Tudo o que ele importa está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

`result.depth_map` carrega um mapa denso de profundidade inversa relativa:
valores maiores significam mais perto da câmera, e os valores não têm unidade
métrica nem escala entre imagens. `save=True` grava em disco uma visualização
desse mapa com mapa de cores; `Results.plot()` não cobre esta família, já que
está definido apenas para normais de superfície e bordas. A resolução de
entrada precisa ser divisível por 14, a grade de patches do DINOv2 sobre a qual
a cabeça DPT é construída; o LibreYOLO confere isso antes de rodar e levanta um
erro se não for. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

Quatro tamanhos de encoder, s/b/l/g, correspondendo a ViT-S/B/L/G. A tabela de
checkpoints abaixo lista apenas s, b e l; nenhum checkpoint Giant foi
publicado. Os quatro compartilham a mesma resolução de entrada, então escolher
um tamanho troca capacidade do encoder, não tamanho de imagem. O licenciamento
também pesa: o checkpoint Small é Apache-2.0, enquanto Base e Large são
CC-BY-NC-4.0, veja Licenciamento abaixo.

Treinamento e fine-tuning não são oferecidos para esta família.
`LibreDepthAnythingV2.train()` levanta `NotImplementedError`
incondicionalmente; em vez disso, converta um checkpoint upstream compatível
com `weights/convert_depth_anything_v2_weights.py`.

## Validação

`val()` roda o validador de profundidade compartilhado: ele alinha cada
predição ao seu ground truth com uma escala e um deslocamento de mínimos
quadrados por imagem, e depois reporta as métricas padrão de profundidade
relativa zero-shot, AbsRel, RMSE e os três limiares delta.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`, com `depth_map` no lugar dos boxes.
[Exportação](/docs/export) lista os argumentos que todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
