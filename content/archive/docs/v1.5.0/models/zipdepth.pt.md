---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: profundidade monocular leve no LibreYOLO'
description: >-
  Use o ZipDepth no LibreYOLO para estimativa de profundidade monocular leve.
  Instale, faça predições, valide e exporte dois checkpoints sob licença MIT.
lead: >-
  O ZipDepth é uma CNN compacta e reparametrizável destilada do Depth Anything
  V2 Large que prediz um mapa denso de profundidade inversa relativa. O
  LibreYOLO oferece suporte a ele na tarefa de profundidade: predição e
  validação zero-shot, sem caminho de treinamento.
keywords:
  - ZipDepth
  - estimativa de profundidade monocular
  - modelo de profundidade para edge
  - profundidade relativa
  - mapa de profundidade
  - depth estimation python
  - CNN reparametrizável
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Mesmo encoder, com uma cabeça de upsampling sem unfold para
        compiladores

        # sem suporte a gather/unfold. A saída é visualmente equivalente à do
        checkpoint b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Instalação

O ZipDepth não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

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
está definido apenas para normais de superfície e bordas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Dois checkpoints, ambos com a mesma capacidade de encoder, diferindo apenas na
cabeça de upsampling treinada. O `b` usa upsampling convexo e roda em GPU ou
CPU. O `bnpu` troca por um decoder sem unfold, para compiladores de NPU e edge
que não têm suporte a gather/unfold; sua saída é documentada como visualmente
equivalente à do `b`. Escolha `bnpu` quando o alvo da exportação for um runtime
restrito, e `b` caso contrário.

Os dois checkpoints foram destilados de pseudo-labels do Depth Anything V2
Large, então esta família é o nível compacto e voltado a edge da tarefa de
profundidade do LibreYOLO, ao lado dos encoders maiores do Depth Anything V2.

O treinamento não é oferecido para esta família. `LibreZipDepth.train()` levanta
`NotImplementedError` incondicionalmente: a receita upstream destila
pseudo-labels sobre um grande conjunto de imagens que não é reproduzível como
uma execução de treinamento do LibreYOLO. Treine upstream em
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) e converta o
resultado com `weights/convert_zipdepth_weights.py`.

## Validação

`val()` roda o validador de profundidade compartilhado: ele alinha cada predição
ao seu ground truth com uma escala e um deslocamento de mínimos quadrados por
imagem, e depois reporta as métricas padrão de profundidade relativa zero-shot,
AbsRel, RMSE e os três limiares delta.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação segue um contrato denso de resolução fixa: a imagem de origem é
redimensionada com distorção para o canvas exportado, e o mapa de profundidade
devolvido é redimensionado de volta ao canvas original em seguida. Um artefato
exportado é carregado de volta por `LibreYOLO()` pelo sufixo do arquivo, então
um arquivo `.onnx` ou `.ncnn` se comporta como um checkpoint e devolve o mesmo
`Results`, com `depth_map` no lugar dos boxes.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
