---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: classifique imagens com o LibreSwin do LibreYOLO'
description: >-
  Faça predição, validação e exportação de classificadores Swin Transformer com
  o LibreYOLO. Pesos MIT; fine-tuning ainda não é suportado.
lead: >-
  Swin Transformer V1: um vision transformer hierárquico que calcula atenção
  dentro de janelas locais deslocadas em vez de sobre a imagem inteira. O
  LibreYOLO traz quatro tamanhos para classificação de imagens.
keywords:
  - Swin Transformer
  - vision transformer hierárquico
  - atenção por janelas deslocadas
  - classificação de imagens python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data é a raiz de um diretório com splits train/ e val/ em pastas por
        # classe (layout ImageFolder), não um YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Instalação

O Swin não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

Um classificador devolve `result.probs` em vez de `result.boxes`: `top1` e
`top5` dão os índices das classes, `top1conf` e `top5conf` dão as confianças
correspondentes. Todos os tamanhos são fixos em uma entrada de 224px, porque o
estágio final de atenção é construído para essa resolução; predição, validação e
exportação levantam erro se você passar um `imgsz` diferente. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos, de tiny a large, construídos a partir da mesma torre de janelas
deslocadas e diferindo na largura do embedding e na profundidade dos estágios. O
large é pré-treinado no ImageNet-22k e passa por fine-tuning no ImageNet-1k; os
outros três são treinados direto no ImageNet-1k. O LibreYOLO traz esta família
apenas para inferência: predição, validação top-1/top-5 no estilo ImageNet e
exportação são suportadas, e a receita de treinamento ImageNet do projeto
original não está implementada.

## Validação

`val()` roda sobre um split no estilo ImageFolder (um diretório com subpastas
`train/` e `val/`, uma pasta por classe) e devolve a acurácia top-1 e top-5.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` a partir da
extensão do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e devolve o mesmo `Results`. [Exportação](/docs/export) lista os
argumentos que todo formato aceita e os extras que alguns deles acrescentam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
