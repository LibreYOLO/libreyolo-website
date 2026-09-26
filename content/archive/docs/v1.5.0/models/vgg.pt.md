---
title: VGG
families:
  - vgg
seo_title: 'VGG: rode classificadores de imagens VGG-16/19 no LibreYOLO'
description: >-
  Faça predição, validação e exportação de classificadores VGG com o LibreYOLO.
  Pesos do torchvision sob licença BSD-3-Clause; o fine-tuning ainda não é
  suportado.
lead: >-
  VGG é um classificador de imagens convolucional construído com pilhas
  uniformes de pequenas convoluções 3x3 em vez de filtros maiores. O LibreYOLO
  traz os tamanhos de 16 e 19 camadas, simples e com batch normalization, para
  classificação de imagens.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - rede neural convolucional
  - classificação de imagens python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data é um diretório raiz com os splits train/ e val/ em pastas por
        # classe (layout ImageFolder), não um YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Instalação

O VGG não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

Um classificador retorna `result.probs` em vez de `result.boxes`: `top1`
e `top5` dão os índices das classes, e `top1conf` e `top5conf` dão as
confianças correspondentes. A predição roda com entrada fixa de 224px e levanta
um erro se você passar um `imgsz` diferente. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos: 16 e 19 camadas convolucionais, cada um com uma variante
simples e uma com batch normalization. Os pesos distribuídos vêm do treinamento
posterior do torchvision na ImageNet, feito do zero, e não de conversões da
versão Caffe original de 2014 publicada pelo grupo de Oxford. O LibreYOLO
distribui esta família apenas para inferência: predição, validação top-1/top-5
no estilo ImageNet e exportação são suportadas, e o fine-tuning não está
implementado.

## Validação

`val()` roda sobre um split no formato ImageFolder (um diretório com as
subpastas `train/` e `val/`, uma pasta por classe) e retorna a acurácia top-1 e
top-5.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é recarregado por `LibreYOLO()` pelo sufixo do arquivo,
então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e retorna
o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que todo
formato aceita e os extras que alguns deles adicionam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>
