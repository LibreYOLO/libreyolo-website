---
title: DeiT
families: [deit]
seo_title: "Classificador de imagens DeiT: predição, validação e exportação"
description: "Rode classificadores de imagens DeiT no LibreYOLO: uma família de museu congelada e somente para inferência, nos tamanhos tiny, small e base, sob Apache-2.0."
lead: "DeiT (Data-efficient image Transformer) é um classificador Vision Transformer puro treinado apenas com o ImageNet-1k, sem dados extras de pré-treinamento. O LibreYOLO traz os tamanhos tiny, small e base com patch-16 como uma peça de museu congelada e somente para inferência."
keywords: [DeiT, Vision Transformer, ViT, "classificação de imagens python", ImageNet, "classificador de imagens pré-treinado", "família de museu"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeiTb-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Instalação

O DeiT não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Esta família é somente para inferência: `train()` levanta `NotImplementedError`,
então esta página não tem seção de treinamento. Predição, validação e exportação
são todas suportadas. Os pesos são baixados do Hugging Face no primeiro uso e
ficam em cache local. O sufixo `-cls` no nome do arquivo é obrigatório e
seleciona a tarefa de classificação.

<code-tabs name="predict" />

O objeto `Results` devolvido carrega um tensor `probs` em vez de `boxes`;
`top1` e `top5` indexam as 1.000 classes do ImageNet-1k e `top1conf` é o score
softmax da predição principal. Cada tamanho tem uma resolução de entrada fixa,
vinda do seu positional embedding: o pré-processamento redimensiona e faz um
recorte central até ela, e passar um `imgsz` diferente levanta um erro em vez de
reamostrar em silêncio. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Validação

`val()` devolve um dicionário com a acurácia top-1 e top-5, medida sobre um
dataset organizado na estrutura de pastas convencional `train/<class>/` e
`val/<class>/`.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` a partir da
extensão do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e devolve o mesmo `Results`. Rodar o grafo em um runtime pelado, sem
o LibreYOLO instalado, também é suportado, mas aí o pré-processamento e o
pós-processamento ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
