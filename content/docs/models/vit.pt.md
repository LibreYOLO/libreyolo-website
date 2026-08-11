---
title: ViT
families:
  - vit
seo_title: 'ViT: rode os clássicos classificadores Vision Transformer no LibreYOLO'
description: >-
  Faça predição, validação e exportação de classificadores ViT com o LibreYOLO.
  Pesos AugReg sob licença Apache-2.0; o fine-tuning ainda não é suportado.
lead: >-
  O clássico Vision Transformer: um transformer puro aplicado a patches de
  imagem de tamanho fixo, com um class token aprendido e sem convoluções. O
  LibreYOLO traz quatro tamanhos pré-treinados com AugReg para classificação de
  imagens.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - classificação de imagens python
  - classificador transformer pré-treinado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreViTti-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data é um diretório raiz com os splits train/ e val/ em pastas por
        # classe (layout ImageFolder), não um YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Instalação

O ViT não precisa de nenhum extra opcional. Tudo o que ele importa está na
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
confianças correspondentes. O pré-processamento redimensiona e faz um recorte
central para uma entrada fixa de 224px, seguindo a receita de avaliação AugReg
do timm: interpolação bicúbica com fração de recorte de 0.9. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos, de tiny a large, que compartilham um mesmo grafo fixo de
224px com patch-16 e diferem na largura do embedding e na profundidade do
transformer. O LibreYOLO distribui esta família apenas para inferência:
predição, validação top-1/top-5 no estilo ImageNet e exportação são suportadas,
e a receita de fine-tuning do AugReg não está implementada.

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

## Citação

<citation-block />
