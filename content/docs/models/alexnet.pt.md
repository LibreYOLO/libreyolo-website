---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: rode o clássico classificador da ImageNet no LibreYOLO'
description: >-
  Faça predição, validação e exportação do AlexNet com o LibreYOLO. Pesos do
  torchvision sob licença BSD-3-Clause; o fine-tuning ainda não é suportado.
lead: >-
  AlexNet é a rede convolucional que venceu a ILSVRC 2012 e ajudou a iniciar a
  era do deep learning em visão computacional. O LibreYOLO traz a revisão
  posterior da arquitetura, de torre única, para classificação de imagens.
keywords:
  - AlexNet
  - ImageNet
  - rede neural convolucional
  - classificação de imagens python
  - classificador de imagens pré-treinado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data é um diretório raiz com os splits train/ e val/ em pastas por
        # classe (layout ImageFolder), não um YAML de dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Instalação

O AlexNet não precisa de nenhum extra opcional. Tudo o que ele importa está na
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
confianças correspondentes. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Um único tamanho. O grafo distribuído é a revisão posterior de torre única
publicada pelo torchvision, com 64 filtros na primeira camada e sem local
response normalization, e não a arquitetura original de duas GPUs de 2012. O
LibreYOLO distribui esta família apenas para inferência: predição, validação
top-1/top-5 no estilo ImageNet e exportação são suportadas, e o fine-tuning
não está implementado.

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
