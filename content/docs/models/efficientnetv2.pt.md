---
title: EfficientNetV2
families:
  - efficientnetv2
seo_title: 'EfficientNetV2: treine, valide e exporte sob Apache-2.0'
description: >-
  Use EfficientNetV2 no LibreYOLO para classificação de imagens. Instale,
  preveja, faça fine-tuning, valide e exporte LibreEfficientNetV2 b0 a b3.
lead: >-
  EfficientNetV2 é um classificador de imagens cuja profundidade, largura e
  escolha de blocos por estágio foram encontradas por neural architecture
  search, otimizando ao mesmo tempo a acurácia e a velocidade de treinamento, em
  vez de só a acurácia. O LibreYOLO o suporta para uma tarefa: classificação.
keywords:
  - EfficientNetV2
  - EfficientNetV2-b0
  - classificação de imagens python
  - neural architecture search
  - MBConv
  - classificador ImageNet
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientNetV2b0-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientNetV2b0-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=onnx

        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=tensorrt
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreEfficientNetV2b0-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 23b7d651fb4eae89
---

## Instalação

O EfficientNetV2 não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` devolvido é o mesmo que todas as famílias devolvem, então
trocar por outro modelo é uma mudança de uma linha. Um classificador não carrega
boxes nem máscaras: `result.probs` guarda a predição da imagem inteira, com
`top1`, `top5`, `top1conf` e `top5conf`. `conf`, `iou` e `max_det` são aceitos
por paridade de API, mas não têm efeito, já que não há nada para limiarizar ou
suprimir em um único vetor de probabilidades. Veja [predição](/docs/predict)
para fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos, de b0 a b3, cada um avaliado na sua própria resolução e
proporção de crop, em vez de compartilharem um único tamanho de entrada em toda
a família. Escolher um tamanho é uma troca direta entre número de parâmetros e
acurácia. A tarefa é fixa: todos os tamanhos cobrem apenas classificação. O nome
do arquivo de pesos termina em `-cls.pt` em todos os tamanhos, e é esse sufixo
que a factory lê para rotear para esta família; nenhum argumento `task=` é
necessário.

## Treinamento

O fine-tuning parte do backbone de ImageNet publicado e reconstrói
automaticamente a camada final do classificador para o número de classes do
dataset alvo. `imgsz` assume por padrão a resolução de avaliação do próprio
tamanho, a menos que seja definido explicitamente.

<code-tabs name="train" />

Sem mexer em nada, o trainer roda 100 épocas com `lr0=1e-3` e AdamW, um batch de
64 e early stopping após 50 épocas sem melhora. `data` aceita a raiz de um
dataset (`train/` e `val/`, uma pasta por classe), um nome curto conhecido como
`imagenette160`, ou uma URL de `.zip`. `lora=True` não é suportado aqui; passar
esse argumento levanta um erro, já que o LoRA no LibreYOLO mira componentes de
transformer com camadas `nn.Linear`, e os blocos MBConv desta família não têm
nenhuma.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

`cls_pw=0` desativa a ponderação da loss; valores até 1 usam pesos de frequência inversa normalizados para média 1. Já `class_weights=True` usa frequências inversas normalizadas por amostra e não pode ser combinado com `cls_pw>0`. Essas configurações devem coincidir ao retomar. Veja [classificação](/docs/tasks/image-classification).

## Validação

`val()` devolve um dicionário de chaves `metrics/`. Para classificação, são a
acurácia top-1 e top-5 sobre o split de validação.

<code-tabs name="val" />

Validação e calibração INT8 usam a transformação de avaliação da família. Os metadados de exportação registram `norm_mean`, `norm_std` e `resize_mode`; artefatos antigos usam os valores da família como fallback. Os pré-processadores de calibração retornam o array CHW e a razão exigidos.

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
