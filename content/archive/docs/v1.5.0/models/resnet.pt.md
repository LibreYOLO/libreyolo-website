---
title: ResNet
families:
  - resnet
seo_title: 'ResNet: treine, valide e exporte sob Apache-2.0'
description: >-
  Use a ResNet no LibreYOLO para classificação de imagens. Instale, preveja,
  faça fine-tuning, valide e exporte LibreResNet18/34/50/101.
lead: >-
  ResNet é um classificador de imagens construído com blocos residuais, skip
  connections que deixam uma rede acrescentar muito mais camadas sem a perda de
  acurácia que pilhas convolucionais simples e profundas sofrem. O LibreYOLO a
  suporta para uma tarefa: classificação.
keywords:
  - ResNet
  - ResNet50
  - classificação de imagens python
  - aprendizado residual
  - redes residuais profundas
  - classificador ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
        libreyolo export model=LibreResNet50-cls.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: e2f46c73716af1b7
---

## Instalação

A ResNet não precisa de nenhum extra opcional. Tudo o que ela importa está na
instalação base.

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

Quatro profundidades, todas treinadas e avaliadas da mesma forma, então
escolher uma é uma troca direta entre número de parâmetros e acurácia. A tarefa
é fixa: todos os tamanhos cobrem apenas classificação. O nome do arquivo de
pesos termina em `-cls.pt` em todos os tamanhos, e é esse sufixo que a factory
lê para rotear para esta família; nenhum argumento `task=` é necessário.

## Treinamento

O fine-tuning parte do backbone de ImageNet publicado e reconstrói
automaticamente a camada final do classificador para o número de classes do
dataset alvo.

<code-tabs name="train" />

Sem mexer em nada, o trainer roda 100 épocas com `lr0=1e-3` e AdamW, um batch de
64 e early stopping após 50 épocas sem melhora. `data` aceita a raiz de um
dataset (`train/` e `val/`, uma pasta por classe), um nome curto conhecido como
`imagenette160`, ou uma URL de `.zip`. `lora=True` não é suportado aqui; passar
esse argumento gera um erro, já que o LoRA no LibreYOLO mira componentes de
transformer com camadas `nn.Linear`, e a ResNet não tem nenhuma.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário de chaves `metrics/`. Para classificação, são a
acurácia top-1 e top-5 sobre o split de validação.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` a partir do sufixo
do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint
e devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita e os extras que alguns deles acrescentam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
