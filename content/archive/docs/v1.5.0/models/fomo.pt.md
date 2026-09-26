---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: localização por pontos, treinamento e exportação no LibreYOLO'
description: >-
  Rode o FOMO (Faster Objects, More Objects) no LibreYOLO: um detector minúsculo
  de localização por pontos para contar muitos objetos pequenos. Instale, faça
  predições, treine e exporte.
lead: >-
  O FOMO é um localizador de pontos baseado em grade: cada célula de uma grade
  de baixa resolução é classificada como fundo ou centro de objeto, sem
  regressão de bounding box. O LibreYOLO o suporta na tarefa de pontos.
keywords:
  - FOMO
  - Faster Objects More Objects
  - localização por pontos
  - detecção de centroides
  - detecção de objetos pequenos
  - edge AI
  - detecção em microcontrolador
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Os pesos do LibreFOMO não são baixados automaticamente (veja
        Checkpoints abaixo).

        # Aponte isto para um checkpoint que você já baixou localmente.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz precisa ser passado: a CLI usa 640 por padrão, e o checkpoint

        # s aceita apenas os seus 96 nativos.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Instalação

O FOMO não precisa de nada além do pacote base.

```bash
pip install libreyolo
```

## Predição

Diferente de todas as outras famílias deste site, os pesos do LibreFOMO não são
baixados automaticamente: `LibreYOLO("LibreFOMOs-point.pt")` procura esse arquivo
no disco e levanta um `ValueError` citando o nome dele em vez de buscá-lo no
Hugging Face. Baixe um checkpoint da [organização LibreYOLO](https://huggingface.co/LibreYOLO)
primeiro e carregue-o pelo caminho local, ou treine o seu (veja Treinamento abaixo).

<code-tabs name="predict" />

O resultado traz um payload `points` no lugar de `boxes`: cada linha é
`x, y, class, confidence`, disponível como `result.points.data` ou pelos
acessores `.xy`, `.xyn`, `.cls` e `.conf`. Não há limiar de `iou` para definir,
porque não existem boxes a suprimir; `predict(..., nms_radius=1)` controla
quantas células de grade duas detecções precisam ter entre si para ambas
sobreviverem, e o nome do arquivo precisa carregar o sufixo de tarefa `-point`
do FOMO para o loader reconhecê-lo. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Três tamanhos, `s`, `m` e `l`, usam backbones no estilo MobileNetV2
progressivamente mais largos em resoluções de entrada fixas e correspondentemente
maiores, cada um atrás de uma única cabeça de classificação 1x1. Esta família não
tem tabela de benchmark aqui; o tamanho do arquivo de checkpoint na tabela abaixo
é o sinal mais claro por tamanho publicado até agora.

## Treinamento

<code-tabs name="train" />

`imgsz` não é uma escolha livre: ele assume por padrão a resolução nativa do
checkpoint carregado, e passar um valor diferente levanta um `ValueError` citando
o tamanho esperado. Esses tamanhos são 96 para o `s`, 192 para o `m` e 224 para o
`l`. A CLI usa 640 como padrão de `imgsz`, então um comando `libreyolo train`
precisa defini-lo explicitamente para bater com o checkpoint.

Se você não mexer em mais nada, o treinador roda 40 épocas com batch 32 usando
Adam a `lr0=3e-4`, sem weight decay, e com a classe de primeiro plano pesando
100x mais que o fundo na loss de cross-entropy por célula, já que quase toda
célula da grade é fundo em uma cena típica. EMA e precisão mista ficam ambos
desligados por padrão, e nenhuma das augmentations geométricas ou de cor usadas
em outras partes do LibreYOLO é aplicada: mosaic, mixup, jitter de HSV, flip,
rotação, translação e shear estão todos zerados.

Esse é o caminho com que os checkpoints publicados do LibreFOMO foram treinados,
do zero na COCO.

Veja [treinamento](/docs/train) para datasets e loggers.

## Validação

`val()` despacha para um validador em nível de grade feito para esta família. Ao
lado das chaves `metrics/precision`, `metrics/recall` e `metrics/mAP@` de
correspondência por pontos, compartilhadas com outras tarefas de pontos, ele
varre limiares de confiança e valores de `nms_radius` e publica a combinação de
melhor F1 em `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall` e
`metrics/grid_mean_distance`, além do limiar e do raio que a produziram em
`decode/threshold` e `decode/nms_radius`.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado carrega de volta pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todo arquivo de pesos publicado desta família. Nenhum deles baixa
automaticamente: pegue o arquivo que você quer na página do Hugging Face
vinculada e passe o caminho local dele para `LibreYOLO()`.

<checkpoint-table />

## Licenciamento

<provenance-box>

Não há repositório de código upstream do FOMO para linkar: a Edge Impulse
descreve a técnica em um post de blog e na documentação do seu produto, mas não
liberou o código de treinamento ou de inferência do FOMO. A arquitetura e o
treinamento aqui são a implementação própria do LibreYOLO dessa descrição
publicada, e os checkpoints publicados do LibreFOMO são treinados do zero na
COCO, então tanto o código quanto esses pesos são MIT, do próprio LibreYOLO. O
nome FOMO e a técnica que ele descreve continuam sendo da Edge Impulse.

</provenance-box>
