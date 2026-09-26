---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet no LibreYOLO: predição, treinamento e exportação'
description: >-
  Rode o PicoDet no LibreYOLO para detecção de objetos em dispositivos móveis.
  Instale, faça predições, treine, valide e exporte sob Apache-2.0.
lead: >-
  O PicoDet é um detector de estágio único feito para CPUs móveis e de borda: um
  backbone ESNet, um neck CSP-PAN e uma cabeça compartilhada com Generalized
  Focal Loss. O LibreYOLO o suporta para detecção.
keywords:
  - PicoDet
  - PP-PicoDet
  - detecção de objetos python
  - detecção de objetos no celular
  - detector leve para edge
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # Vale a pena definir o imgsz: a CLI assume 640 por padrão, enquanto o

        # checkpoint s é nativo em 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Instalação

O PicoDet não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` define o limiar de
confiança e `iou` o limiar do NMS. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Três tamanhos, cada um com sua própria resolução de entrada fixa: `s` o menor e
`l` o maior. A resolução cresce junto com o tamanho, então checkpoints maiores
também são mais caros de rodar por imagem, além de carregarem mais parâmetros.

<benchmark-table task="detect" />

<va-embed />

## Treinamento

<code-tabs name="train" />

Os componentes da loss e o assigner seguem a receita do upstream: VFL, DFL,
GIoU e SimOTA, com ponderação por qualidade de classificação e alvos VFL de IoU
dinâmico. A inferência é bit a bit equivalente à do upstream no mesmo
checkpoint.

O que não foi verificado, conforme a própria docstring de `train()`:
convergência em dataset completo, comportamento multi-GPU e qualquer data
augmentation além do flip horizontal. O checkpoint `s`, na sua resolução nativa
de 320, também não passou de forma confiável no piso de acurácia do LibreYOLO
na fixture de 30 imagens e duas classes com que a biblioteca testa fine-tunes
pequenos. Esse tamanho se encaixa melhor na escala do COCO completo.

`train()` também aceita um argumento `pretrained`, mas o valor nunca é lido
dentro do método: o treinamento sempre continua a partir dos pesos com que o
modelo foi construído, então `pretrained=False` não reinicializa a rede. Deixe
`imgsz` sem definir no Python e ele assume a resolução nativa do checkpoint
carregado: 320 para o `s`, 416 para o `m` e 640 para o `l`. A CLI sempre envia
um `imgsz`, com 640 por padrão, então defina esse valor lá para bater com o
checkpoint.

Sem outros ajustes, o treinador roda 300 épocas com SGD a `lr0=0.01`, momentum
0.9, weight decay 4e-5 e um warmup de 1 época em um schedule cosseno. O flip
horizontal é o único data augmentation aplicado.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado volta a carregar por `LibreYOLO()` pela extensão do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O port do LibreYOLO segue o Bo396543018/Picodet_Pytorch, uma reimplementação em
PyTorch do PP-PicoDet original do PaddleDetection, com o mmcv removido e cada
ativação replicada exatamente, de modo que checkpoints do PaddlePaddle
convertidos pelo pipeline do Bo carregam sem nenhum desvio numérico. Ambas as
fontes carregam os mesmos termos Apache-2.0 dos autores do artigo.

</provenance-box>

## Citação

<citation-block />
