---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): detecção de objetos no LibreYOLO'
description: >-
  Rode o SSD300 no LibreYOLO: um detector single-shot baseado em VGG16 para
  predição, validação e exportação para ONNX sob licença BSD-3-Clause. Sem
  caminho de treinamento.
lead: >-
  O SSD (Single Shot MultiBox Detector) prevê cada caixa e cada score de classe
  a partir de uma grade densa de caixas padrão em uma única passagem para
  frente, sem uma etapa separada de proposta de regiões. O LibreYOLO inclui o
  checkpoint SSD300 baseado em VGG16 como um detector somente de inferência.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - detecção de objetos python
  - VGG16
  - detector baseado em âncoras
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgsz foi deixado de fora aqui de propósito: o SSD300 é traçado na
        # tela nativa do seu checkpoint, e qualquer outro valor gera um erro
        # antes de a exportação começar.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A factory roteia pela extensão do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo
        # objeto Results.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Instalação

O SSD não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. O SSD decodifica sua grade
de caixas padrão com scores por classe e depois roda a non-maximum suppression,
então `conf`, `iou` e `max_det` têm efeito real aqui, ao contrário dos
detectores baseados em queries desta biblioteca. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

O SSD vem com um único checkpoint: a rede SSD300 baseada em VGG16, na sua tela
nativa fixa. Não há escolha de tamanho ou escala nesta família; predição,
validação e exportação usam esse mesmo grafo.

O arquivo de pesos é `LibreSSD300.pt`, o prefixo da família seguido da sua única
chave de tamanho, `"300"`. A classe por trás dele é `LibreSSD`, então a
construção direta é `LibreSSD(size="300")`, e não uma classe com o nome do
arquivo.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

O SSD exporta apenas para ONNX; todos os outros formatos estão bloqueados para
esta família no momento. A exportação sempre usa a tela nativa do checkpoint, e
o grafo expõe a cabeça bruta empacotada do SSD em vez de uma saída de
non-maximum-suppression fundida, então `nms=True` não é aceito na hora da
exportação. Os próprios backends do LibreYOLO rodam a etapa de decode e
supressão depois de carregar o grafo de volta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O código do SSD300 do LibreYOLO não foi portado da versão em Caffe publicada
pelos próprios autores do artigo; ele deriva da implementação SSD300 do
torchvision, sob BSD-3-Clause, e é esse o repositório vinculado acima como fonte
upstream. Os pesos VGG16 do backbone remontam ainda mais atrás ao VGGNet
reduzido totalmente convolucional de Oxford, publicado sob CC BY 4.0 por Karen
Simonyan e Andrew Zisserman.

</provenance-box>

## Citação

<citation-block />
