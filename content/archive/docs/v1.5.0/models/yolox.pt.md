---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: prever, treinar e exportar sob Apache-2.0'
description: >-
  Use o YOLOX no LibreYOLO para detecção de objetos: instale, faça predições,
  treine, valide e exporte sob Apache-2.0.
lead: >-
  O YOLOX é um detector de estágio único e sem âncoras (anchor-free), com uma
  cabeça desacoplada de classificação e regressão, treinado com atribuição de
  rótulos SimOTA. O LibreYOLO o suporta para detecção.
keywords:
  - YOLOX
  - detecção de objetos python
  - detector anchor-free
  - detecção de objetos em tempo real
  - cabeça desacoplada
  - SimOTA
  - treinar yolox dataset próprio
  - exportar yolox onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Contra o COCO
      language: bash
      code: |
        # O yaml do COCO incluído carrega um script de download embutido, então
        # precisa de permissão explícita a menos que o dataset já esteja local.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pela extensão do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Instalação

O YOLOX não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` define o limiar de
confiança e `iou` o limiar de NMS aplicado nas três escalas de predição
desacopladas. Veja [predição](/docs/predict) para fontes, streaming e tratamento
de resultados.

## Variantes

Seis tamanhos compartilham o mesmo backbone CSP e o mesmo neck PAFPN. Os dois
menores, `n` e `t`, rodam em uma resolução de entrada fixa menor que a dos
outros quatro; a tabela de benchmark abaixo traz o número exato de cada um.

<benchmark-table task="detect" />

<va-embed />

## Treinamento

<code-tabs name="train" />

Sem nenhum ajuste, o treinador roda 300 épocas com `lr0=0.01`, SGD com momentum
0.9, um warmup de 5 épocas e o data augmentation de mosaic e mixup desligado nas
últimas 15 épocas. `train()` também aceita um argumento `pretrained`, mas o
valor nunca é lido dentro do método: o treinamento sempre continua a partir dos
pesos com que o modelo foi construído, então `pretrained=False` não
reinicializa a rede.

`imgsz` assume por padrão um valor fixo da configuração base de treinamento, e
não a resolução nativa do checkpoint carregado. Isso afeta especificamente os
checkpoints `n` e `t`: continuar a treinar qualquer um dos dois sem definir
`imgsz` explicitamente o eleva ao padrão maior, em vez do tamanho menor com que
foi publicado.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado carrega de volta através de `LibreYOLO()` pela extensão do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta. Uma exportação para CoreML pode embutir o NMS no grafo com
`nms=True`; YOLOX e YOLOv9 são as duas únicas famílias que essa flag aceita
atualmente.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
