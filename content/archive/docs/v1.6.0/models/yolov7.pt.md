---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 no LibreYOLO: predição, treinamento e exportação sob a MIT'
description: >-
  Use o YOLOv7 no LibreYOLO para detecção de objetos: instale, rode predições,
  treine, valide e exporte, com código e pesos sob a licença MIT.
lead: >-
  O YOLOv7 é um detector de estágio único baseado em âncoras, cuja cabeça soma
  offsets de conhecimento implícito aprendidos antes da convolução final. O
  LibreYOLO suporta seu único tamanho publicado para detecção.
keywords:
  - YOLOv7
  - detecção de objetos
  - YOLOv7 python
  - treinar YOLOv7 dataset próprio
  - detecção em tempo real
  - ImplicitA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Warm start a partir de um modelo novo
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True sempre carrega o checkpoint LibreYOLO7b.pt publicado,
        # independentemente dos pesos com que esta instância foi construída.
        # Construir a classe diretamente, em vez de passar pelo LibreYOLO(),
        # começa sem nenhum peso carregado.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Instalação

O YOLOv7 não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` define o limiar de
confiança e `iou`, o limiar do NMS aplicado depois que a cabeça baseada em
âncoras é decodificada. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

O LibreYOLO traz um tamanho, `b`. O upstream publica um único modelo YOLOv7,
então não há tamanho a escolher.

## Treinamento

<code-tabs name="train" />

O `pretrained` é lido aqui, diferente do no-op de mesmo nome em algumas outras
famílias: passe `True` para dar warm start a partir do checkpoint
`LibreYOLO7b.pt` publicado (baixado automaticamente), ou um caminho ou nome para
qualquer outro. Esse checkpoint publicado é COCO de 80 classes, então, se você o
pedir em um modelo já reconstruído para outra contagem de classes, ele primeiro
reconstrói de volta para 80, carrega o checkpoint e depois transfere todo tensor
de shape compatível para a contagem da cabeça alvo assim que a contagem de
classes do dataset é lida. `resume=True` não pode ser combinado com
`pretrained`. Deixado no padrão `None`, o treinamento continua a partir dos
pesos com que o modelo foi construído, ou de uma inicialização aleatória se nada
foi carregado.

Sem mexer em nada, o trainer roda 300 épocas em `lr0=0.01` com SGD de momentum
0.937, um warmup de 3 épocas, e a mesma atribuição SimOTA e a mesma fase final
de 15 épocas sem data augmentation que o YOLOX usa, adaptadas à cabeça baseada
em âncoras. A única diferença: o YOLOX adiciona um refinamento L1 da regressão
de caixas durante essas épocas finais, que o v7 pula, porque a loss SimOTA do v7
não tem um branch L1 de offsets brutos para refinar.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

O `val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
