---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet no LibreYOLO: predição, treinamento e exportação'
description: >-
  Use o RTMDet no LibreYOLO para detecção de objetos e segmentação de instâncias
  com o RTMDet-Ins. Instale, rode predições, treine, valide e exporte sob a
  Apache-2.0.
lead: >-
  O RTMDet é um detector de estágio único que prediz a partir de um prior por
  ponto em cada posição da grade, sem âncoras, através de uma cabeça cujas
  convoluções são compartilhadas entre os níveis de features. O LibreYOLO o
  suporta para detecção e para segmentação de instâncias com o RTMDet-Ins.
keywords:
  - RTMDet
  - detecção de objetos
  - segmentação de instâncias
  - RTMDet-Ins
  - detecção anchor-free
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -seg no nome do arquivo seleciona a cabeça de máscaras do
        # RTMDet-Ins, então nenhum argumento task é necessário aqui.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # caixas
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Instalação

O RTMDet não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Um nome de arquivo com
`-seg` já resolve sozinho para a tarefa do RTMDet-Ins, e aí `result.masks`
carrega as máscaras de instância junto com as caixas. `conf` define o limiar de
confiança e `iou`, o limiar do NMS. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Cinco tamanhos, de `t` a `x`, compartilham uma mesma arquitetura em uma
resolução de entrada comum. Esta família não traz tabela de benchmark aqui:
compare os tamanhos pelo tamanho do arquivo de cada checkpoint na tabela abaixo.

## Treinamento

<code-tabs name="train" />

A detecção treina pelo `train()`. Os componentes QualityFocalLoss, GIoU e
DynamicSoftLabelAssigner são portados do mmdetection upstream, e o forward pass
e a exportação para ONNX são equivalentes bit a bit a ele, com o
pós-processamento batendo com a saída do mmdet dentro de 0.001 mAP em
subconjuntos do val2017.

O que não foi conferido, conforme a própria docstring de `train()`: a
convergência de um fine-tuning em dataset pequeno, a paridade com o paper
treinando do zero, o comportamento multi-GPU, o throughput de Mosaic e MixUp em
cache, a troca estrita para o pipeline de dois estágios do upstream, e os
overrides de weight decay por parâmetro que zeram o decay nos parâmetros de norm
e de bias.

O RTMDet-Ins não tem caminho de treinamento. Chamar `train()` em um checkpoint
`-seg`, ou com `task="segment"`, levanta `NotImplementedError`; a segmentação de
instâncias suporta apenas inferência e validação.

O `train()` também aceita um argumento `pretrained`, mas o valor nunca é lido
dentro do método: o treinamento sempre continua a partir dos pesos com que o
modelo foi construído, então `pretrained=False` não reinicializa a rede.

Sem mexer em nada, o trainer roda 300 épocas com AdamW em `lr0=0.004` e
`weight_decay=0.05`, um warmup de 1 época em um cronograma cosseno, e Mosaic e
MixUp desligados nas 20 épocas finais.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

O `val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

Contra um checkpoint `-seg`, a chave `metrics/mAP50-95` pura contém a pontuação
das máscaras, e a mesma execução também reporta as caixas em `(B)` e as máscaras
em `(M)`, então as duas ficam disponíveis em uma única passada.

## Exportação

<export-matrix />

A detecção exporta para a maioria dos formatos; a segmentação de instâncias hoje
não exporta para nenhum deles, e a matriz acima reflete essa divisão. Um artefato
de detecção exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
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
