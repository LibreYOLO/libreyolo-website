---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: treine, faça fine-tuning e exporte sob a MIT'
description: >-
  Use o RF-DETR no LibreYOLO para detecção, segmentação de instâncias, pose e
  caixas orientadas. Instale, rode predições, treine, valide e exporte, tudo
  licenciado sob a MIT.
lead: >-
  Um transformer de detecção que prevê um conjunto fixo de objetos em vez de uma
  grade densa, então não precisa de NMS na inferência. O LibreYOLO o suporta
  para quatro tarefas.
keywords:
  - RF-DETR
  - transformer de detecção em tempo real
  - DETR
  - detecção de objetos python
  - segmentação de instâncias
  - estimativa de pose
  - caixas orientadas
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, detecção em vídeo a 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice de
        # webcam, stream RTSP ou uma lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() retorna um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Contra o COCO
      language: bash
      code: |
        # O yaml do COCO que vem junto carrega um script de download embutido,
        # então ele precisa de permissão explícita a menos que o dataset já
        # esteja local.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Argumentos aceitos por todos os formatos:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" é um alias para tensorrt, e "litert" para tflite.

        #   imgsz     int, ou (altura, largura). Por padrão, a resolução nativa

        #             do checkpoint.

        #   batch     int, padrão 1.

        #   half      bool, exporta em FP16. Padrão False.

        #   int8      bool, exporta em INT8. Padrão False. Precisa de `data`.

        #   data      caminho para um YAML de dataset, usado para calibrar o
        int8.

        #   fraction  float, parcela desse conjunto de calibração a usar. Padrão
        1.0.

        #   dynamic   bool, eixos dinâmicos. Padrão True.

        #   simplify  bool, roda a simplificação do grafo ONNX. Padrão True.

        #   opset     int, opset do ONNX. Escolhido por família quando não
        informado.

        #   device    str, dispositivo em que rastrear. Padrão: o do modelo.

        #   output_path  str, padrão é um nome derivado do checkpoint.

        #   verbose   bool, padrão False.

        #   allow_download_scripts  bool, padrão False. Permite Python embutido

        #             em um YAML de dataset que precise ser baixado.

        #

        # Alguns formatos aceitam argumentos extras próprios, como uma

        # plataforma alvo de RKNN. Esses estão documentados na página de cada

        # formato.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Sem o LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Rodar o grafo diretamente significa fazer seu próprio
        pré-processamento

        # e pós-processamento. Inspecione a assinatura antes de ligar qualquer

        # coisa.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Instalação

O RF-DETR precisa do seu próprio extra, que instala o `transformers` para o
backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram
a seleção de queries; não existe etapa de NMS para ajustar. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Quatro tamanhos, e quatro tarefas que compartilham uma mesma arquitetura:
segmentação, pose e caixas orientadas reaproveitam o decoder de detecção com uma
cabeça diferente, então aceitam os mesmos argumentos. Os tamanhos têm contagens
de parâmetros parecidas e diferem principalmente na resolução de entrada.

<benchmark-table task="detect" />

<va-embed />

## Treinamento

O treinamento parte de um checkpoint publicado, para as quatro tarefas. O
RF-DETR lista `pretrained` entre os argumentos que seu trainer nativo ignora,
então passar `pretrained=False` não te dá aqui um modelo inicializado
aleatoriamente.

<code-tabs name="train" />

Dois argumentos importam mais aqui do que em um detector CNN. Mantenha `lr0` em
`1e-4` ou abaixo, já que detectores transformer divergem com learning rates que
um modelo YOLO tolera. Deixe `imgsz` na resolução nativa do checkpoint, a menos
que você tenha um motivo para mudar. A entrada precisa ser divisível exatamente
pelo tamanho de patch do backbone vezes o número de janelas; o LibreYOLO
verifica isso antes de a execução começar e indica os tamanhos válidos mais
próximos.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidas contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Rodar o grafo em um runtime pelado, sem o LibreYOLO
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
