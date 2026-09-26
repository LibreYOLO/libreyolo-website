---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN no LibreYOLO: predição, validação e exportação'
description: >-
  Rode o Mask R-CNN no LibreYOLO para detecção de objetos e segmentação de
  instâncias. Instale, faça predição, validação e exportação do port
  BSD-3-Clause do torchvision.
lead: >-
  O Mask R-CNN acrescenta ao Faster R-CNN um ramo de máscara por região,
  prevendo uma máscara de segmentação junto de cada box que detecta. O LibreYOLO
  porta a implementação do torchvision para detecção e segmentação de
  instâncias.
keywords:
  - Mask R-CNN
  - segmentação de instâncias
  - detecção de objetos python
  - Faster R-CNN
  - torchvision
  - detector de dois estágios
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Só boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" pula a cabeça de máscaras e retorna os boxes do mesmo
        # checkpoint, sem máscaras no resultado.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # boxes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Instalação

O Mask R-CNN não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Carregar o checkpoint sem
o argumento `task` retorna máscaras de instância, já que a segmentação é a
tarefa padrão desta família; `result.masks` então as carrega junto dos boxes.
Passar `task="detect"` carrega os mesmos pesos sem a cabeça de máscaras e
retorna só os boxes. `conf` e `iou` definem os limiares de confiança e de NMS;
o Mask R-CNN mantém a etapa de NMS do upstream, diferente de um detector
baseado em queries. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

Um backbone: ResNet-50 com uma pirâmide de características, usando o builder v2
de Mask R-CNN do torchvision. O checkpoint publicado tem licença BSD-3-Clause e
serve às duas tarefas desta família, então não há tamanho nenhum para escolher.

## Validação

`val()` retorna um dicionário de chaves `metrics/`. Na tarefa de segmentação
padrão deste checkpoint, a chave `metrics/mAP50-95` pura contém a pontuação das
máscaras, e a mesma execução reporta os boxes sob o sufixo `(B)`, então as duas
ficam disponíveis em uma única passada.

<code-tabs name="val" />

## Exportação

<export-matrix />

O Mask R-CNN exporta só para ONNX, com tamanho de batch 1. O grafo exportado
mantém dentro dele as etapas de redimensionamento e de colagem de máscaras do
upstream, então o LibreYOLO força `dynamic=True` independentemente do que for
passado, para manter o grafo válido para fontes que não são quadradas. Um
arquivo `.onnx` exportado é recarregado por `LibreYOLO()` pelo sufixo do arquivo
e retorna o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família. O único checkpoint abaixo
aparece listado sob detect, mas o mesmo arquivo carrega para segmentação
também: não passe o argumento `task` e ele retorna máscaras por padrão.

<checkpoint-table />

## Licenciamento

<provenance-box>

O Mask R-CNN é construído como uma subclasse do wrapper de Faster R-CNN do
LibreYOLO: compartilha a mesma fonte do torchvision e a mesma licença
BSD-3-Clause, e adiciona o preditor de máscaras e a cabeça RoI de máscaras do
mesmo commit portado.

</provenance-box>
