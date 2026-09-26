---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: estimativa de pose top-down no LibreYOLO'
description: >-
  Use o HRNet no LibreYOLO para estimativa de pose top-down COCO-17. Instale,
  faça predições, valide e exporte os checkpoints W32 e W48, com licença MIT.
lead: >-
  O HRNet é uma rede convolucional que mantém um fluxo de características em
  alta resolução por meio de fusões multiescala repetidas, em vez de recuperar a
  resolução depois de reduzi-la. O LibreYOLO empacota a variante oficial de pose
  top-down para inferência e validação.
keywords:
  - HRNet
  - estimativa de pose humana
  - pose top-down
  - keypoints COCO-17
  - hrnet pose pytorch
  - estimativa de pose python
  - detectar pontos do corpo em imagem
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sem fonte de pessoas informada: o HRNet se pareia sozinho com um
        # detector LibreYOLO9t leve e registra essa escolha uma vez.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Fonte de pessoas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Pula a detecção por completo: trata a imagem inteira como uma pessoa.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Ou passe ao HRNet boxes de um detector que você já rodou.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Ou pareie com um detector LibreYOLO específico em vez do
        # padrão LibreYOLO9t.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # O grafo exportado é só a cabeça de mapas de calor de tela fixa: ela
        # recebe um batch de recortes de pessoa já recortados e já
        # normalizados e devolve mapas de calor brutos. A detecção de pessoas,
        # a geometria do recorte, a decodificação dos mapas de calor e a
        # supressão por OKS não fazem parte deste grafo; rodá-lo fora do
        # LibreYOLO significa reimplementar essa etapa de decodificação por
        # conta própria.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Instalação

O HRNet não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

Seu detector de pessoas padrão, um checkpoint LibreYOLO9t leve, é baixado
automaticamente na primeira vez que o HRNet se pareia com ele.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O HRNet é um estimador de pose top-down: ele precisa de uma caixa de pessoa
antes que a cabeça de pose possa rodar, então toda chamada resolve uma. Se você
não disser nada, ele se pareia com um detector LibreYOLO9t na primeira vez e
registra essa escolha. `cropped=True` pula a detecção e trata a imagem inteira
como uma pessoa; `person_boxes` aceita boxes de um detector que você já rodou;
`person_detector` aceita `"auto"`, `"rfdetr"`, qualquer modelo de detecção do
LibreYOLO ou um callable simples. `flip_test=True` roda o modelo também no
recorte espelhado horizontalmente e faz a média dos dois mapas de calor, o data
augmentation em tempo de teste do próprio HRNet; o `augment=True` genérico não é
definido aqui. Fontes com várias imagens rodam sequencialmente: o detector do
HRNet e o número variável de pessoas por imagem não suportam predição
empilhada. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Variantes

Dois tamanhos, `w32` e `w48`, ambos prevendo o conjunto padrão de keypoints
COCO-17 a partir de um recorte de pessoa em resolução fixa; o `w48` é o mais
largo dos dois backbones.

O model zoo upstream reporta a acurácia de pose de cada tamanho com o próprio
detector de pessoas, a própria configuração de flip-test e o protocolo oficial
de avaliação do COCO. O pareamento padrão do LibreYOLO usa um detector
diferente, então uma rodada de validação aqui mede essa combinação, não a do
upstream; bater com os números do upstream exige as mesmas caixas de pessoa, os
mesmos scores do detector e a mesma configuração de flip que a avaliação
original usou.

## Validação

`val()` roda o OKS-AP de keypoints no estilo COCO e aceita um `data.yaml` no
formato YOLO-pose ou um JSON de keypoints do COCO junto com um diretório de
imagens. O backend de métricas é o faster-coco-eval por padrão, com o
`pycocotools` sendo usado automaticamente quando o faster-coco-eval não está
instalado; `faster_coco_eval=False` força o caminho do `pycocotools`.

<code-tabs name="val" />

A validação aciona internamente o próprio `predict()` do HRNet, então ela usa
qualquer detector de pessoas com que o modelo tenha sido construído ou chamado.
Construa o modelo com um `person_detector=` explícito para manter essa fonte
fixa entre as rodadas, em vez de deixar cada chamada resolver o padrão de novo.

## Exportação

<export-matrix />

O contrato de exportação do HRNet cobre apenas ONNX, TorchScript, OpenVINO e
TensorRT; qualquer outro formato levanta erro antes de o trace começar. Toda
exportação é só a cabeça de mapas de calor de tela fixa, FP32 com batch um, que
recebe um recorte de pessoa e devolve mapas de calor brutos: a geometria afim do
recorte antes dela e a decodificação dos mapas de calor, a restauração do flip e
a supressão por OKS depois dela ficam em Python, então um pipeline completo de
imagem na entrada e keypoints na saída ainda precisa do LibreYOLO do outro lado.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
