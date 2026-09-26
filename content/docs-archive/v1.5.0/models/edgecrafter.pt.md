---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: detecte, estime pose e segmente no LibreYOLO'
description: >-
  Use o EdgeCrafter no LibreYOLO para detecção, pose e segmentação de
  instâncias. Instale, rode predições, valide e exporte, com código licenciado
  sob a MIT.
lead: >-
  Um vision transformer compacto para predição densa em hardware de borda,
  publicado no upstream como três modelos irmãos: ECDet, ECPose e ECSeg. O
  LibreYOLO carrega os três como uma única família, com a tarefa carregada pelo
  checkpoint.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - vision transformer compacto
  - detecção de objetos python
  - estimativa de pose keypoints
  - segmentação de instâncias
  - inferência em dispositivo de borda
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -pose no nome do arquivo seleciona a cabeça de keypoints,
        # então nenhum argumento de tarefa é necessário aqui.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Precisa de um dataset de keypoints de classe única cujo data.yaml
        # declare kpt_shape, e imgsz no tamanho nativo do checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentação de instâncias
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Precisa de labels em polígono, e imgsz no tamanho nativo do
        checkpoint.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # caixas
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # é carregado como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Instalação

O EdgeCrafter não precisa de nenhum extra opcional. Tudo que ele importa já está
na instalação base.

```bash
pip install libreyolo
```

O fine-tuning com adaptadores via `lora=True` é a exceção, e precisa do extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

A tarefa vem do nome do arquivo, então um checkpoint `-pose` ou `-seg` seleciona
a própria cabeça e não recebe argumento de tarefa. Os três retornam o objeto
`Results` que todas as famílias retornam, com `result.keypoints` a mais para
pose e `result.masks` para segmentação. A pose cobre uma classe, pessoa, com os
17 keypoints do COCO, e a contagem é fixada quando o modelo é construído. Ela
não tem cabeça de caixas, então cada caixa de pose é a extensão que envolve os
próprios keypoints, e o terceiro canal do keypoint é uma constante em vez de uma
pontuação por ponto.

`conf` e `max_det` filtram a seleção de queries; `iou` é aceito por paridade de
API mas não tem efeito, porque as três cabeças decodificam um conjunto de
queries sem etapa de NMS. Veja [predição](/docs/predict) para fontes, streaming
e tratamento de resultados.

## Variantes

Quatro tamanhos. Todos rodam na mesma resolução de entrada, então a tabela os
separa por número de parâmetros e acurácia.

<benchmark-table task="detect" />

<va-embed />

O upstream publica ECDet, ECPose e ECSeg como três modelos separados, em vez de
um modelo com três cabeças. Eles compartilham o backbone ECViT e o encoder
híbrido e diferem apenas na cabeça, então o LibreYOLO os agrupa em uma única
família e deixa o nome do arquivo do checkpoint carregar a tarefa. Uma letra de
tamanho significa, portanto, o mesmo backbone e o mesmo encoder nos três, e
predição, validação e exportação recebem os mesmos argumentos, seja qual for o
que você carregar.

## Treinamento

As três tarefas treinam pelo `train()`, que lê a tarefa do checkpoint carregado
e escolhe o trainer correspondente.

<code-tabs name="train" />

O que foi conferido para detecção e segmentação: paridade de inferência com o
upstream em 1e-5, camada por camada e por tamanho, e que a loss e um único passo
de treinamento rodam com entrada sintética. O que não foi, conforme a própria
docstring de `train()`: a convergência de um fine-tuning completo, o treinamento
multi-GPU, a etapa de recarregar o melhor checkpoint ao parar o data
augmentation, e o remapeamento de classes de Objects365 para COCO. O caminho de
pose segue a receita publicada do DETRPose, um matcher húngaro sobre custos de
classe, L1 de keypoints e OKS com remoção contrastiva de ruído nos keypoints, e
a convergência dele também não foi conferida de ponta a ponta.

Sem mexer em nada, o trainer roda 74 épocas com `lr0=5e-4` e precisão mista
ligada, seguindo a receita do upstream: AdamW, um cronograma cosseno achatado,
EMA em 0.9999 e entradas normalizadas para o ImageNet. Pose e segmentação exigem
`imgsz` no tamanho nativo do checkpoint, porque a grade de âncoras de avaliação
delas é construída quando o modelo é criado; um valor diferente levanta erro
antes de a execução começar. A pose também exige um dataset de classe única cujo
`data.yaml` declare `kpt_shape`, com uma contagem de keypoints que bata com a
cabeça.

`lora=True` vale só para detecção; pose e segmentação levantam `ValueError` com
ele. No Apple silicon o trainer mantém a execução na GPU e manda uma operação
para a CPU, o backward do grid-sample dentro da atenção deformável, que o
PyTorch não implementa no Metal.

Veja [treinamento](/docs/train) para datasets, augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário indexado pelo nome da métrica, e imprime os
resultados por classe quando `verbose` fica ligado.

<code-tabs name="val" />

A pose reporta as métricas OKS de keypoints em `metrics/keypoints_*`. A
segmentação reporta as máscaras na chave `metrics/mAP50-95` pura e repete as
duas visões em uma única passada, caixas em `(B)` e máscaras em `(M)`.

## Exportação

<export-matrix />

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Pose e segmentação exportam com entrada fixa de 640
por 640 em vez de formas dinâmicas, e vários alvos de detecção também são de
canvas fixo, incluindo OpenVINO, Paddle, MNN, ExecuTorch e Core AI.
[Exportação](/docs/export) lista os argumentos que todo formato aceita e os
extras que alguns deles acrescentam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
