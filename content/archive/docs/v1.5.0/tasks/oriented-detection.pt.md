---
title: Detecção orientada
seo_title: Detecção orientada no LibreYOLO
description: >-
  Detecte objetos rotacionados no LibreYOLO: as famílias que atendem a caixas
  orientadas, a linha de rótulo com quatro cantos e as chamadas de predição,
  treinamento, validação e exportação.
lead: >-
  A detecção orientada de objetos localiza cada instância com um retângulo
  rotacionado em vez de um alinhado aos eixos, então um objeto inclinado fica
  delimitado de forma justa em vez de por uma caixa cheia de fundo. A chave da
  tarefa é obb.
keywords:
  - detecção de caixas orientadas
  - detecção de objetos rotacionados
  - OBB python
  - dataset DOTA
  - detecção de objetos em imagens aéreas
  - IoU rotacionado
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        # Precisa do extra rfdetr: pip install "libreyolo[rfdetr]"

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O sufixo -obb no nome do arquivo seleciona a tarefa, então nenhum

        # argumento de tarefa é necessário.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        result = model(SAMPLE_IMAGE, save=True)


        obb = result.obb

        print(obb.xywhr)   # (N, 5): centro x, centro y, largura, altura,
        radianos

        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Cantos em vez de ângulos
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)

        obb = result.obb


        print(obb.xyxyxyxy.shape)    # (N, 4, 2) pontos dos cantos em pixels

        print(obb.xyxyxyxyn.shape)   # os mesmos, normalizados

        print(obb.xyxy.shape)        # (N, 4) caixa envolvente alinhada aos
        eixos
    - label: Um checkpoint menor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Pesos do DOTA v1.0, 15 classes aéreas a 1024 px. O grafo orientado
        # é reconhecido pelos próprios tensores do checkpoint, então nenhum
        # argumento de tarefa.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane, ship, harbor, helicopter e mais 11
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continua a partir de pesos orientados publicados. data precisa apontar

        # para um dataset cujas linhas de rótulo carreguem quatro cantos.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: A partir de pesos de detecção
      language: bash
      code: |
        # Pesos de detecção não carregam predição de ângulo, então isto é uma
        # transferência explícita. Pedir task=obb é o que a autoriza.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() retorna um dict simples, não um objeto.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX e TorchScript são os alvos validados aqui, em FP32,

        # batch 1, em um canvas fixo de 1024 por 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como um checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Definição

A detecção orientada acrescenta um número à detecção: o ângulo. Cada instância
recebe um retângulo rotacionado, uma classe e uma pontuação. O ganho está no
ajuste justo. Um navio a 45 graus, o telhado de um galpão, uma fileira de
caminhões estacionados: uma caixa alinhada aos eixos em volta de qualquer um
deles é quase toda fundo, e duas caixas vizinhas se sobrepõem mesmo quando os
objetos não se sobrepõem. É por isso que a tarefa é padrão em imagens aéreas e
em layout de documentos, e por isso que o dataset de referência para ela é o
DOTA.

`obb` é a chave canônica da tarefa, e o sufixo `-obb` no nome de arquivo de um
checkpoint a seleciona, então `task=` não é necessário ao carregar pesos
publicados.

`predict()` preenche `result.obb`. `.xywhr` é a forma canônica `(N, 5)`:
centro x, centro y, largura, altura e um ângulo em radianos dando a rotação do
lado da largura em torno do centro. `.conf` e `.cls` carregam a pontuação e o
índice da classe em `result.names`, e `.id` um id de rastreamento quando há
rastreamento. `.xyxyxyxy` converte cada linha nos seus quatro pontos de canto
como `(N, 4, 2)` pixels, `.xyxyxyxyn` normaliza esses cantos, e `.xyxy` dá a
caixa envolvente alinhada aos eixos, que é o que usar quando o código downstream
só entende retângulos. `result.boxes` também é preenchido, com a forma alinhada
aos eixos.

## Modelos

Duas famílias atendem a esta tarefa, e qual delas escolher depende de você
precisar ou não treinar.

O [RF-DETR](/docs/models/rf-detr) é o que treina. Ele prediz, treina, valida e
exporta caixas orientadas, e vem com checkpoints orientados publicados em
quatro tamanhos, n, s, m e l. Ele precisa de um extra próprio,
`pip install "libreyolo[rfdetr]"`, e a página do modelo traz a licença dos
pesos e a proveniência.

Leia a seção abaixo sobre o que esses checkpoints realmente predizem antes de
planejar em cima deles.

O [RT-DETRv2](/docs/models/rt-detr) é o que tem pesos aéreos. Ele publica de
`LibreRTDETRv2n-obb.pt` a `LibreRTDETRv2x-obb.pt`, os checkpoints oficiais
single-scale do DOTA v1.0 convertidos para o formato do LibreYOLO, cobrindo as
15 classes do DOTA a 1024 px. Ele não precisa de nenhum extra além do pacote
base, o grafo orientado é reconhecido pelos próprios tensores do checkpoint, e
predição, validação e exportação para ONNX e TorchScript são todas suportadas.
Treinamento não é: a tarefa orientada é somente inferência nessa família,
`train()` lança erro, e não há transferência a partir dos pesos de detecção
dela, que usam um backbone diferente. Rastreamento e test-time augmentation
também não estão disponíveis para caixas orientadas.

Então: categorias do DOTA prontas para uso, RT-DETRv2. Seus próprios rótulos
orientados, RF-DETR.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

Saiba o que são os checkpoints publicados do RF-DETR antes de rodá-los. Apesar
de o DOTA ser o benchmark de referência para esta tarefa, esses pesos não foram
treinados nele. Todos os quatro foram inicializados a partir dos pesos de
detecção do RF-DETR e passaram por fine-tuning em um único dataset do Roboflow
Universe com imagens de drone, com seis classes de veículos: bike, bus, car,
other_vehicle, taxi e truck. Os model cards descrevem esses checkpoints como
pesos de desenvolvimento, produzidos durante a validação do suporte a treinamento
orientado, e dizem que não devem ser lidos como pesos de produção nem como
pesos oficiais de benchmark.

Na prática isso significa que eles são um ponto de partida funcional para
caixas orientadas em veículos vistos de cima, e para verificar que o seu
pipeline roda de ponta a ponta. Qualquer outro domínio significa treinar com os
seus próprios rótulos orientados, e para as categorias aéreas pelas quais o
DOTA é conhecido, os checkpoints do RT-DETRv2 são os que de fato foram
treinados nesses dados. `conf` e `max_det` moldam a saída como fazem na
detecção. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Formato do dataset

O layout é o layout de detecção: um arquivo de rótulo `.txt` por imagem,
encontrado trocando `images` por `labels` no caminho da imagem e mudando a
extensão.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Uma linha tem exatamente nove campos, um índice de classe seguido de quatro
pontos de canto em ordem:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Os quatro pontos são floats normalizados em `[0, 1]` e precisam formar um
retângulo orientado não degenerado. Nenhum ângulo é armazenado no arquivo de
rótulo: o loader deriva o `xywhr` canônico a partir dos cantos. O parser é
estrito por padrão e rejeita coordenadas fora do intervalo, enquanto a ingestão
de dataset e de validação pode primeiro fazer clipping para `[0, 1]` em rótulos
de fronteira de recorte que sejam válidos no resto, e ainda assim rejeitar
caixas degeneradas.

O parsing das linhas leva a tarefa em conta. Nove campos significam uma caixa
orientada apenas no modo `obb`; no modo `segment` a mesma linha é lida como um
polígono de quatro pontos.

O YAML é o YAML de detecção:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

COCO JSON nativo também é carregado, com um mapeamento `annotations` de nome do
split para arquivo JSON. As anotações são lidas em ordem de prioridade: um
campo `obb` com oito cantos em espaço de pixels, um campo `obb` com
`[cx, cy, w, h, angle]` e o ângulo em radianos, um polígono `segmentation` ou
RLE reajustado ao seu retângulo de área mínima, ou um `bbox` COCO simples, que
é tratado como um retângulo alinhado aos eixos e canonicalizado para `xywhr`.

O parser canônico de linhas é `libreyolo.data.parse_yolo_obb_label_line`.

## Treinamento

<code-tabs name="train" />

Treinar nesta tarefa significa RF-DETR. O treinamento continua a partir de um
checkpoint `-obb` publicado por padrão. Partir de pesos de detecção é uma
transferência deliberada: esses pesos não predizem ângulo, e passar `task=obb`
é o que autoriza a troca. Mantenha `lr0` em `1e-4` ou abaixo, como nas outras
tarefas da família. Os checkpoints orientados do RT-DETRv2 não podem passar por
fine-tuning; use-os como estão, ou treine um modelo RF-DETR com os seus
próprios rótulos. Veja [treinamento](/docs/train) para datasets, augmentation,
multi-GPU e loggers.

## Validação

`val()` retorna um dicionário simples de chaves `metrics/`. O pareamento usa
IoU rotacionado, calculado entre retângulos orientados em vez de entre suas
caixas envolventes alinhadas aos eixos, então uma predição com a posição certa
e o ângulo errado conta como um erro.

<code-tabs name="val" />

`metrics/mAP50-95` é a precisão média (mean average precision) calculada sobre
limiares de IoU de 0.50 a 0.95 em passos de 0.05, e é o número principal.
Diferentemente do caminho COCO usado pela detecção, esta tarefa respeita
`iou_thresholds` na configuração de validação, então a varredura pode ser
alterada. `metrics/mAP50` e `metrics/mAP75` são as versões de limiar único.
`metrics/precision` e `metrics/recall` são precisão e recall de verdade a IoU
0.50, lidos no ponto de operação mais permissivo: toda predição que sobreviveu ao
limiar de confiança é contada, e esse limiar tem 0.001 como padrão durante a
validação. Portanto, aumentar `conf` move esses números, enquanto os valores de
mAP, que usam a curva de precisão-recall inteira, ficam onde estão. Quatro
deles se repetem com um sufixo `(OBB)`, `metrics/mAP50-95(OBB)`,
`metrics/mAP50(OBB)`, `metrics/precision(OBB)` e `metrics/recall(OBB)`, que é
como quem chama distingue um resultado orientado de um alinhado aos eixos
quando os dois estão na mesma tabela. `metrics/mAP75` não tem gêmeo com sufixo.

Duas opções não fazem nada nesta tarefa. `save_json` e `save_plots` são aceitas
e registram um aviso: dumps de predição orientada e gráficos de validação não
estão implementados.

## Exportação

<code-tabs name="export" />

Um artefato exportado é carregado de volta por `LibreYOLO()` a partir do sufixo
do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint
e retorna o mesmo `Results`. A cobertura de formatos difere por tarefa dentro da
mesma família, e a matriz na página do modelo é gerada a partir do conjunto
validado e informa o motivo de um alvo estar indisponível. Veja
[exportação e deploy](/docs/export) para os formatos, seus extras e suas
restrições.
