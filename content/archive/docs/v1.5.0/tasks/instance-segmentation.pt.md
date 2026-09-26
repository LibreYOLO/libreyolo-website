---
title: Segmentação de instâncias
seo_title: Segmentação de instâncias no LibreYOLO
description: >-
  Segmente objetos individuais no LibreYOLO: as famílias que atendem à tarefa, o
  formato de rótulos em polígono e as chamadas de predição, treinamento,
  validação e exportação.
lead: >-
  A segmentação de instâncias localiza cada instância de objeto e retorna uma
  máscara por pixel para cada uma, junto com a caixa, a classe e a pontuação que
  um detector retorna. A chave da tarefa é segment.
keywords:
  - segmentação de instâncias python
  - máscara de objeto yolo
  - treinar modelo de segmentação
  - rótulos de polígono yolo
  - biblioteca de segmentação licença MIT
  - mAP de máscara
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -seg no nome do arquivo seleciona a cabeça de máscaras,
        # então nenhum argumento de tarefa é necessário.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), uma máscara por detecção
        print(result.boxes.xyxy.shape)   # (N, 4), as mesmas N linhas
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Contornos das máscaras
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy é uma lista de contornos (P, 2) em pixels, .xyn os mesmos
        normalizados.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Outra família, mesma chamada'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continua a partir de pesos de segmentação publicados, com a cabeça de
        máscaras.

        # data deve apontar para um dataset cujos rótulos tenham polígonos.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: A partir de pesos de detecção
      language: bash
      code: |
        # Pesos de detecção não trazem cabeça de máscaras, então isto é uma
        # transferência explícita: a cabeça começa sem treinamento. Pedir
        # task=segment é o que autoriza isso.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # máscaras
        print(metrics["metrics/mAP50-95(M)"])    # máscaras, explícito
        print(metrics["metrics/mAP50-95(B)"])    # caixas
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # é carregado como um checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definição

Segmentação de instâncias é detecção mais forma. Cada instância de objeto
continua recebendo uma caixa, uma classe e uma pontuação, e também recebe uma
máscara binária cobrindo os pixels que pertencem a ela. As máscaras podem se
sobrepor, e os pixels que não pertencem a nenhum objeto ficam sem atribuição, o
que separa a tarefa da
[segmentação semântica](/docs/tasks/semantic-segmentation) e da
[segmentação panóptica](/docs/tasks/panoptic-segmentation).

`segment` é a chave canônica da tarefa, e o sufixo `-seg` no nome de arquivo de
um checkpoint a seleciona, então `task=` não é necessário ao carregar pesos
publicados.

`predict()` preenche `result.masks` junto com `result.boxes`. `.data` é uma
pilha `(N, H, W)` no canvas da imagem original, alinhada linha a linha com as
caixas, então a máscara `i` pertence à caixa `i`. `.xy` converte cada máscara em
seu maior contorno externo como um array `(P, 2)` de pixels, e `.xyn` dá o mesmo
contorno normalizado.

## Modelos

Quatro famílias tanto treinam quanto predizem máscaras:
[RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) e
[RTMDet](/docs/models/rtmdet). A RF-DETR precisa do próprio extra,
`pip install "libreyolo[rfdetr]"`; as outras três rodam no pacote base.

O [Mask R-CNN](/docs/models/mask-rcnn) prediz, valida e exporta máscaras, mas
seu `train()` levanta `NotImplementedError`.

O [EoMT](/docs/models/eomt) prediz e valida máscaras e também não treina, e sua
exportação é ainda mais restrita: `export()` só aceita a tarefa semântica, e
levanta `NotImplementedError` para `segment` e `panoptic`, porque o contrato de
runtime de máscaras por query de que essas duas precisam não foi definido. Use o
EoMT para máscaras de instância em Python, e não por meio de um grafo exportado.

Um grupo à parte segmenta a partir de um prompt em vez de uma lista de classes:
um clique, uma caixa ou uma frase escolhe o objeto, e o modelo retorna a máscara
dele. [SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) e [PicoSAM3](/docs/models/picosam3) funcionam
assim, e o [SenseNova-Vision](/docs/models/sensenova-vision) também, cuja
segmentação é do tipo referring: ele recebe uma frase que nomeia um objeto. Eles
são carregados pela própria factory e pelos próprios extras, e cada página de
modelo traz a chamada exata.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

`conf` e `max_det` moldam a saída do mesmo jeito que na detecção, e as
máscaras são filtradas junto com as caixas às quais pertencem. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

O layout é o layout de detecção: um arquivo `.txt` de rótulos por imagem,
encontrado trocando `images` por `labels` no caminho da imagem e mudando a
extensão.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

O que muda é a linha. Um segmento é um índice de classe seguido de um polígono
achatado:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

No mínimo três pontos, então a contagem de coordenadas depois do índice de
classe é par e no mínimo seis, e o polígono precisa ser não degenerado. As
coordenadas são floats em `[0, 1]` relativos à largura e à altura originais da
imagem. Uma linha de detecção de cinco campos também é aceita em um dataset de
segmentação e é lida como um segmento retangular, o que permite carregar um
dataset só de caixas sem uma etapa de conversão.

O YAML é o YAML de detecção:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

COCO JSON nativo também funciona: adicione um mapeamento `annotations` de nome
do split para arquivo JSON, e o caminho do split dá a raiz das imagens.

## Treinamento

<code-tabs name="train" />

Por padrão, o treinamento continua a partir de um checkpoint `-seg` publicado.
Começar a partir de pesos de detecção é possível, mas é uma transferência
deliberada: esses pesos não trazem cabeça de máscaras, então ela começa sem
treinamento, e passar `task=segment` é o que autoriza a troca. Veja
[treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário simples de chaves `metrics/`. Caixas e máscaras
são pontuadas separadamente, ambas com avaliação COCO, e os números das
máscaras são os principais.

<code-tabs name="val" />

As chaves sem sufixo trazem os resultados de máscara: `metrics/mAP50-95`,
`metrics/mAP50`, `metrics/mAP75`, depois `metrics/mAP_small`,
`metrics/mAP_medium` e `metrics/mAP_large` por área do objeto, e `metrics/AR1`,
`metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`,
`metrics/AR_large` para o recall médio. `metrics/AR_max_det` e
`metrics/max_det` registram o limite de detecções que a execução usou.

Quatro números também são publicados sob um sufixo explícito, `(M)` para
máscara e `(B)` para caixa, para que uma comparação nunca dependa de qual número
a família decidiu chamar de principal: `metrics/mAP50-95(M)` e
`metrics/mAP50-95(B)`, `metrics/mAP50(M)` e `metrics/mAP50(B)`,
`metrics/precision(M)` e `metrics/precision(B)`, `metrics/recall(M)` e
`metrics/recall(B)`. Não existe `metrics/precision` nem `metrics/recall` sem
sufixo nesta tarefa.

Leia as chaves de precisão e recall com atenção. Elas são mantidas por
compatibilidade retroativa e são aliases, não um ponto de operação:
`metrics/precision(M)` guarda o mesmo valor que `metrics/mAP50-95(M)`, e
`metrics/recall(M)` o mesmo valor que o AR de máscara em 100 detecções, com
`(B)` se comportando do mesmo jeito para as caixas. Plotar um par delas informa
o mesmo número duas vezes.

## Exportação

<code-tabs name="export" />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. A cobertura de segmentação é mais restrita que a
cobertura de detecção na mesma família. A matriz em cada página de modelo é
gerada a partir do conjunto validado e indica o motivo de um alvo não estar
disponível. Veja
[exportação e deploy](/docs/export) para os formatos, seus extras e suas
restrições.
