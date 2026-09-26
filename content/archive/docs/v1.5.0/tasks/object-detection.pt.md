---
title: Detecção de objetos
seo_title: Detecção de objetos no LibreYOLO
description: >-
  Detecte objetos como caixas alinhadas aos eixos no LibreYOLO: as famílias que
  atendem à tarefa, o formato dos rótulos e as chamadas de predição,
  treinamento, validação e exportação.
lead: >-
  A detecção de objetos localiza cada instância de objeto em uma imagem e
  devolve um retângulo alinhado aos eixos, um rótulo de classe e um score para
  cada uma. A chave da tarefa é detect.
keywords:
  - detecção de objetos python
  - detectar objetos em imagem
  - bounding box python
  - biblioteca de detecção de objetos licença MIT
  - alternativa ao YOLO
  - treinar detector de objetos dataset próprio
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Outra família, mesma chamada'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo checkpoint, e todo detector devolve o mesmo
        # objeto Results, então trocar de família é uma mudança de uma linha.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: Vídeo e streams
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice
        # de webcam, stream RTSP ou uma lista .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml baixa uma amostra de 128 imagens no primeiro uso. Aponte
        # data para o YAML do seu próprio dataset em uma execução real.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() devolve um dict simples, não um objeto.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como um checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definição

A detecção de objetos responde onde cada objeto está e o que ele é. Uma imagem
entra, uma linha por instância sai: quatro números para o retângulo, um índice
de classe e um score. Nada de forma em pixels, orientação ou partes é incluído,
e é isso que a separa da [segmentação de instâncias](/docs/tasks/instance-segmentation),
das [caixas orientadas](/docs/tasks/oriented-detection) e da
[pose](/docs/tasks/pose-estimation).

`detect` é a chave canônica da tarefa e o padrão: um checkpoint cujo nome de
arquivo não traz sufixo de tarefa é carregado como detector.

`predict()` preenche `result.boxes`. `.xyxy` dá os cantos em pixels no canvas
da imagem original, `.conf` o score e `.cls` o índice da classe em
`result.names`. `.xywh`, `.xyxyn` e `.xywhn` são visões derivadas das mesmas
linhas, e `.id` carrega um id de track assim que um tracker é acoplado. Iterar
um objeto `Boxes` produz fatias de uma linha, então `box.cls`, `box.conf` e
`box.xyxy` funcionam por detecção.

## Modelos

Doze famílias treinam e fazem predição: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) e [PicoDet](/docs/models/picodet). YOLOv9 e
RF-DETR são as duas famílias principais, e as novidades chegam nelas primeiro.
O RF-DETR precisa do seu próprio extra, `pip install "libreyolo[rfdetr]"`; o
resto roda no pacote base.

Outras onze fazem predição, validação e exportação, mas o `train()` delas
lança `NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) e
[EfficientDet](/docs/models/efficientdet).

A linhagem Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) e
[YOLOv4](/docs/models/yolov4), é mantida congelada, como peça de museu:
predição, validação e exportação funcionam, treinamento não.

Um grupo à parte recebe sua lista de classes em runtime, e não do checkpoint,
então detecta nomes nunca vistos no treinamento:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) e [OV-DEIM](/docs/models/ov-deim),
além das famílias de visão-linguagem
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) e
[LibreMODUS](/docs/models/libremodus). Esses carregam pela sua própria factory
e pelos seus extras; cada página de modelo traz a chamada exata.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

`conf` define o limiar de confiança e `max_det` limita o número de linhas.
`iou` é o limiar do NMS, então só tem efeito em uma família que roda NMS; o
RF-DETR e a cabeça end-to-end do YOLOv9 decodificam um conjunto fixo de
predições e o ignoram. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Formato do dataset

Um arquivo `.txt` de rótulos por imagem, encontrado trocando `images` por
`labels` no caminho da imagem e mudando a extensão.

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

Cada linha tem exatamente cinco campos, um índice de classe seguido de uma
caixa normalizada de centro e tamanho:

```text
<class_id> <cx> <cy> <w> <h>
```

As coordenadas são floats em `[0, 1]`, relativas à largura e à altura da imagem
original. `w` e `h` precisam ser positivos. Um arquivo de rótulos ausente ou
vazio significa que a imagem não tem objetos. As linhas não carregam confiança
nem id de track.

O YAML nomeia os splits e as classes:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` e `val` podem ser diretórios de imagens, arquivos `.txt` com listas de
imagens ou listas de qualquer um dos dois. `nc` é opcional e precisa bater com
`names` quando estiver presente. COCO JSON nativo também funciona: adicione um
mapeamento `annotations` de nome do split para arquivo JSON, e o caminho do
split passa a dar a raiz das imagens. Quando `names` está presente, ele define
os ids dos rótulos, então os nomes de categoria do JSON têm que bater com ele.

## Treinamento

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` e `lr0` são os primeiros argumentos a mexer. `lr0`
é o que não se transfere entre famílias: uma taxa que um detector convolucional
tolera faz um transformer divergir, então pegue o valor na página
do modelo em vez do exemplo de outra família. Uma família também pode ignorar
um argumento por completo, e a página dela lista quais. Veja
[treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário simples de chaves `metrics/`, calculadas com a
avaliação COCO sobre o split indicado por `val` no YAML do dataset.

<code-tabs name="val" />

`metrics/mAP50-95` é a média da mean average precision sobre os limiares de
IoU de 0.50 a 0.95, e é o número de destaque. `metrics/mAP50` e
`metrics/mAP75` são as versões de um único limiar. `metrics/mAP_small`,
`metrics/mAP_medium` e `metrics/mAP_large` separam a mesma média por área do
objeto, e `metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` e `metrics/AR_large` são os números correspondentes de
average recall. `metrics/AR_max_det` e `metrics/max_det` registram o limite de
detecções que a execução usou.

Leia `metrics/precision` e `metrics/recall` com cuidado nesta tarefa. Elas são
mantidas por compatibilidade retroativa e são apelidos, não um ponto de
operação: `metrics/precision` guarda o mesmo valor que `metrics/mAP50-95`, e
`metrics/recall` o mesmo valor que `metrics/AR100`. Plotar as duas como um par
precisão-recall reporta o mesmo número duas vezes. Quatro chaves também se
repetem com o sufixo `(B)`, de box, para que uma chave de detecção seja lida do
mesmo jeito em um modelo que também prediz máscaras: `metrics/mAP50-95(B)`,
`metrics/mAP50(B)`, `metrics/precision(B)` e `metrics/recall(B)`.

## Exportação

<code-tabs name="export" />

Um artefato exportado volta a carregar por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. A cobertura de formatos varia por família; a matriz
em cada página de modelo é gerada a partir do conjunto validado, não digitada à
mão. Veja [exportação e deploy](/docs/export) para os formatos, seus extras e
suas restrições.
