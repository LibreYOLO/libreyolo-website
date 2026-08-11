---
title: Segmentação semântica
seo_title: Segmentação semântica no LibreYOLO
description: >-
  Rotule cada pixel com uma classe no LibreYOLO: as famílias que atendem à
  tarefa, o formato de máscara densa e as chamadas de predição, treinamento,
  validação e exportação.
lead: >-
  A segmentação semântica atribui uma classe a cada pixel de uma imagem e não
  faz distinção entre instâncias da mesma classe. A chave da tarefa é semantic.
keywords:
  - segmentação semântica python
  - classificação de pixels
  - predição densa
  - treinar modelo de segmentação
  - mIoU
  - biblioteca de segmentação licença MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -sem no nome do arquivo seleciona a tarefa, então nenhum
        # argumento task é necessário.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe no canvas original
        print(mask.classes)      # ids de classe presentes, ordenados, sem o 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Uma classe por vez
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # booleano (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Outra família, mesma chamada'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Com ADE20K
      language: bash
      code: >
        # ade20k.yaml traz um script de download embutido para o arquivo de ~1
        GB,

        # então precisa de permissão explícita a menos que os dados já estejam
        locais.

        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() retorna um dict comum, não um objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como um checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definição

A segmentação semântica rotula pixels, não objetos. Cada pixel recebe um id de
classe, e dois carros que se tocam na imagem viram uma única região da classe
carro, sem fronteira entre eles. Contar instâncias é
[segmentação de instâncias](/docs/tasks/instance-segmentation); rotular cada
pixel e separar instâncias ao mesmo tempo é
[segmentação panóptica](/docs/tasks/panoptic-segmentation).

`semantic` é a chave canônica da tarefa, e o sufixo `-sem` no nome do arquivo de
um checkpoint a seleciona, então `task=` não é necessário ao carregar pesos
publicados.

`predict()` preenche `result.semantic_mask`. `.data` é um mapa inteiro de
classes `(H, W)` no canvas da imagem original, `.classes` lista os ids
presentes em ordem crescente, e `.class_mask(id)` retorna a seleção booleana
`(H, W)` de uma classe. O valor `255` é o rótulo de ignorar: nunca é uma
classe, fica de fora da loss e das métricas, e `.classes` o deixa de lado.

## Modelos

Três famílias tanto treinam quanto predizem:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) e
[DINOv2](/docs/models/dinov2). SegFormer e LingBot-Vision rodam com o pacote
base e trazem pesos publicados. DINOv2 precisa de
`pip install "libreyolo[rfdetr]"` e não tem checkpoint hospedado pelo LibreYOLO:
carrega o backbone do upstream e sua cabeça densa começa com inicialização
aleatória, então é um ponto de partida para treinamento em vez de um preditor
pronto para uso.

Outras quatro predizem, validam e exportam, mas seu `train()` lança
`NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) e
[EoMT](/docs/models/eomt).

Os conjuntos de classes variam por checkpoint, não por família. Os pesos
publicados vêm de datasets cujos espaços de rótulos têm pouco em comum, entre
eles as 150 classes do ADE20K contra as 19 do Cityscapes, então o `names` de um
checkpoint é o que diz o que ele consegue rotular, e dois checkpoints só são
comparáveis quando foram treinados no mesmo espaço de rótulos.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O mapa é um argmax por pixel, então não há etapa de NMS e `iou` nunca tem
efeito. `conf` e `max_det` são aceitos por paridade de API e não fazem nada no
SegFormer, no PIDNet e nos demais preditores densos; o EoMT é a exceção, onde
`conf` filtra a seleção de queries. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Formato do dataset

Cada imagem é pareada com uma máscara densa de canal único em vez de um arquivo
de rótulos `.txt`, encontrada trocando `images` pelo diretório de máscaras no
caminho da imagem.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

As máscaras são imagens de canal único sem perdas, normalmente PNG, e PNGs em
modo paleta são lidos como índices de paleta. Cada valor de pixel é um id de
classe em `0..nc-1`, o valor `255` significa ignorar, e a resolução da máscara
tem que ser igual à da imagem pareada.

O YAML aceita duas chaves além do contrato compartilhado:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` é o nome de diretório que substitui `images`, e por padrão é
`masks`. `label_mapping` é um remapeamento opcional `{source_id: train_id}`
aplicado aos valores de pixel da máscara no carregamento, que é como um dataset
numerado de 1 a 150 passa a ir de 0 a 149; qualquer valor de origem que fique
sem mapeamento vira ignorar, e todo train id tem que cair em `0..nc-1`.

Omitir `masks_dir` muda o loader para um modo alternativo: as máscaras são
rasterizadas no carregamento a partir de rótulos em polígono resolvidos pela
convenção usual de `images` para `labels`, e uma classe `background` é
adicionada depois das classes de objeto, então `nc` cresce em um.

O loader canônico é `libreyolo.data.SemanticDataset`.

## Treinamento

<code-tabs name="train" />

Aqui o `imgsz` tem uma restrição que não existe em um detector. Cada família
declara um divisor do qual a entrada precisa ser múltipla, definido pela grade
de patches ou pelo stride de saída, e tanto o treinamento quanto a validação
lançam um `ValueError` antes de a execução começar quando o `imgsz` não é
múltiplo dele. O divisor é 32 para SegFormer, 16 para LingBot-Vision e EoMT, 14
para DINOv2, e 8 para FCN e PIDNet. Veja [treinamento](/docs/train) para datasets,
data augmentation, multi-GPU e loggers.

## Validação

`val()` retorna um dicionário comum de chaves `metrics/`, calculado sobre o
split nomeado por `val` no YAML do dataset.

<code-tabs name="val" />

`metrics/mIoU` é a média da interseção sobre a união: para cada classe, a
sobreposição entre os pixels preditos e os verdadeiros dividida pela união
deles, com média sobre as classes. É o número principal e o usado para escolher
a melhor época durante o treinamento. `metrics/pixel_accuracy` é a proporção de
pixels que receberam a classe correta, que uma classe de fundo grande pode
inflar, então mIoU é o número com que comparar. Pixels marcados como `255` não
contam para nenhum dos dois. O dicionário também traz `fitness`, uma cópia do
valor de mIoU.

## Exportação

<code-tabs name="export" />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do seu
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. A cobertura de formatos varia por família; a matriz
em cada página de modelo é gerada a partir do conjunto validado em vez de
escrita à mão. Veja [exportação e deploy](/docs/export) para os formatos, seus
extras e suas restrições.
