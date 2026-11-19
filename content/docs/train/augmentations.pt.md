---
title: Data augmentation
seo_title: 'Data augmentation de treinamento no LibreYOLO, com exemplos'
description: >-
  O que cada parâmetro de data augmentation do LibreYOLO faz, mostrado em uma
  imagem real pelo próprio código de treinamento da biblioteca, com sua faixa,
  seu valor padrão e as famílias que o leem.
lead: >-
  O data augmentation de treinamento consiste em transformações aleatórias
  aplicadas a cada amostra de treinamento. Elas são configuradas por parâmetros
  do TrainConfig, e cada família de modelos roda seu próprio pipeline, então um
  parâmetro só altera as amostras onde esse pipeline o lê.
keywords:
  - data augmentation yolo
  - aumento de dados yolo
  - mosaic augmentation
  - mixup
  - jitter hsv
  - transformação afim aleatória
  - random erasing
  - detecção de objetos pequenos
  - exemplos de data augmentation
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # A CLI escreve mosaic_prob como mosaic e mixup_prob como mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # O YOLOX parte de mosaic_scale=(0.1, 2.0), que pode encolher um
        # objeto a um décimo do seu tamanho. (0.5, 1.5) para na metade.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # Na CLI, um par são dois números separados por vírgula.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: Ler a tabela de suporte de uma família
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Só os ignorados
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Pacote de classificação
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 72d22a38d647de0a
---

## Definir os parâmetros

Os parâmetros de data augmentation são argumentos comuns de `train()`.

<code-tabs name="train" />

Três deles têm grafias mais curtas, aceitas tanto por `train()` quanto pela CLI:
`mosaic` corresponde a `mosaic_prob`, `fliplr` a `flip_prob` e, na detecção,
`mixup` corresponde a `mixup_prob`. Passar uma grafia curta e o seu campo com
valores diferentes levanta exceção. Na versão 1.8.0, `perspective` não tem flag
na CLI, então ele é definido em Python.

Os parâmetros de classificação são validados quando o pipeline deles é montado,
e a CLI também confere `flip_prob`, `flipud`, `cutmix` e `erasing`. O
`TrainConfig` não verifica a faixa dos parâmetros de detecção, então
`degrees=100` treina com rotações de até 100 graus. A [matriz de data
augmentation](/docs/reference/augmentation-matrix) dá a faixa de cada parâmetro.

## Três estados, não dois

Se um parâmetro tem algum efeito depende da família. A biblioteca mantém uma
tabela declarativa disso, e cada entrada é um de três estados.

`used` significa que o parâmetro chega ao pipeline e altera as amostras.
`ignored` significa que ele nunca chega ao pipeline, então defini-lo não faz
nada. `gated_by_mosaic` significa que ele só se aplica às amostras que passaram
pelo ramo do mosaico, então com `mosaic_prob=0` ele nunca dispara, mesmo estando
ligado.

Esse terceiro estado é o que pega as pessoas de surpresa. Em um pipeline no
estilo YOLOX, a transformação afim roda sobre o canvas do mosaico e o MixUp
mistura uma amostra de mosaico, então `mosaic_prob=0` desativa silenciosamente
`degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` e
`mixup_scale` de uma vez só. O treinador registra um aviso especificamente para
o caso do MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

A CLI também avisa sobre os parâmetros ignorados, listando só os que você
realmente digitou:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Chamado a partir do Python, `train()` não avisa sobre os parâmetros ignorados. A
[matriz de data augmentation](/docs/reference/augmentation-matrix) traz o estado de cada
parâmetro em cada família e o valor de que cada família parte. A tabela é a spec
que vem em `libreyolo/data/augment/spec.py`, que os próprios testes da
biblioteca verificam contra os pipelines, e ela pode ser lida diretamente:

<code-tabs name="support" />

## O que cada parâmetro faz

As figuras abaixo são produzidas pelas próprias transformações de treinamento do
LibreYOLO 1.8.0, executadas na imagem de exemplo por meio de um dataset de uma
única imagem, então cada peça do mosaico e cada imagem parceira do MixUp é a
mesma foto. O bounding box é o da pessoa que o LibreYOLO9t detecta nessa imagem.
Abaixo de cada painel ficam a configuração e o tamanho do maior bounding box, em
pixels na resolução de treinamento de 640 pixels. Sem data augmentation, a
pessoa mede 291 × 217 px.

Vários parâmetros geométricos são faixas: `degrees=10` significa um ângulo
sorteado em qualquer ponto de -10 a +10 graus. Nos painéis deles, o sorteio fica
fixo no valor da legenda, então eles mostram o efeito mais forte que uma
configuração permite, e não uma amostra típica. As figuras de detecção vêm do
pipeline do YOLOv9. O YOLOv7 treina por ele sem alterações, e o RT-DETR e o
RT-DETRv2 usam o wrapper de mosaico dele com uma transformação por amostra
própria. O YOLOX, o RTMDet, o PicoDet e o FOMO montam o mosaico e a
transformação da mesma forma em um wrapper separado, cujo MixUp é diferente.

### mosaic_prob

A probabilidade de uma amostra ser um mosaico: quatro imagens colocadas em um
canvas com o dobro do tamanho de entrada, que a transformação afim depois reduz
ao tamanho de entrada. Nos pipelines no estilo YOLOX, a transformação e o MixUp
rodam só em amostras de mosaico, então os dois param quando este valor é 0. O
RT-DETR, o RT-DETRv2, o PicoDet e o FOMO vêm com ele em 0.

<aug-figure name="mosaic" />

O segundo painel fixa a escala da transformação em 0.5×, em que o canvas inteiro
cabe. Os dois últimos são sorteios comuns com os padrões do YOLOv9, e a legenda
dá a escala que cada um sorteou.

### mosaic_scale

A faixa de onde é sorteado o zoom da transformação. Em 1× um objeto mantém o
tamanho que tem na inferência, e em 0.5× fica com a metade dele. O padrão base é
`(0.1, 2.0)`, que o YOLOX, o YOLOv7, o RTMDet, o PicoDet, o RT-DETR, o RT-DETRv2
e o FOMO mantêm, então os objetos deles podem encolher a um décimo do tamanho. O
YOLOv9 vem com `(0.5, 1.5)`. O YOLO-NAS e o PP-YOLOE não têm mosaico e
reaproveitam essa faixa como escala da transformação afim por amostra deles.

<aug-figure name="mosaic_scale" />

Acima de 1×, a saída mostra só a parte superior esquerda do canvas, então os
objetos são cortados na borda ou saem do quadro.

### degrees

A faixa de rotação: cada amostra transformada gira um ângulo sorteado de
`-degrees` a `+degrees`. A rotação tem como pivô o canto superior esquerdo do
canvas, então um ângulo grande joga boa parte da imagem para fora do quadro. O
YOLOX e o YOLOv7 mantêm o padrão base de 10, e o YOLOv9 vem com 0.

<aug-figure name="degrees" />

### translate

A faixa de deslocamento, como fração do lado da entrada: cada amostra
transformada se move até `translate` vezes 640 px em cada eixo com `imgsz=640`.
O padrão base é 0.1. O YOLO-NAS e o PP-YOLOE vêm com 0.25, e o RTMDet, o PicoDet
e o FOMO com 0.

<aug-figure name="translate" />

### shear

A faixa de cisalhamento em graus, sorteada separadamente para cada eixo. O YOLOX
e o YOLOv7 mantêm o padrão base de 2, e todas as outras famílias que o leem vêm
com 0.

<aug-figure name="shear" />

### perspective

Uma transformação projetiva por cima da rotação, da escala, do cisalhamento e da
translação. Cada canto do canvas se move para dentro por uma quantidade
aleatória de até `min(perspective * 100, 0.2)` vezes metade do lado do canvas,
por eixo, então `perspective=0.001` move um canto em no máximo 5 por cento do
lado em qualquer resolução, e todo valor de `0.002` para cima chega ao mesmo
máximo. O padrão `0` mantém a transformação afim, e um valor negativo levanta
`ValueError` quando a primeira amostra é transformada. Os painéis compartilham
um único sorteio aleatório, escalado pelo valor. O YOLOv9 e suas variantes E2E e
P2, o YOLOX, o YOLOv7, o YOLO-NAS, o PP-YOLOE, o RTMDet, o PicoDet, o RT-DETR e o
RT-DETRv2 a aplicam.

<aug-figure name="perspective" />

### hsv_prob

A probabilidade de um jitter de cor HSV. Só a probabilidade é um parâmetro. O
jitter desloca o matiz em até 5 na escala de matiz de 0 a 179 do OpenCV, e a
saturação e o valor em até 30 na escala de 0 a 255, e cada um dos três
deslocamentos é pulado com probabilidade de um meio. Todo pipeline de detecção
que lê `hsv_prob` usa essas quantidades. Os três últimos painéis são os mais
visíveis de 119 sorteios.

<aug-figure name="hsv" />

### flip_prob e flipud

As probabilidades de flip horizontal e vertical. `flip_prob` começa em 0.5 em
todas as famílias, exceto no FOMO, que vem com 0. `flipud` começa em 0. O
RTMDet, o PicoDet, o RT-DETR, o RT-DETRv2, o FOMO, o PP-YOLOE e as famílias no
estilo DETR não têm flip vertical e o ignoram. A receita do PP-YOLOE usa uma
rotação aleatória de 90 graus no lugar.

<aug-figure name="flip" />

### mixup_prob e mixup_scale

O MixUp mistura uma segunda amostra na atual e mantém os bounding boxes das
duas. No pipeline do YOLOv9, a parceira é uma amostra inteira sem mosaico, a
razão de mistura é sorteada uniformemente entre 0 e 1, e `mixup_scale` nunca é
lido, embora a spec da biblioteca o liste como condicionado pelo mosaico. No
wrapper do YOLOX, usado pelo YOLOX, pelo RTMDet, pelo PicoDet e pelo FOMO, e no
do YOLO-NAS, a parceira é primeiro reescalada por um fator sorteado de
`mixup_scale` e as duas imagens são combinadas pela média, com uma razão fixa de
0.5. O padrão base de `mixup_prob` é 1.0, e o YOLOv9 vem com 0.

<aug-figure name="mixup" />

Os dois últimos painéis misturam uma segunda amostra em um mosaico a 0.5×. A
legenda deles dá a razão de mistura λ, o peso da amostra de mosaico.

### no_aug_epochs

`no_aug_epochs` é `used` em todo lugar, mas não significa a mesma coisa em todo
lugar. Nos pipelines de mosaico ele desliga o mosaico e o MixUp nas épocas
finais. Nos pipelines no estilo DETR ele interrompe o data augmentation
fotométrico, de zoom-out e de recorte, e molda a cauda do schedule. Nos
pipelines de classificação ele desativa auto-augmentation, erasing, MixUp e
CutMix, e o recorte e os flips permanecem.

## Receitas no estilo DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 e TinyFormer treinam com uma única
transformação do torchvision, sem mosaico e sem transformação afim, e o mesmo
vale para o EdgeCrafter (EC) na detecção. As três operações fortes dessa
transformação são constantes da receita, e não parâmetros:

- `RandomPhotometricDistort` aplica cada mudança de cor (brilho de 0.875 a
  1.125, contraste e saturação de 0.5 a 1.5, matiz ±0.05 e uma troca aleatória
  da ordem dos canais) com probabilidade 0.5.
- `RandomZoomOut`, com probabilidade 0.5, coloca a imagem em um canvas preto de
  até quatro vezes o lado dela, então os objetos podem encolher a um quarto do
  tamanho.
- `RandomIoUCrop`, aplicado com probabilidade 0.8, recorta uma região entre 0.3
  e 1.0 de cada lado, escolhida para manter pelo menos parte dos bounding boxes.

Depois dessas etapas, os bounding boxes com 1 px ou menos em qualquer um dos
lados são descartados. As três operações param na época de parada da família,
no mais tardar quando começam as `no_aug_epochs` épocas finais. Os únicos
parâmetros que esse pipeline lê são `flip_prob` e `no_aug_epochs`.

<aug-figure name="detr_ops" />

Três amostras do início de um treinamento do D-FINE, com as três operações
ativas:

<aug-figure name="detr_samples" />

O RF-DETR tem um pipeline próprio: um flip horizontal; depois, com probabilidade
`crop_resize_prob`, um redimensionamento do lado menor para 400, 500 ou 600 px e
um recorte quadrado aleatório de 384 a 600 px; e, por fim, um redimensionamento
direto para a resolução de treinamento quadrada, que estica a imagem em vez de
usar letterbox. `crop_resize_prob` tem padrão 0.5 e é definido em Python.

<aug-figure name="rfdetr_crop" />

## Objetos pequenos

`erasing` não se aplica à detecção. O Random Erasing pertence ao pacote de
classificação e nenhum pipeline de detecção tem uma etapa de erasing, então um
detector treina igual com qualquer valor dele, e não há nada para desligar.
Estas são as etapas que encolhem ou removem objetos pequenos:

- `mosaic_scale`. Nos pipelines de mosaico, o tamanho de um objeto é
  multiplicado pela escala sorteada, então o limite inferior define o menor
  tamanho que ele atinge. Com o padrão base `(0.1, 2.0)`, um objeto de 20 px pode
  sair com 2 px. Em seguida, o pipeline do YOLOv9 descarta os bounding boxes com
  2 px ou menos em qualquer um dos lados, e a transformação final dele, como a
  do YOLOX, descarta os de 1 px ou menos em qualquer um dos lados. Os pixels do
  objeto continuam na imagem, sem rótulo. Um limite inferior de 0.5, como no `(0.5, 1.5)` do YOLOv9,
  mantém todo objeto com metade do tamanho de inferência ou mais.
- `degrees`, `translate`, `shear` e `perspective` empurram os objetos para a
  borda, onde os bounding boxes deles são cortados e passam pelo mesmo filtro de
  tamanho.
- Nas receitas no estilo DETR, `RandomZoomOut` pode encolher os objetos a um
  quarto do tamanho e `RandomIoUCrop` pode deixá-los fora do recorte. Nenhum dos
  dois é um parâmetro, e ambos param na época de parada.
- O recorte do RF-DETR pode cortar objetos na borda dele. `crop_resize_prob=0`
  o desliga.
- A resolução de treinamento escala todos os objetos: o letterbox redimensiona
  uma imagem por um fator de `imgsz` dividido pelo lado maior dela, então com
  `imgsz=1280` todo objeto tem o dobro da largura e da altura que tem em 640. O
  [YOLOv9](/docs/models/yolov9) também tem uma variante P2 com uma cabeça de
  stride 4 para objetos pequenos.

<code-tabs name="small" />

## O pacote de classificação

Seis parâmetros atuam apenas no pipeline de classificação. As famílias de
detecção ignoram os seis.

<code-tabs name="classify" />

`scale` é a faixa de área do recorte aleatório. `scale=0.5` significa
`(0.5, 1.0)`, e um par explícito define os dois limites. Um limite inferior
pequeno pode deixar o objeto inteiramente fora do recorte, como no terceiro
painel.

<aug-figure name="cls_crop" />

`auto_augment` aceita `"randaugment"`, `"autoaugment"`, `"augmix"` ou `None`, e
aplica um sorteio dessa política a cada amostra.

<aug-figure name="auto_augment" />

`erasing` é a probabilidade do Random Erasing, em `[0, 1)`. Ele roda depois da
normalização e preenche com zeros um retângulo que cobre de 2 a 33 por cento da
imagem, o que, depois da normalização, corresponde à cor média do ImageNet.

<aug-figure name="erasing" />

`mixup` e `cutmix` são probabilidades por batch que produzem rótulos suaves. No
máximo um roda por batch, o MixUp primeiro, então os dois são aditivos e devem
somar no máximo 1. Nos painéis, A e B são dois recortes da mesma foto, e cada
legenda dá o rótulo suave com que a imagem misturada é treinada.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` e `cutmix` vêm desligados por padrão, então o
treinamento de classificação não muda a menos que você peça.
`train()` e a CLI direcionam `mixup` por tarefa: a classificação usa a mistura
de batches, enquanto a detecção usa `mixup_prob`.

`crop_pct=None` preserva a razão de redimensionamento de avaliação da família.
Uma substituição altera o recorte de avaliação da etapa de validação executada
durante o treinamento, enquanto a exportação mantém o pré-processamento nativo
da família.

A CLI expõe `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr` e
`flipud`. Na classificação, `mixup` significa mistura de batches e tem padrão
0.0. O flip horizontal tem padrão 0.5, e o vertical, 0.0.

`no_aug_epochs` desativa auto-augmentation, erasing, MixUp e CutMix nas épocas
finais, mantendo o recorte e os flips. As receitas de classificação usam 0 por
padrão para esse trecho final.

## Parâmetros específicos de cada família

Alguns parâmetros ficam na subclasse de config de uma família em vez de na
classe base, então existem só para aquela família e não têm flag na CLI.

| Família | Parâmetro | Efeito |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilidade do data augmentation de instâncias copy-paste, só para `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reaproveita a mesma amostra espelhada, `"mixup"` puxa uma segunda amostra |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilidade de rotação aleatória de 90 graus |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Limite de ground truth por imagem nas transformações de treinamento, 300 por padrão |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste para `task="segment"`, só no modo `"flip"` |
| RF-DETR | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios no pipeline nativo, 0.5 por padrão |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios, só para `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilidades do jitter do caminho de pose e da transformação afim ciente dos keypoints |

`max_labels` é o que perde dados. Os bounding boxes que passam do limite são
descartados com um aviso único em vez de um erro, então imagens densas, como as
de fotografia aérea, precisam que ele seja aumentado. A memória da atribuição de
rótulos acompanha a imagem mais densa do batch, não o limite.

O mosaico e o MixUp ficam desativados no treinamento com caixas orientadas,
independentemente dos parâmetros, porque o data augmentation ciente dos cantos
para caixas rotacionadas não está implementado.

## Relacionados

- [Matriz de data augmentation](/docs/reference/augmentation-matrix) para o
  estado de cada família em cada parâmetro, as faixas e os padrões de cada
  família.
- [Hiperparâmetros](/docs/train/hyperparameters) para `no_aug_epochs` como
  argumento do schedule e o resto de `train()`.
- [Datasets](/docs/train/datasets) para os formatos de rótulo que essas
  transformações consomem.
