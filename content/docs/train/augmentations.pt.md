---
title: Data augmentation
seo_title: Data augmentation de treinamento no LibreYOLO
description: >-
  Os parâmetros de data augmentation do TrainConfig, os quatro formatos de
  pipeline por trás deles e a tabela por família que diz quais parâmetros são
  usados, condicionados ou ignorados.
lead: >-
  O data augmentation é configurado por parâmetros do TrainConfig, mas cada
  família de modelos roda seu próprio pipeline de treinamento, e um pipeline que
  não tem ramo de mosaico ignora mosaic_prob em vez de aproximá-lo.
keywords:
  - data augmentation yolo
  - aumento de dados yolo
  - mosaic augmentation
  - mixup
  - jitter hsv
  - transformação afim aleatória
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
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
source_hash: 47461cd13aab580c
---

## Definir os parâmetros

Os parâmetros de data augmentation são argumentos comuns de `train()`.

<code-tabs name="train" />

Dois deles têm uma grafia mais curta na CLI: `mosaic` corresponde a `mosaic_prob`
e `mixup` corresponde a `mixup_prob`. Todos os outros parâmetros se escrevem
igual nos dois lugares.

## Três estados, não dois

Se um parâmetro tem algum efeito depende da família. A biblioteca mantém
uma tabela declarativa disso, e cada entrada é um de três estados.

`used` significa que o parâmetro chega ao pipeline e altera as amostras.
`ignored` significa que ele nunca chega ao pipeline, então defini-lo não faz
nada. `gated_by_mosaic` significa que ele só se aplica às amostras que passaram
pelo ramo do mosaico, então com `mosaic_prob=0` ele nunca dispara, mesmo estando
ligado.

Esse terceiro estado é o que pega as pessoas de surpresa. Em um pipeline no
estilo YOLOX a transformação afim roda sobre o canvas do mosaico e o MixUp
mistura uma amostra de mosaico, então `mosaic_prob=0` desativa silenciosamente
`degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` e
`mixup_scale` de uma vez só. O treinador registra um aviso especificamente para o
caso do MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

A CLI também avisa sobre os parâmetros ignorados, listando só os que você
realmente digitou:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Quatro formatos de pipeline

As famílias se agrupam em quatro pipelines de treinamento, e o pipeline
determina quase todas as respostas.

O pipeline de mosaico no estilo YOLOX aplica o jitter HSV e os flips por amostra
e depois roda a transformação afim e o MixUp dentro do ramo do mosaico. Ele cobre
YOLOX, YOLOv7, YOLOv9 e suas variantes E2E e P2, RTMDet, PicoDet, RT-DETR,
RT-DETRv2 e FOMO.

O pipeline de passagem direta no estilo DETR não tem mosaico nem transformação
afim. A distorção fotométrica, o zoom-out e o recorte por IoU dele são constantes
da receita e não parâmetros de configuração, então só `flip_prob` e
`no_aug_epochs` ficam ativos. Ele cobre D-FINE, Dome-DETR, DEIM, DEIMv2,
RT-DETRv4, EC e, com uma mudança, RF-DETR.

O pipeline de classificação com ImageFolder ignora todos os parâmetros de
detecção. O flip horizontal dele é um 0.5 fixo que `flip_prob` não alcança. Em
vez disso, ele tem seu próprio pacote de parâmetros, descrito abaixo.

O YOLO-NAS é um formato à parte: nada de mosaico, uma transformação afim por
amostra sempre ligada e o MixUp aplicado de forma independente em vez de
condicionado. O valor de `mosaic_scale` dele é reaproveitado como faixa de escala
dessa transformação afim.

SegFormer e NAFNet rodam cada um um pipeline específico da tarefa, cuja
aleatoriedade está fixada na família em vez de ser configurável. No SegFormer os
parâmetros ativos são os atributos de classe `semantic_scale_jitter` e
`semantic_hsv_prob`, não `mosaic_scale` e `hsv_prob`. O recorte e os flips do
NAFNet são operações acopladas de entrada e alvo com uma probabilidade fixa de
0.5.

## Qual família respeita qual parâmetro

A tabela abaixo é a spec que vem em
`libreyolo/data/augment/spec.py`, verificada contra a ligação real do pipeline
pelos próprios testes da biblioteca. Consulte a spec ali em vez de deduzi-la pela
arquitetura.

<code-tabs name="support" />

Resumido por pipeline, para os parâmetros base:

| Parâmetro | Estilo YOLOX | YOLO-NAS | Estilo DETR | Classificação |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | condicionado pelo mosaico | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | condicionado pelo mosaico | used | ignored | ignored |
| `translate` | condicionado pelo mosaico | used | ignored | ignored |
| `shear` | condicionado pelo mosaico | used | ignored | ignored |
| `perspective` | condicionado pelo mosaico | used | ignored | ignored |
| `mosaic_scale` | condicionado pelo mosaico | used | ignored | ignored |
| `mixup_scale` | condicionado pelo mosaico | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

Exceções dentro dessas colunas, todas elas restritivas:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 e FOMO não têm flip vertical, então
  `flipud` é ignorado. O wrapper de mosaico do FOMO também é construído sem
  perspectiva.
- O pipeline nativo do RF-DETR não tem jitter HSV, então `hsv_prob` é ignorado
  além do que diz a coluna do estilo DETR.
- O EC respeita `hsv_prob`, `degrees` e `translate`, mas só para `task="pose"`,
  cuja transformação ciente dos keypoints os lê. Os caminhos de detect e segment
  dele usam receitas fotométricas fixas.
- O DINOv2 segue a coluna do estilo DETR para as tarefas detect e semantic e
  acrescenta o pacote de classificação para `task="classify"`.

`no_aug_epochs` é `used` em todo lugar, mas não significa a mesma coisa em todo
lugar. Nos pipelines de mosaico ele desliga o mosaico e o MixUp nas épocas
finais. Nos pipelines no estilo DETR ele interrompe o data augmentation
fotométrico, o zoom-out e o recorte, e molda a cauda do schedule. Nos pipelines
de classificação e de segmentação semântica ele só molda a cauda.

## O pacote de classificação

Quatro parâmetros governam o pipeline de classificação e mais nada. As famílias
de detecção ignoram os quatro.

<code-tabs name="classify" />

`auto_augment` aceita `"randaugment"`, `"autoaugment"`, `"augmix"` ou `None`.
`erasing` é a probabilidade do RandomErasing. `mixup` e `cutmix` são
probabilidades por batch que produzem rótulos suaves; no máximo um roda por
batch, o MixUp primeiro, então os dois são aditivos e a soma deles não deveria
passar de 1.

Os quatro vêm desligados por padrão, então o treinamento de classificação não
muda a menos que você peça.

Vale dizer com todas as letras que existe uma colisão de nomes: na CLI, `mixup` é
o alias do `mixup_prob` de detecção. O campo `mixup` de classificação não tem
grafia própria na CLI e só é alcançável por `model.train(mixup=...)` no Python.

## Parâmetros específicos de cada família

Alguns parâmetros ficam na subclasse de config de uma família em vez de na classe
base, então existem só para aquela família e não têm flag na CLI.

| Família | Parâmetro | Efeito |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilidade do data augmentation de instâncias copy-paste, só para `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reaproveita a mesma amostra espelhada, `"mixup"` puxa uma segunda amostra |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilidade de rotação aleatória de 90 graus |
| YOLOv9 | `max_labels` | Limite de ground truth por imagem nas transformações de treinamento, 100 por padrão |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste para `task="segment"`, só no modo `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilidades do jitter do caminho de pose e da transformação afim ciente dos keypoints |

`max_labels` é o que perde dados em silêncio. Os bounding boxes que passam do
limite são descartados sem erro, então imagens densas como as de fotografia aérea
precisam que ele seja aumentado.

O mosaico e o MixUp ficam desativados no treinamento com caixas orientadas
independentemente dos parâmetros, porque o data augmentation ciente dos cantos
para caixas rotacionadas não está implementado.

## Relacionados

- [Hiperparâmetros](/docs/train/hyperparameters) para `no_aug_epochs` como
  argumento do schedule e o resto de `train()`.
- [Datasets](/docs/train/datasets) para os formatos de label que essas transformações consomem.
