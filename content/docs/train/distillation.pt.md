---
title: Destilação de conhecimento
seo_title: Destilação de conhecimento no LibreYOLO
description: >-
  Treine um detector pequeno contra um teacher maior ou contra um backbone
  DINOv2 congelado: as losses MGD, CWD e feature-MSE, os pontos de captura e o
  suporte por família.
lead: >-
  A destilação adiciona um segundo termo de loss que aproxima os mapas de
  características intermediários do student dos de um teacher congelado. O
  LibreYOLO captura as características com forward hooks, de modo que a cabeça e
  a loss do próprio teacher nunca entram em jogo.
keywords:
  - destilação de conhecimento
  - masked generative distillation
  - destilação channel-wise
  - destilação de características
  - teacher dinov2
  - treinamento teacher student
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Um checkpoint maior da mesma família supervisiona o menor.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Um ViT auto-supervisionado e congelado supervisiona um estágio do backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Ajustar a loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # peso global da destilação
            distill_tau=1.0,   # temperatura do softmax do CWD
        )
source_hash: 7210031328f6826f
---

## Destilar a partir de um checkpoint maior

Definir `distill_model` liga a destilação. O valor é um checkpoint de teacher,
carregado pela mesma factory que qualquer outro modelo.

<code-tabs name="detector" />

O teacher roda o forward sob `no_grad`, e sob autocast quando o AMP está ligado,
de modo que o modelo congelado não gasta computação em precisão total a cada
passo. Forward hooks capturam os mapas de características dele em pontos de
captura nomeados, a loss os compara com os do student, e o resultado é somado à
loss de treinamento e reportado como um componente chamado `distill`.

## Destilar a partir de um backbone fundacional congelado

Um ViT auto-supervisionado pode, em vez disso, supervisionar um único estágio do
backbone do student. As características do teacher vêm do extrator de
características dele mesmo, e não de hooks, e a loss cuida do descompasso entre
uma grade de patches e um stride convolucional.

<code-tabs name="foundation" />

`distill_model` reconhece `dinov2`, que é o DINOv2-base, além de `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`,
e qualquer id bruto do hub que comece com `facebook/dinov2`. Qualquer outra coisa
é tratada como o caminho de um checkpoint de teacher.

Esse caminho usa `feat_mse` independentemente de `distill_loss_type`, e precisa
do `transformers` instalado. Um teacher que carrega com chaves de pesos faltando
aborta em vez de destilar contra um backbone parcialmente aleatório.

## Quais famílias

O suporte a destilação é um método no modelo student, e há dois deles.

`get_distill_config()` fornece os pontos de captura multiescala que um teacher
detector supervisiona. YOLOv9, YOLOX e RF-DETR o implementam.

`get_backbone_distill_config()` fornece o único estágio do backbone que um
teacher fundacional supervisiona. O YOLOv9 o implementa, e é a única família que
faz isso.

Qualquer outra coisa levanta um erro em vez de treinar sem a loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Pontos de captura

Os pontos de captura são fixos por família e por papel, de modo que teacher e
student não precisam ser a mesma arquitetura; precisam de strides de
características que coincidam.

| Família | Papel | Pontos de captura | Strides |
|---|---|---|---|
| YOLOv9 | teacher ou student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | student fundacional | `backbone.elan3` | 16 |
| YOLOX | teacher ou student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher ou student | `model.backbone.0.projector.stages.0` | sondado na inicialização |

Strides que não coincidem levantam um erro antes de o treinamento começar:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Essa verificação é ignorada para teachers fundacionais, cuja razão de ser é
justamente o fato de as grades serem diferentes.

## As três losses

`distill_loss_type` seleciona a loss de características para um teacher detector.
Um teacher fundacional sempre usa `feat_mse`.

`mgd`, masked generative distillation, mascara uma fração das posições espaciais
do student e treina um pequeno gerador de duas convoluções para reconstruir o
mapa de características completo do teacher a partir do que sobrou.
`distill_mask_ratio` define a fração mascarada, 0.65 por padrão.

`cwd`, channel-wise distillation, transforma as ativações espaciais de cada canal
em uma distribuição de probabilidade e minimiza a divergência KL canal a canal.
`distill_tau` é a temperatura do softmax, 1.0 por padrão.

`feat_mse` alinha os canais do student aos do teacher com uma convolução 1x1,
redimensiona a grade do teacher para a do student de forma bilinear, e calcula o
erro quadrático médio. `distill_normalize=True` normaliza primeiro os dois mapas
de características com L2 sobre a dimensão de canais, o que torna a comparação
apenas angular e invariante à escala. O padrão é `False`.

`dis` é o peso global aplicado por cima. Se não for definido, cada loss usa o
próprio valor padrão publicado: 2e-5 para MGD, 1.0 para CWD e 1.0 para o feature
MSE. Eles diferem em cinco ordens de grandeza, então um peso ajustado para um
tipo de loss não significa nada para outro.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` e `distill_normalize` não têm flags de CLI.
São argumentos de Python ou chaves YAML de `cfg=`. O RF-DETR também é só Python
para a destilação como um todo, porque o mapeamento de argumentos de CLI dele não
carrega as chaves de destilação.

## Adaptadores, checkpoints e multi-GPU

Cada loss constrói pequenos módulos treináveis que vivem fora do student: os
adaptadores de canal 1x1, e o gerador do MGD. Eles recebem o próprio grupo de
parâmetros no otimizador, com o learning rate efetivo da execução.

Esses módulos são gravados no checkpoint sob uma chave `distiller` e restaurados
ao retomar, então uma execução retomada não reinicia seus projetores do zero.

Sob DDP os adaptadores ficam fora do student encapsulado, o que significa que o
reducer do DDP nunca vê os gradientes deles. O trainer faz um all-reduce
explícito deles a cada passo, de modo que todos os ranks treinam os mesmos
adaptadores.

A captura de CUDA graphs não está disponível em uma execução com destilação.
Passar `cuda_graph=True` registra uma linha no log e treina em modo eager. Veja
[Desempenho do treinamento](/docs/train/performance).

## Relacionado

- [Congelamento de camadas](/docs/train/layer-freezing) e
  [fine-tuning com LoRA](/docs/train/lora), nenhum dos quais é impedido de se
  combinar com a destilação.
- [Hiperparâmetros](/docs/train/hyperparameters) para o resto do `train()`.
