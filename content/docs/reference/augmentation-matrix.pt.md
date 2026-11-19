---
title: Matriz de data augmentation
seo_title: Qual família do LibreYOLO respeita qual data augmentation
description: >-
  Cada parâmetro de data augmentation com sua faixa e sua grafia na CLI, o
  status de cada parâmetro em cada uma das 31 famílias treináveis e os valores
  padrão com que cada família vem.
lead: >-
  Definir um parâmetro de data augmentation não garante que ele chegue ao
  pipeline. Esta página registra como cada família treinável trata cada
  parâmetro do TrainConfig, e o valor de que cada família parte, gerados a
  partir da biblioteca na versão que ela documenta.
keywords:
  - data augmentation libreyolo
  - parâmetros de data augmentation
  - valores padrão de data augmentation
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - matriz de suporte a data augmentation
  - parâmetros do TrainConfig
last_verified: 1.8.0
verification: >-
  Gerado por scripts/build-augmentation/collect.py a partir do LibreYOLO 1.8.0:
  a lista de parâmetros, os status e as notas, de
  libreyolo/data/augment/spec.py, que os testes da biblioteca ancoram nos
  pipelines reais; os valores padrão, da subclasse de TrainConfig de cada
  família; e as grafias da CLI, do comando libreyolo train. As faixas e o único
  ponto em que a spec e um pipeline divergem são verificados contra a biblioteca
  pelo mesmo script.
snippets:
  usage:
    - label: Consultar a spec diretamente
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: Os valores padrão de uma família
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Os parâmetros

Estes são nomes de campos do `TrainConfig`. `train()` e a CLI também aceitam as
grafias curtas da coluna CLI, então `mosaic` define `mosaic_prob`. Na CLI,
`mixup` é direcionado por tarefa: mistura de batches de classificação para os
classificadores, `mixup_prob` para os detectores. `perspective` não tem flag na
CLI e é definido em Python.

<aug-knobs />

O padrão base é o valor definido no próprio `TrainConfig`. A maioria das
famílias substitui alguns deles, e [Valores padrão por
família](#defaults-by-family) dá os valores de que um treinamento realmente
parte. Para ver imagens do que cada parâmetro faz, consulte [Data
augmentation](/docs/train/augmentations).

As faixas entre colchetes são intervalos, então `[0, 1)` inclui 0 e exclui 1. Os
parâmetros de classificação são verificados quando o pipeline deles é montado e
levantam `ValueError` fora da faixa. Os parâmetros de detecção não são
verificados pelo `TrainConfig`, então um valor fora da faixa treina em silêncio.
As exceções são uma escala sorteada de 0 ou menos e um `perspective` negativo,
que levantam `ValueError` quando uma amostra é transformada.

<code-tabs name="usage" />

## Os três status

| Status | Significado |
|---|---|
| `used` | O parâmetro chega ao pipeline de treinamento da família e altera as amostras |
| `gated_by_mosaic` | O parâmetro se aplica apenas às amostras que passaram pelo ramo do mosaico, então com `mosaic_prob == 0` ele nunca dispara |
| `ignored` | O parâmetro nunca chega ao pipeline; defini-lo não faz nada |

`ignored` é o que vale conferir antes de uma execução, porque nada falha. A CLI
avisa quando um parâmetro de treinamento definido explicitamente é um dos que a
família selecionada ignora, e o treinador avisa quando `mixup_prob > 0` não pode
disparar porque a família condiciona o MixUp ao mosaico e `mosaic_prob` é zero.

## Qual família respeita qual parâmetro

<aug-matrix />

Toda família coberta segue um de seis pipelines. No pipeline no estilo YOLOX, o
pré-processamento por amostra aplica o jitter HSV e os flips, enquanto a
transformação afim e o MixUp rodam apenas dentro do ramo do mosaico. Já o
YOLO-NAS roda uma transformação afim por amostra que está sempre ligada, ignora
o mosaico e aplica o MixUp de forma independente, reaproveitando `mosaic_scale`
como faixa de escala da transformação afim.

O pipeline no estilo DETR é uma transformação de passagem direta, sem mosaico. A
distorção fotométrica, o zoom-out e o recorte por IoU dele são constantes da
receita, e não parâmetros configuráveis, e é por isso que `hsv_prob` e os
parâmetros geométricos nunca chegam a ele. A classificação usa `flip_prob` para
os flips horizontais e `flipud` para os verticais. O jitter de escala e o HSV
das famílias semânticas vêm de atributos de classe da família, e não de
parâmetros de configuração, e os flips de restauração são operações acopladas de
entrada e alvo com uma probabilidade fixa de 0.5.

`no_aug_epochs` é respeitado em todo lugar, embora o que ele desliga varie:
mosaico e MixUp no estilo YOLOX, a transformação afim e o MixUp no YOLO-NAS, o
data augmentation forte fotométrico e de recorte, mais a cauda do learning rate,
no estilo DETR, e auto-augmentation, erasing, MixUp e CutMix na classificação. O
recorte e os flips da classificação continuam ativos.

Onde uma família difere do resto do seu pipeline, o motivo é um destes:

| Família | Diferença |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignorado: os transforms deles não têm flip vertical |
| `fomo` | `perspective` e `flipud` ignorados |
| `ppyoloe` | `flipud` ignorado: a receita de origem usa uma rotação aleatória de 90 graus no lugar |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignorado: o MixUp deles mistura imagens inteiras com uma razão aleatória, sem reescalar nenhuma das duas |
| `gtr` | `mosaic_prob` e `mixup_prob` usados, durante as primeiras `mosaic_epochs` épocas; `degrees`, `translate` e `mosaic_scale` moldam a transformação afim que segue o mosaico |
| `rfdetr` | `hsv_prob` ignorado: o pipeline nativo não tem jitter HSV |
| `ec` | `hsv_prob`, `degrees` e `translate` usados, apenas para `task="pose"`; detect e segment usam receitas fotométricas fixas |
| `dinov2` | O pacote de classificação é usado, apenas para `task="classify"` |

`ec` e `dinov2` são famílias multitarefa, então um parâmetro só é marcado como
ignorado quando todas as tarefas treináveis da família o ignoram. Isso impede que
o aviso da CLI seja errado para uma tarefa enquanto está certo para outra.

O Dome-DETR herda os transforms do D-FINE sem alterações. A única coisa que ele
não aceita é o treinamento multiescala, que a configuração dele desativa, e não
a spec de data augmentation.

Trinta e uma famílias são cobertas. Para uma família fora desta lista,
`ignored_aug_params` retorna um conjunto vazio, então nenhum aviso é emitido
para ela.

## Valores padrão por família

O valor de que parte a config de cada família de detecção. Uma célula vazia
significa que a família ignora esse parâmetro, então o valor padrão dele não
muda nada.

<aug-defaults />

Duas regularidades importam mais do que qualquer valor isolado. O RT-DETR, o
RT-DETRv2, o PicoDet e o FOMO começam com `mosaic_prob=0`, então nessas famílias
os parâmetros marcados como condicionados na matriz não fazem nada até o mosaico
ser ligado. O YOLOv9 começa com a rotação, o cisalhamento e o MixUp desligados e
com `mosaic_scale=(0.5, 1.5)`, enquanto o padrão base é `(0.1, 2.0)`.

O EdgeCrafter lê `hsv_prob`, `degrees` e `translate` apenas para `task="pose"`,
então a linha dele mostra os valores padrão da tarefa de pose para esses três
parâmetros.

As famílias de classificação partem dos padrões base: `flip_prob=0.5`,
`flipud=0.0`, `scale=(0.5, 1.0)`, o `crop_pct` próprio da família, e
`auto_augment`, `erasing`, `mixup` e `cutmix` desligados.

## Parâmetros específicos de família

Algumas famílias carregam parâmetros de data augmentation na sua própria
subclasse de `TrainConfig`, e não na base. A CLI não expõe esses parâmetros;
defina-os pela API Python.

| Família | Parâmetro | Significado |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilidade do data augmentation de instâncias copy-paste, apenas `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Origem do copy-paste: `flip` espelha a mesma amostra, `mixup` usa uma segunda amostra |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilidade de rotação aleatória de 90 graus |
| `rfdetr` | `copy_paste` | Probabilidade de copy-paste para `task="segment"`, apenas no modo `flip` |
| `rfdetr` | `copy_paste_mode` | Modo de origem do copy-paste para `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios no pipeline nativo |
| `dfine` | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilidade de recorte e redimensionamento aleatórios, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilidade de jitter de brilho e contraste, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilidade da transformação afim ciente de keypoints, `task="pose"` |

`rot90` se aplica a detect e OBB no `yolo9`.

## Consultando a spec

| Helper | Retorna |
|---|---|
| `aug_support(family)` | A tabela de parâmetro para `Support`, ou `None` para uma família desconhecida |
| `ignored_aug_params(family)` | O conjunto de nomes de parâmetros que a família ignora; vazio para uma família desconhecida |
| `uses_mosaic_gating(family)` | Se o MixUp da família dispara apenas em amostras de mosaico |
| `display_name(family)` | O nome da família voltado para humanos, usado nos avisos |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | O texto do aviso quando o MixUp nunca pode disparar, senão `None` |

Um `Support` é uma tupla nomeada de `status` e `note`, em que a nota explica por
que um parâmetro é ignorado ou condicionado para aquela família. Na 1.8.0, a
spec ainda lista `mixup_scale` como condicionado para as famílias que treinam
pelo wrapper de mosaico do YOLOv9, então `aug_support` o informa como
condicionado nelas, mesmo que a matriz acima, que é verificada contra o
pipeline, o marque como ignorado.

## O gate do mosaico

Para uma família no estilo YOLOX, `mixup_prob=0.5` com `mosaic_prob=0` desativa
o MixUp por completo, porque o MixUp se aplica apenas a amostras de mosaico.
Essa combinação é fácil de alcançar ao desligar o mosaico no fim do
treinamento. O treinador registra um aviso nomeando a família, e
`mixup_gating_warning` é a função pura por trás dele.
