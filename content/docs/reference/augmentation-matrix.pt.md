---
title: Matriz de data augmentation
seo_title: Qual família do LibreYOLO respeita qual parâmetro de data augmentation
description: >-
  Suporte a parâmetros de data augmentation por família: os dezesseis parâmetros
  do TrainConfig, os três status, os seis arquétipos de pipeline e os parâmetros
  que uma família ignora em silêncio.
lead: >-
  Definir um parâmetro de data augmentation não garante que ele chegue ao
  pipeline. Esta página registra como cada família treinável trata cada
  parâmetro do TrainConfig, usando a tabela declarativa que a biblioteca entrega
  como sua única fonte de verdade.
keywords:
  - data augmentation libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - matriz de suporte a data augmentation
  - parâmetros do TrainConfig
last_verified: 1.6.0
verification: >-
  A lista de parâmetros, os status, os arquétipos, os desvios por família e as
  funções helper foram lidos de libreyolo/data/augment/spec.py na v1.6.0. Essa
  tabela está ancorada nos pipelines reais por tests/unit/test_augment_spec.py.
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
source_hash: f3cba41ceadf131f
---

## Os parâmetros

Estes são nomes de campos do `TrainConfig`, não a grafia da CLI. A CLI mapeia
seus próprios aliases sobre eles, então `--mosaic` define `mosaic_prob`.

| Parâmetro | Significado |
|---|---|
| `mosaic_prob` | Probabilidade de montar uma amostra em mosaico de 4 imagens |
| `mixup_prob` | Probabilidade de mesclar uma segunda amostra |
| `hsv_prob` | Probabilidade de jitter de cor HSV |
| `flip_prob` | Probabilidade de flip horizontal |
| `degrees` | Faixa de rotação aleatória para o warp afim, em graus |
| `translate` | Fração de translação aleatória para o warp afim |
| `mosaic_scale` | Faixa de escala aleatória para o warp afim |
| `mixup_scale` | Faixa de escala de jitter aplicada à imagem parceira do MixUp |
| `shear` | Faixa de cisalhamento aleatório para o warp afim, em graus |
| `perspective` | Magnitude do warp projetivo para o warp afim |
| `flipud` | Probabilidade de flip vertical |
| `no_aug_epochs` | Épocas finais treinadas com o data augmentation forte desativado |
| `auto_augment` | Política AutoAugment de classificação: randaugment, autoaugment ou augmix |
| `erasing` | Probabilidade de RandomErasing na classificação |
| `mixup` | Probabilidade de MixUp por batch na classificação, com soft labels |
| `cutmix` | Probabilidade de CutMix por batch na classificação, com soft labels |

Os quatro últimos formam os controles de classificação. As famílias de detecção os ignoram. A CLI direciona `mixup` para a mistura de lotes nos classificadores e para `mixup_prob` nos detectores.

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

## Arquétipos de pipeline

Toda família coberta segue um de seis pipelines, com um punhado de desvios por
família listados abaixo.

| Parâmetro | Estilo YOLOX | YOLO-NAS | Estilo DETR | Classificação | Semântica | Restauração |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | used | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

No pipeline no estilo YOLOX, o pré-processamento por amostra aplica o jitter HSV
e os flips, enquanto o warp afim e o MixUp rodam apenas dentro do ramo do
mosaico. Já o YOLO-NAS roda um afim por amostra que está sempre ligado, ignora o
mosaico e aplica o MixUp de forma independente, reaproveitando `mosaic_scale`
como faixa de escala do afim.

O pipeline do tipo DETR usa uma transformação sem mosaic. Distorção fotométrica, zoom-out e recorte por IoU são constantes da receita, não parâmetros configuráveis; por isso, `hsv_prob` e os controles geométricos não os alteram. A classificação usa `flip_prob` para inversões horizontais e `flipud` para verticais. A variação de escala e HSV da segmentação semântica vem de atributos da classe do modelo; na restauração, entrada e alvo são invertidos juntos com probabilidade fixa de 0.5.

`no_aug_epochs` é respeitado em todos os pipelines, mas desativa operações diferentes: mosaic e MixUp nos pipelines do tipo YOLOX; transformação afim e MixUp no YOLO-NAS; aumentos fotométricos fortes e recorte, com a parte final do cronograma da taxa de aprendizado, no DETR; aumento automático, erasing, MixUp e CutMix na classificação. Os recortes e as inversões da classificação continuam ativos.

## Famílias por arquétipo

| Arquétipo | Famílias |
|---|---|
| Estilo YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Estilo DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Classificação | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semântica | `segformer` |
| Restauração | `nafnet` |

Vinte e cinco famílias são cobertas. Uma família fora desta lista retorna um
conjunto de ignorados vazio, então nenhum aviso é emitido para ela.

## Desvios

| Família | Diferença em relação ao seu arquétipo |
|---|---|
| `rtmdet` | `flipud` ignorado: seu transform não tem flip vertical |
| `picodet` | `flipud` ignorado |
| `rtdetr` | `flipud` ignorado |
| `rtdetrv2` | `flipud` ignorado |
| `fomo` | `perspective` e `flipud` ignorados |
| `ec` | `hsv_prob`, `degrees` e `translate` usados, apenas para `task="pose"`; detect e segment usam receitas fotométricas fixas |
| `dinov2` | O pacote de classificação é usado, apenas para `task="classify"` |

`ec` e `dinov2` são famílias multitarefa, então um parâmetro só é marcado como
ignorado quando todas as tarefas treináveis da família o ignoram. Isso impede que
o aviso da CLI seja errado para uma tarefa enquanto está certo para outra.

O Dome-DETR herda os transforms do D-FINE sem alterações. A única coisa que ele
não aceita é o treinamento multiescala, que sua configuração desativa, e não a
spec de augmentation.

## Parâmetros específicos de família

Algumas famílias carregam parâmetros de data augmentation na sua própria
subclasse de `TrainConfig`, e não na base. A CLI não expõe esses parâmetros;
defina-os pela API Python.

| Família | Parâmetro | Significado |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilidade do augmentation de instâncias copy-paste, apenas `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Origem do copy-paste: `flip` espelha a mesma amostra, `mixup` usa uma segunda amostra |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilidade de rotação aleatória de 90 graus |
| `rfdetr` | `copy_paste` | Probabilidade de copy-paste para `task="segment"`, apenas no modo `flip` |
| `rfdetr` | `copy_paste_mode` | Modo de origem do copy-paste para `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilidade de crop-resize aleatório no pipeline nativo |
| `dfine` | `crop_resize_prob` | Probabilidade de crop-resize aleatório, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilidade de crop-resize aleatório, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilidade de jitter de brilho e contraste, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilidade de afim ciente de keypoints, `task="pose"` |

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
que um parâmetro é ignorado ou condicionado para aquela família.

## O gate do mosaico

Para uma família no estilo YOLOX, `mixup_prob=0.5` com `mosaic_prob=0` desativa o
MixUp por completo, porque o MixUp se aplica apenas a amostras de mosaico. Essa
combinação é fácil de alcançar ao desligar o mosaico no fim do treinamento. O
treinador registra um aviso nomeando a família, e `mixup_gating_warning` é a
função pura por trás dele.
