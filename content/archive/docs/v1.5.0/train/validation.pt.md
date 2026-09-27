---
title: Validação e métricas
seo_title: Validação e métricas no LibreYOLO
description: >-
  Rode val() em qualquer modelo, leia as chaves de métrica que cada tarefa
  retorna, escolha um backend de avaliação e ligue uma loss de validação junto
  com a métrica de acurácia.
lead: >-
  A validação roda um modelo sobre um split do dataset através do val() e
  retorna um dicionário plano de chaves de métrica e valores float. As chaves
  são strings literais, e quais delas você recebe depende da tarefa, não da
  família.
keywords:
  - map50-95
  - avaliação coco
  - métricas de validação yolo
  - faster-coco-eval
  - pycocotools
  - loss de validação
  - miou segmentação semântica
  - panoptic quality
  - acurácia top1
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Em outro split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Gravar predições no formato COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Rodar uma validação

`val()` recebe o dataset e retorna as métricas.

<code-tabs name="val" />

O valor de retorno é um `dict[str, float]` simples. Toda chave é literal, então
leia pelo nome, e não pela posição.

Os argumentos principais são `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` e `verbose`. `conf` tem padrão
`0.001` e `iou`, `0.6`, ambos bem mais frouxos que os padrões de predição,
porque uma varredura de mAP precisa da cauda de baixa confiança. `imgsz` assume
por padrão o tamanho de entrada do próprio modelo, em vez de um número fixo.
`split` aceita `val`, `test` ou `train`, e nada mais.

Qualquer outro campo da configuração de validação passa adiante como argumento
nomeado, incluindo `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`,
`cache` e `save_plots`.

## Chaves de métrica por tarefa

A detecção retorna a família de números do COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Duas delas são armadilhas. `metrics/precision` e `metrics/recall` são aliases
mantidos por compatibilidade retroativa: eles carregam os valores de mAP 50-95 e
AR@100, não um par de precisão e recall. Use as chaves nomeadas.

A segmentação de instâncias retorna os números de mAP e AR acima como números de
máscara nas chaves sem sufixo, com as versões de box sob o sufixo `(B)` e as
versões de máscara repetidas sob `(M)`. Precisão e recall existem apenas na forma
sufixada nessa tarefa, como `metrics/precision(B)`/`metrics/recall(B)` e
`metrics/precision(M)`/`metrics/recall(M)`, e os dois pares carregam os mesmos
valores de alias que os de detect: o par `(B)` é o mAP50-95 de box e o AR@100
de box, o par `(M)` é o mAP50-95 de máscara e o AR@100 de máscara.

| Tarefa | Chaves |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, mais os detalhamentos por tamanho e de recall acima |
| segment | versões de máscara das chaves de detect acima (as chaves sem sufixo são as de máscara); `precision`/`recall` existem apenas como `(B)`/`(M)`, os dois com o mesmo alias |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, e as chaves `keypoints_AR` correspondentes |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, mais cópias sufixadas com `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, mais uma chave de varredura de mAP |

`metrics/precision` e `metrics/recall` do OBB não são aliases: são a precisão e
o recall reais em IoU 0.50, tomados no ponto de operação mais frouxo (toda
predição que sobrevive ao `conf`, padrão `0.001`). As cópias sufixadas com
`(OBB)` repetem os mesmos quatro valores sob um nome específico da tarefa, a
mesma convenção de `(B)` e `(M)` acima.

`accuracy_top5` é na verdade top-`min(5, num_classes)`, então em um dataset de
três classes ele é top-3, que toda amostra satisfaz e que, portanto, marca 1.0.

A chave de varredura da tarefa point é montada a partir dos limiares de
distância, então com os padrões ela fica `metrics/mAP@[0.01:0.10]` e a chave de
limiar único fica `metrics/mAP@0.01`. Passar `dist_thresholds` muda as duas
strings.

A maioria das tarefas também retorna uma chave `fitness`, o número único que a
seleção do melhor checkpoint usa por padrão. Detecção, segmentação e OBB não
trazem uma; suas famílias são selecionadas por `metrics/mAP50-95`, que os
dicionários delas de fato retornam. Pose não retorna nem `fitness` nem
`metrics/mAP50-95`; seus treinadores definem `best_metric_key` como
`metrics/keypoints_mAP50-95` no lugar.

## Chaves de velocidade

Todo validador adiciona medições de tempo:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

São milissegundos por imagem, na média da execução. Eles descrevem a máquina e as
configurações em que você rodou, então um número tirado dali só faz sentido
quando reportado junto com o hardware, o tamanho de batch e a precisão.

## Backend de avaliação

As métricas de detecção e segmentação são calculadas por um avaliador COCO, e
`faster_coco_eval=True`, o padrão, seleciona o backend em C++ quando o pacote
`faster-coco-eval` está instalado. Quando não está, a execução recorre ao
pycocotools com um aviso por processo:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Qual backend realmente rodou fica registrado no modelo como `last_eval_backend`,
e a CLI reporta isso na saída para tarefas do estilo detecção. Defina
`LIBREYOLO_FASTER_COCO_EVAL` para sobrescrever o valor da configuração a partir
do ambiente.

`iou_thresholds` só é respeitado no caminho do OBB. O caminho COCO avalia pela
sua própria varredura fixa de 0.50 a 0.95 e ignora o valor.

## Loss de validação

Por padrão a validação reporta apenas acurácia. `val_loss=True` também calcula o
objetivo de treinamento da família sobre os batches de validação.

<code-tabs name="valloss" />

Ele emite `metrics/loss` mais um `metrics/loss/<component>` por termo,
ponderados exatamente como o treinamento os pondera, de modo que os componentes
somam o total. Em um logger, eles aparecem como `val/loss` e
`val/loss/<component>`, e o `libreyolo monitor` sobrepõe `metrics/loss` a
`train/loss`.

Os componentes são os da própria família:

| Tarefa | Famílias | Componentes |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Vem desligado por padrão porque a atribuição de alvos adiciona tempo e memória à
validação. O validador reaproveita a saída do modelo já produzida para a métrica
de acurácia em vez de rodar um segundo forward, roda sob `no_grad` no modelo de
avaliação ou de EMA, e em treinamento multi-GPU é calculado localmente no rank 0,
sem operações coletivas. A seleção do melhor checkpoint continua na métrica de acurácia.

Três coisas que ele deliberadamente não faz. Nunca inclui termos de contrastive
denoising, porque esses precisam do ground truth na hora do forward e a validação
faz o forward sem ele. Ele reporta o modelo em modo de avaliação, então onde o
forward de treinamento e o de avaliação de uma família realmente diferem, em
estatísticas de BatchNorm ou stochastic depth, o número reflete o modo de
avaliação; essa é a comparação pretendida. E uma tarefa para a qual uma família
não implementou isso gera um erro de configuração já na inicialização, em vez de
pular silenciosamente:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

O FOMO é a exceção que não muda nada: seu validador sempre calculou essa loss, e
`val_loss=True` só afeta sob quais chaves ela é publicada.

Validação com augmentation e loss de validação não podem ser combinadas, e pedir
as duas gera um erro.

## Arquivos que uma validação escreve

`val()` sempre escreve `config.yaml` no seu diretório de saída, que por padrão é
`runs/val/<model>_<size>_<timestamp>` quando `save_dir` não é informado.

<code-tabs name="json" />

`save_json=True` escreve `predictions.json` para detecção, e
`predictions_bbox.json` mais `predictions_masks.json` para segmentação. OBB não
suporta e avisa.

`save_plots=True` escreve em um subdiretório `plots/`. A detecção ganha
`box_metrics.png`, gráficos de AP e recall por classe, curvas de precisão-recall
e de confiança, uma matriz de confusão e imagens de exemplo anotadas quando o
OpenCV está instalado. A segmentação adiciona as cópias do lado da máscara de
cada um, e pose ganha seu próprio conjunto de métricas e curvas. Os outros
validadores não implementam gráficos; classificação, semantic, panoptic, depth,
normal, edge, restore, matte, OCR, OBB e point não escrevem nada ali. Uma falha
ao plotar emite um aviso e nunca aborta a execução.

## Validação durante o treinamento

O treinamento valida a cada `eval_interval` épocas no split `val` do dataset, e
as métricas que ele produz são as que determinam a seleção do `best.pt`, o
early stopping por `patience` e as chaves `val/` em todo logger. A validação
roda sobre os pesos de EMA quando o EMA está ligado.

Veja [Hiperparâmetros](/docs/train/hyperparameters) para `eval_interval`,
`patience` e `save_plots`, e [Loggers de experimentos](/docs/train/loggers) para
onde os números vão.

## Relacionado

- [Datasets](/docs/train/datasets) para as chaves de split e os formatos que os validadores leem.
