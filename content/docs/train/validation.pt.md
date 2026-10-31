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
  - matriz de confusão yolo
  - nms agnóstico a classes
last_verified: 1.6.0
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
  confusion:
    - label: Matriz de confusão
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")

        metrics = model.val(data="coco8.yaml")


        cm = metrics.confusion_matrix

        print(cm.matrix.shape)      # (81, 81): 80 classes mais background

        print(cm.class_accuracy())  # nome da classe: proporção dos objetos
        encontrados que receberam essa classe

        tp, fp = cm.tp_fp()
  curves:
    - label: Curvas por classe
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        box = metrics.box
        print(box.px.shape, box.f1_curve.shape)  # (1000,) e (classes, 1000)
        for index, f1 in zip(box.ap_class_index, box.f1):
            print(model.names[int(index)], round(float(f1), 3))
source_hash: a5704354163337cf
---

## Rodar uma validação

`val()` recebe o dataset e retorna as métricas.

<code-tabs name="val" />

O valor de retorno é um `dict` plano de chaves de métrica para números. Toda
chave é literal, então leia pelo nome, e não pela posição. Para detect, segment e
classify ele é uma subclasse de `dict` que também carrega resultados por classe
como atributos, descritos abaixo.

Os argumentos principais são `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` e `verbose`. `conf` tem padrão
`0.001` e `iou`, `0.6`, ambos bem mais frouxos que os padrões de predição,
porque uma varredura de mAP precisa da cauda de baixa confiança. `imgsz` assume
por padrão o tamanho de entrada do próprio modelo, em vez de um número fixo.
`split` aceita `val`, `test` ou `train`, e nada mais. Sem `data`, um checkpoint
treinado valida no dataset em que foi treinado; pesos publicados não carregam
nenhum e precisam de `data=`.

Qualquer outro campo da configuração de validação passa adiante como argumento
nomeado, incluindo `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache`,
`save_plots`, `single_cls` e `agnostic_nms`. Com `half=True` em CUDA, a validação
roda sob autocast em `amp_dtype`, e as saídas em bfloat16 são convertidas para
float32 antes do pós-processamento; o treinamento com `amp_dtype="bfloat16"`
valida do mesmo jeito.

`agnostic_nms=True` mantém só o box de maior score entre os boxes que se
sobrepõem acima de `iou`, quaisquer que sejam as classes. Ele roda sobre as
detecções já prontas da família, depois de `classes=`, então o mAP, a matriz de
confusão, as curvas e as contagens por imagem pontuam todos os mesmos boxes. Ele
cobre detect e segment, e qualquer outra tarefa levanta `ValueError`. O lado do
predict está em [Limiares e filtragem](/docs/predict/thresholds).

`single_cls=True` pontua a detecção como uma classe só. O ground truth é colapsado
para a classe 0, e as detecções do modelo são mantidas conforme `classes=` no
espaço de labels do próprio modelo, depois renomeadas para 0, então um checkpoint
treinado com várias classes é pontuado como um detector de uma classe.

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
Os resultados de classificação também expõem os dois valores como os atributos
`top1` e `top5`, ao lado de `confusion_matrix`.

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

A classificação ImageFolder adiciona as métricas macro `metrics/precision`, `metrics/recall` e `metrics/f1`, calculadas pela média das classes presentes nos alvos de validação. Top-1 continua sendo o fitness padrão. A detecção também retorna `metrics/best_conf` e `metrics/best_conf_f1`, escolhendo o limiar ótimo de micro-F1 em IoU 0.50, e coloca limiares com nomes de classes como chaves em `metrics.box.best_conf_per_class`. Detecções com pontuação igual permanecem agrupadas; empates escolhem o limiar mais alto. Se não houver F1 positivo, o resultado é 0.0. A segmentação não expõe essas chaves de limiar.

## Matriz de confusão

Os resultados de detect, segment e classify carregam `results.confusion_matrix`.
Ela é calculada em toda validação, com ou sem gráficos gravados, e a sua `matrix`
é um array de inteiros indexado `matrix[predicted, true]`.

<code-tabs name="confusion" />

Para detect e segment, as predições com confiança 0.25, ou `conf` se for maior,
são pareadas um a um com boxes de ground truth de qualquer classe em IoU 0.5, do
maior IoU para o menor. Um par conta na sua classe predita e na verdadeira, então
um box bem posicionado com a classe errada cai fora da diagonal. A matriz tem uma
linha e uma coluna extras `background`: um ground truth sem par conta na linha
`background`, e uma predição sem par na coluna `background`. A segmentação pareia
boxes, não máscaras. Com `classes=`, o ground truth das outras classes fica de
fora.

As regiões de crowd do COCO (`iscrowd=1`) seguem a regra que o mAP aplica. Uma
predição que nenhum ground truth da sua classe pareou, e que fica ao menos pela
metade dentro de uma região de crowd da mesma classe (interseção sobre a área da
própria predição), é ignorada em vez de contada. `box.image_metrics` e as imagens
de `visualize=True` usam os mesmos limiares e a mesma regra de crowd. Os rótulos
em formato YOLO não têm regiões de crowd.

Para classify, cada imagem conta uma vez, na sua classe top-1 contra o seu
rótulo, e não existe `background`.

| Método | Retorna |
|---|---|
| `summary(normalize=False)` | Um dict por classe predita, `{"Predicted": name, <true class>: value}`; com `normalize`, as proporções de cada classe verdadeira |
| `to_csv()`, `to_json()` | O summary como texto CSV ou JSON |
| `to_df()` | O summary como um DataFrame do Polars; precisa de `pip install polars` |
| `tp_fp()` | Verdadeiros positivos por classe (a diagonal) e falsos positivos (o resto da linha predita) |
| `nonzero()` | As células preenchidas como arrays alinhados `(predicted, true, count)` |
| `class_accuracy()` | Por classe, a proporção das suas instâncias dadas àquela classe; na detecção, só dos seus objetos encontrados |
| `plot(normalize=True, save_dir="")` | Grava `confusion_matrix_normalized.png`, ou `confusion_matrix.png` com `normalize=False`, e retorna o caminho |

Acima de 2,000 classes a matriz densa de classificação não é construída:
`matrix`, `summary()`, as exportações `to_*()` e `plot()` levantam erro, enquanto
`tp_fp()`, `nonzero()` e `class_accuracy()` continuam funcionando.

A classe é `libreyolo.validation.ConfusionMatrix`, construída como
`ConfusionMatrix(nc, iou_thres=0.5, conf_thres=0.25, *, names=None, task="detect")`,
onde `task` é `"detect"` ou `"classify"`.

## Precisão, recall e F1 por classe

Os resultados de detect e segment carregam a precisão, o recall e o F1 de cada
classe ao longo dos limiares de confiança, pareados em IoU 0.5.

<code-tabs name="curves" />

`box.px` guarda as 1000 confianças, igualmente espaçadas em `[0, 1]`.
`box.p_curve`, `box.r_curve` e `box.f1_curve` são arrays `(classes, 1000)`, uma
linha por entrada de `box.ap_class_index`, os índices de classe que têm ground
truth no split, em ordem crescente. `box.p`, `box.r` e `box.f1` são os valores por
classe na confiança que maximiza o F1 médio sobre essas classes. A segmentação
calcula todos eles a partir dos boxes. Eles são arrays vazios quando a execução
não produziu curvas.

## Métricas de caixas por imagem

Os resultados de detecção e segmentação continuam compatíveis com dicionários e também expõem `results.box.image_metrics`. Cada nome de arquivo corresponde a `precision`, `recall`, `f1`, `tp`, `fp` e `fn`, usando a regra de correspondência da visualização mesmo quando ela está desativada. A segmentação conta caixas aqui. Nomes de arquivo duplicados usam caminhos completos após a primeira ocorrência. Denominadores zero resultam em 0.0. Esses registros não são reunidos entre ranks distribuídos.

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
`project`, `name` e `exist_ok` o escolhem do mesmo jeito que o treinamento:
`project/name`, com um sufixo incrementado a menos que `exist_ok=True`.

<code-tabs name="json" />

`save_json=True` escreve `predictions.json` para detecção, e
`predictions_bbox.json` mais `predictions_masks.json` para segmentação. OBB não
suporta e avisa.

`save_plots=True`, ou o seu alias `plots=True`, escreve em um subdiretório
`plots/` e precisa do matplotlib, que `pip install "libreyolo[plots]"` instala. A
detecção ganha `box_metrics.png`, gráficos de AP e recall por classe, curvas de
precisão-recall e de confiança, as curvas de F1 por classe em `f1_conf_box.png`,
a matriz de confusão como contagens em `confusion_matrix.png` e como proporções
de cada classe verdadeira em `confusion_matrix_normalized.png`, e imagens de
exemplo anotadas quando o OpenCV está instalado. A segmentação adiciona cópias do
lado da máscara do gráfico de métricas, dos gráficos por classe e das curvas de
precisão-recall e de confiança; as curvas de F1 e a matriz de confusão continuam
no nível dos boxes. Pose ganha seu próprio conjunto de métricas e curvas. A
classificação escreve os dois arquivos da matriz de confusão, exceto acima de
2,000 classes. Os outros validadores não implementam gráficos; semantic, panoptic, depth,
normal, edge, restore, matte, OCR, OBB e point não escrevem nada ali. Uma falha
ao plotar emite um aviso e nunca aborta a execução.

`visualize=True` grava imagens de caixas TP/FP/FN considerando a classe para detecção e segmentação, ou imagens de label versus top-1 para classificação ImageFolder, em `visualize/errors/` e `visualize/correct/`. A correspondência usa IoU 0.5 e confiança `max(0.25, conf)`. Os padrões são `visualize=False`, `show_labels=True` e `show_conf=True`. Tarefas não suportadas e a validação de clipes V-JEPA 2 rejeitam visualização.

`plot_samples=8` limita o gráfico separado de amostras; 0 o desativa e -1 mantém todas as imagens. Isso não muda as métricas nem a saída de visualização.

## Validação durante o treinamento

O treinamento valida a cada `eval_interval` épocas, e sempre após a época final,
no split `val` do dataset, e as métricas que ele produz são as que determinam a
seleção do `best.pt`, o early stopping por `patience` e as chaves `val/` em todo
logger. A validação roda sobre os pesos de EMA quando o EMA está ligado. Seus
arquivos vão para um diretório `val/` dentro da execução. `val=False` desliga a
validação durante o treinamento, incluindo a época final.

Veja [Hiperparâmetros](/docs/train/hyperparameters) para `eval_interval`,
`patience` e `save_plots`, e [Loggers de experimentos](/docs/train/loggers) para
onde os números vão.

## Relacionado

- [Datasets](/docs/train/datasets) para as chaves de split e os formatos que os validadores leem.
