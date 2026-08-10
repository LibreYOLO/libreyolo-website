---
title: libreyolo train
seo_title: "referência do comando libreyolo train"
description: "Treina um modelo pela linha de comando: os 59 argumentos com seus valores padrão, como os padrões de cada família os sobrescrevem e quais argumentos uma família ignora."
lead: "Treina um modelo em um dataset e escreve checkpoints, métricas e logs em um diretório de execução. Cada argumento abaixo tem um valor padrão vindo da definição do comando, que a config de treinamento própria de uma família de modelos pode substituir."
keywords: [libreyolo train cli, treinar yolo linha de comando, comando libreyolo train, argumentos libreyolo train, treinar yolo com dataset próprio, congelar camadas yolo]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo train
    mono: true
  - label: Obrigatório
    value: data
    mono: true
  - label: Saída
    value: "Checkpoints, métricas e logs em runs/train/exp"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # coco8.yaml vem com o pacote e baixa suas 8 imagens no primeiro uso.
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640 batch=8
    - label: Conferir primeiro a config resolvida
      language: bash
      code: |
        # Imprime o que a execução usaria, incluindo os padrões da família, e sai
        # sem treinar nem carregar dados.
        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10 dry_run=true
    - label: Execução nomeada com uma receita explícita
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
---

## Sinopse

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`epochs=50` e `--epochs 50` são o mesmo argumento. Os booleanos aceitam `true` e
`false`: `amp=false` vira `--no-amp` onde a flag tem forma negativa.

## Argumentos

### Modelo e dados

| Argumento | Padrão | Significado |
|---|---|---|
| `data` | | Caminho para o YAML do dataset (formato YOLO, ex.: `coco8.yaml`). Obrigatório |
| `model` | `yolox-s` | Nome do modelo ou caminho para os pesos |
| `task` | | Sobrescrita explícita da tarefa: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Usa pesos pré-treinados. `false` constrói a arquitetura e treina do zero |
| `allow_download_scripts` | `false` | Permite Python embutido nos blocos de download do YAML do dataset |

### Loop de treinamento

| Argumento | Padrão | Significado |
|---|---|---|
| `epochs` | `300` | Épocas de treinamento |
| `batch` | `16` | Tamanho de batch por dispositivo |
| `imgsz` | `640` | Tamanho da imagem de treinamento: `640` (quadrada) ou `480x640` (altura x largura) |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Workers do dataloader |
| `cache` | `false` | Guarda as imagens em cache para acelerar o carregamento de dados: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Semente aleatória |
| `resume` | | Retoma o treinamento: `true`, ou um caminho para um checkpoint |
| `amp` | `true` | Precisão mista automática (AMP) |
| `amp_dtype` | `float16` | dtype do AMP em CUDA: `float16` ou `bfloat16` |
| `cuda_graph` | `false` | Captura o forward e o backward do treinamento em CUDA graphs. Só uma GPU e só famílias suportadas; as demais rodam em modo eager |
| `lora` | `false` | Fine-tuning com LoRA, para as famílias transformer listadas em Notas |
| `freeze` | | Congela camadas: um número inteiro, uma lista de índices ou nomes de módulos |

### Destilação

| Argumento | Padrão | Significado |
|---|---|---|
| `distill_model` | | Teacher: um checkpoint de detector, ou um id de foundation teacher como `dinov2` para destilação de características do backbone |
| `dis` | | Peso da loss de destilação. Quando não definido, o padrão publicado para o tipo de loss |
| `distill_loss_type` | `mgd` | Loss de características para teachers do tipo detector: `mgd`, `cwd`. Foundation teachers sempre usam `feat_mse` |

### Otimizador

| Argumento | Padrão | Significado |
|---|---|---|
| `optimizer` | `sgd` | Otimizador: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate inicial |
| `momentum` | `0.937` | Momentum do SGD, e o coeficiente de primeiro momento para os otimizadores Adam |
| `weight_decay` | `0.0005` | Regularização L2 |
| `nesterov` | `true` | Momentum de Nesterov |

### Scheduler

| Argumento | Padrão | Significado |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Tipo de agendamento do LR |
| `warmup_epochs` | `5` | Duração do warmup |
| `warmup_lr_start` | `0.0` | LR inicial do warmup |
| `min_lr_ratio` | `0.05` | Razão mínima de LR |
| `lr_drop` | `100` | Época da queda escalonada do LR no RF-DETR |

### Data augmentation

| Argumento | Padrão | Significado |
|---|---|---|
| `mosaic` | `1.0` | Probabilidade de mosaic |
| `mixup` | `1.0` | Probabilidade de mixup |
| `hsv_prob` | `1.0` | Probabilidade de jitter HSV |
| `flip_prob` | `0.5` | Probabilidade de flip horizontal |
| `degrees` | `10.0` | Faixa de rotação, para mais e para menos, em graus |
| `translate` | `0.1` | Razão de translação |
| `shear` | `2.0` | Ângulo de cisalhamento |
| `mosaic_scale` | `(0.1,2.0)` | Faixa de escala do mosaic |
| `mixup_scale` | `(0.5,1.5)` | Faixa de escala do mixup |
| `no_aug_epochs` | `15` | Desativa o data augmentation nas N últimas épocas |

### EMA

| Argumento | Padrão | Significado |
|---|---|---|
| `ema` | `true` | Média móvel exponencial |
| `ema_decay` | `0.9998` | Fator de decaimento da EMA |

### Validação durante o treinamento

| Argumento | Padrão | Significado |
|---|---|---|
| `val` | `true` | Valida durante o treinamento |
| `eval_interval` | `10` | Valida a cada N épocas |
| `max_det` | `300` | Máximo de predições por imagem depois do NMS de validação |
| `eval_max_det` | | Teto do avaliador COCO. Quando não definido, a convenção AP@100 do pycocotools |
| `faster_coco_eval` | `true` | Usa o backend C++ faster-coco-eval para as métricas COCO quando instalado; se não, recorre ao pycocotools |
| `save_plots` | `false` | Salva os gráficos finais de validação durante o treinamento |
| `patience` | `50` | Paciência do early stopping. `0` desativa |

### Saída

| Argumento | Padrão | Significado |
|---|---|---|
| `project` | `runs/train` | Raiz do diretório de saída |
| `name` | `exp` | Nome do experimento |
| `exist_ok` | `false` | Reutiliza um diretório de saída existente |
| `save_period` | `10` | Salva um checkpoint a cada N épocas |
| `log_interval` | `10` | Registra a loss a cada N batches |

### Flags de agente

| Argumento | Padrão | Significado |
|---|---|---|
| `json` | `false` | Saída JSON pelo stdout |
| `quiet` | `false` | Silencia o stderr |
| `dry_run` | `false` | Resolve e imprime a config sem executar |
| `help_json` | `false` | Despeja o schema do comando como JSON e sai |

## Exemplos

<code-tabs name="examples" />

## Notas

### Os padrões acima nem sempre são os valores usados

Cada família de modelos carrega sua própria config de treinamento e, onde essa
config difere da base, o valor dela substitui o padrão do comando para qualquer
argumento que você não tenha definido explicitamente. Definir o argumento você
mesmo sempre vence. `libreyolo cfg` imprime os padrões base e as sobrescritas
por família, que é a forma de ver o que uma dada família vai realmente usar.

`imgsz` é o argumento em que isso mais importa. O padrão do comando é `640`, que
não é a entrada nativa de todo checkpoint: os tamanhos de detecção publicados do
RF-DETR são 384, 512, 576 e 704, e os checkpoints `n` e `t` do YOLOX são 416.
RF-DETR e DEIMv2 são tratados repassando `imgsz` apenas quando ele foi definido
explicitamente, então o tamanho próprio deles continua valendo caso contrário.
As outras famílias recebem o valor como veio e treinam nele. FOMO é a rigorosa:
cada tamanho aceita apenas sua entrada nativa (96, 192 e 224), então uma execução
de FOMO precisa de `imgsz` definido para corresponder, ou ela para com erro. O
RF-DETR ainda exige que o valor seja divisível pelo tamanho do patch vezes a
contagem de janelas, e informa os dois tamanhos válidos mais próximos quando não
é.

### Argumentos que uma família ignora

Nem toda família lê todo argumento, e é nos de data augmentation que isso
aparece. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 e DINOv2 treinam por pipelines
de passagem direta, sem mosaic, sem mixup e sem warp afim, então `mosaic`,
`mixup`, `hsv_prob`, `degrees`, `translate`, `shear`, `mosaic_scale` e
`mixup_scale` não chegam a nada ali. A EC compartilha esse pipeline, mas lê sim
`hsv_prob`, `degrees` e `translate` quando sua tarefa é pose. As famílias de
classificação, SegFormer e NAFNet ignoram esse conjunto inteiro e `flip_prob`
junto, porque o flip delas roda com uma probabilidade fixa em vez de
configurável. A YOLO-NAS ignora só `mosaic`, já que faz augmentation com uma
transformação afim por amostra sempre ligada. O RF-DETR ignora mais três além
dessa lista: `optimizer`, `momentum` e `nesterov`.

Definir um desses não é erro. A execução registra uma linha no stderr nomeando a
família e os argumentos que ela vai ignorar, depois treina, e essa linha é a
lista autoritativa para a versão instalada. Ela também é o único sinal, então uma
execução em script com `quiet=true` suprime o aviso junto com todo o resto do
stderr.

`val=false` é um caso relacionado. Ele define `eval_interval` como `0` para a
maioria das famílias; o RF-DETR não consegue desativar a validação desse jeito e
registra que ignorou o pedido.

### Outros comportamentos que vale conhecer

`lora=true` é aceito por RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4, EC e
ConvNeXt. Qualquer outra família sai com `config_unsupported` em vez de treinar
sem isso.

`pretrained=false` combinado com `resume` é recusado para as famílias que
suportam treinamento do zero, já que os dois pedem coisas opostas.

`mosaic` e `mixup` são as grafias de linha de comando dos campos de config
`mosaic_prob` e `mixup_prob`. Nas famílias cujo mixup só se aplica a amostras de
mosaic, `mixup` acima de zero com `mosaic` em zero nunca dispara, e a execução
diz isso.

`dry_run=true` resolve a referência do modelo, aplica os padrões da família e
imprime a config com que treinaria. Ele não carrega o dataset, então é a forma
barata de confirmar que um argumento chegou ao valor que você esperava.

O stdout carrega o objeto de resultado final; progresso e avisos vão para o
stderr. O código de saída é `0` em caso de sucesso, `2` para um erro de uso ou de
configuração, `3` quando o dataset não pode ser encontrado ou lido, `4` quando o
modelo não pode ser carregado, e `1` para outras falhas de runtime.

Relacionados: [`libreyolo doctor`](/docs/cli/doctor) para conferir um dataset
antes de se comprometer com uma execução, [`libreyolo monitor`](/docs/cli/monitor)
para acompanhar uma execução no navegador, [`libreyolo val`](/docs/cli/val) para
medir o resultado.
