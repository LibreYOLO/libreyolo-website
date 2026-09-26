---
title: libreyolo val
seo_title: referência do comando libreyolo val
description: >-
  Avalia um checkpoint em um split de um dataset pela linha de comando: cada
  argumento com seu valor padrão e as chaves de métricas que cada tarefa
  retorna.
lead: >-
  Avalia um modelo contra um split de um dataset e imprime as métricas. O
  conjunto de métricas depende da tarefa do modelo, e os números são os mesmos
  com que uma linha de benchmark é construída.
keywords:
  - libreyolo val cli
  - comando de validação libreyolo
  - avaliar modelo yolo linha de comando
  - calcular mAP50-95 terminal
  - argumentos libreyolo val
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo val
    mono: true
  - label: Obrigatório
    value: 'model, data'
    mono: true
  - label: Saída
    value: Métricas no stdout. Gráficos e JSON COCO em runs/val/exp quando pedidos
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Gráficos e JSON COCO
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Legível por máquina
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Sinopse

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`batch=8` e `--batch 8` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Caminho dos pesos do modelo ou nome de CLI. Obrigatório |
| `data` | | Caminho para o YAML do dataset (formato YOLO, p. ex. `coco8.yaml`). Obrigatório |
| `data_dir` | | Diretório do dataset direto, ignorando o caminho indicado no YAML |
| `split` | `val` | Split do dataset: `val`, `test`, `train` |
| `batch` | `16` | Tamanho de batch |
| `imgsz` | | Tamanho da imagem: `640` (quadrada) ou `480x640` (altura x largura). O tamanho de entrada do próprio modelo quando não definido |
| `conf` | `0.001` | Limiar de confiança |
| `iou` | `0.6` | Limiar de IoU do NMS |
| `max_det` | `300` | Máximo de predições por imagem após o NMS |
| `eval_max_det` | | Teto do avaliador COCO. A convenção AP@100 do pycocotools quando não definido |
| `faster_coco_eval` | `true` | Usa o backend C++ do faster-coco-eval para as métricas COCO quando instalado; volta para o pycocotools |
| `half` | `false` | Inferência em FP16 |
| `amp_dtype` | `float16` | Dtype do autocast do CUDA quando `half=true`: `float16` ou `bfloat16` |
| `save_json` | `false` | Salva os resultados em JSON no formato COCO |
| `save_plots` | `false` | Salva os gráficos de validação: métricas, AP por classe, matriz de confusão, amostras |
| `workers` | `4` | Workers do dataloader |
| `device` | `auto` | Dispositivo |
| `project` | `runs/val` | Raiz do diretório de saída |
| `name` | `exp` | Nome do experimento |
| `exist_ok` | `false` | Reutiliza o diretório de saída |
| `allow_download_scripts` | `false` | Permite Python embutido nos blocos de download do YAML do dataset |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `verbose` | `true` | Saída detalhada |
| `help_json` | `false` | Despeja o esquema do comando como JSON e sai |

## Exemplos

<code-tabs name="examples" />

## Notas

### O que são as métricas

O conjunto impresso segue a tarefa do modelo, e a saída JSON usa as mesmas
chaves.

Detecção, segmentação e caixas orientadas reportam `mAP50`, `mAP50_95`,
`precision` e `recall`. Quando um modelo prediz mais de um tipo de saída, os
grupos por tipo aparecem ao lado como `box_metrics`, `mask_metrics` e
`obb_metrics`, cada um com essas mesmas quatro chaves.

A classificação reporta `accuracy_top1` e `accuracy_top5`. A detecção de pontos
reporta `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` e `mAP_sweep`. A
profundidade reporta `abs_rel`, `rmse`, `delta1`, `delta2` e `delta3`. A
segmentação semântica reporta `mIoU` e `pixel_accuracy`. A restauração reporta
`PSNR` e `SSIM`.

O resultado JSON carrega também `eval_backend`, que nomeia a biblioteca de
avaliação COCO e a versão que produziram os números, de modo que duas execuções
possam ser comparadas sabendo se o mesmo backend pontuou as duas.

### Limiares

Os valores padrão daqui são valores de avaliação, não de predição: `conf` é
`0.001` e `iou` é `0.6`, enquanto [`libreyolo predict`](/docs/cli/predict) usa
`0.25` e `0.45`. Subir `conf` para um limiar de exibição derruba o recall e com
ele a mAP, então um número obtido dessa forma não é comparável a um publicado.

`imgsz` não vem definido por padrão, o que significa o tamanho de entrada do
próprio modelo. Defini-lo avalia no tamanho indicado, que é como um checkpoint
acaba sendo medido fora da sua resolução nativa.

### Datasets que baixam sozinhos

Um YAML de dataset cujo campo `download` é uma URL baixa no primeiro uso sem
nenhuma permissão extra. Um que carrega um script de download em Python
embutido precisa de `allow_download_scripts=true`, e o comando avisa no stderr
que a execução de código local foi habilitada. Os `coco8.yaml` e `coco128.yaml`
que vêm inclusos são baseados em URL, então não precisam de nada.

### Saída e códigos de saída

O stdout carrega as métricas; o progresso vai para o stderr. `json=true` imprime
um objeto com `schema_version`, e `quiet=true` silencia o stderr.

O código de saída é `0` em caso de sucesso, `2` para um erro de uso ou de
configuração, `3` quando o dataset não é encontrado, `4` quando o modelo não
pode ser carregado e `1` para outras falhas em tempo de execução.

Relacionado: [`libreyolo train`](/docs/cli/train), que roda essa mesma avaliação
no seu próprio ritmo através de `eval_interval`.
