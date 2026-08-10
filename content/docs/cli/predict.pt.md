---
title: libreyolo predict
seo_title: "referência do comando libreyolo predict"
description: "Roda inferência pela linha de comando: cada argumento, seu valor padrão lido da definição da CLI e as flags que mudam o que chega ao stdout."
lead: "Roda um modelo carregado sobre uma fonte e imprime as predições. A fonte pode ser uma imagem, um diretório, um vídeo, uma URL ou um stream ao vivo; o modelo pode ser um checkpoint ou um artefato exportado."
keywords: [libreyolo predict cli, inferência yolo linha de comando, comando predict libreyolo, argumentos libreyolo predict, yolo saída json terminal]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo predict
    mono: true
  - label: Obrigatório
    value: source
    mono: true
  - label: Saída
    value: "Predições no stdout. Com save=true, arquivos anotados em runs/detect/predict"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Salvar imagens anotadas
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Classes filtradas, JSON no stdout
      language: bash
      code: |
        # a classe 0 é person na lista de classes COCO que acompanha o checkpoint.
        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50 \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Sinopse

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Os argumentos são pares `key=value`. O mesmo comando também aceita a forma
POSIX, então `conf=0.4` e `--conf 0.4` são intercambiáveis, e um booleano
escrito `save=true` vira `--save`. Nomes com underscore aceitam qualquer uma das
grafias: `max_det=50` e `--max-det 50` chegam à mesma opção.

`libreyolo detect predict ...` é aceito e se comporta de forma idêntica; a
palavra da tarefa é descartada antes da análise.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `source` | | Caminho de imagem, diretório ou URL. Obrigatório |
| `model` | `yolox-s` | Nome ou caminho do modelo |
| `conf` | `0.25` | Limiar de confiança |
| `iou` | `0.45` | Limiar de IoU do NMS |
| `imgsz` | | Tamanho da imagem de entrada: `640` (quadrada) ou `480x640` (altura x largura). O tamanho de entrada do próprio modelo quando não definido |
| `classes` | | Filtra por IDs de classe, p. ex. `[0,2,5]`. Um inteiro solto é aceito |
| `max_det` | `300` | Máximo de detecções por imagem |
| `half` | `false` | Inferência em FP16 (somente CUDA, exige suporte do modelo) |
| `save` | `false` | Salva as imagens anotadas |
| `batch` | `1` | Imagens por passada para fontes do tipo diretório. Acima de 1 roda inferência em batch de verdade nos modelos que suportam |
| `stream` | `false` | Devolve os resultados de forma incremental. Ativado automaticamente para webcams e streams ao vivo |
| `stream_buffer` | `false` | Armazena em buffer cada frame ao vivo em vez de manter só o mais recente |
| `vid_stride` | `1` | Processa um a cada N frames de vídeo ou ao vivo |
| `show` | `false` | Exibe os resultados de vídeo e ao vivo; `q` interrompe |
| `tiling` | `false` | Inferência por blocos (tiles) para imagens grandes |
| `overlap_ratio` | `0.2` | Proporção de sobreposição entre blocos |
| `output_path` | | Caminho de saída explícito. Caso contrário, `project/name` quando `save=true` |
| `color_format` | `auto` | Cor de entrada: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Formato de saída: `jpg`, `png`, `webp` |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Modelo detector de rostos (caminho ou nome de CLI). Obrigatório para os modelos de gaze |
| `gallery` | | Galeria de rostos `.npz` do `libreyolo enroll` contra a qual identificar rostos. Somente para modelos de embedding facial |
| `gallery_threshold` | `0.4` | Limiar de cosseno para uma correspondência de identidade na galeria |
| `project` | `runs/detect` | Raiz do diretório de saída |
| `name` | `predict` | Nome do experimento |
| `exist_ok` | `false` | Reaproveita o diretório de saída existente |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `verbose` | `false` | Saída detalhada no stderr |
| `help_json` | `false` | Despeja o schema do comando como JSON e sai |

## Exemplos

<code-tabs name="examples" />

## Notas

Um artefato exportado é carregado do mesmo jeito que um checkpoint, então
`model=weights/LibreYOLO9s.onnx` e `model=weights/LibreYOLO9s.engine` são
valores válidos para `model`. Três opções são recusadas nesses runtimes em vez
de ignoradas: `tiling`, `overlap_ratio` e `output_file_format` saem com
`config_unsupported` quando um backend de runtime não consegue atendê-las.

`half` funciona ao contrário. Os runtimes exportados recebem a opção e rodam em
FP16; a inferência nativa em PyTorch registra que ela foi ignorada e continua em
FP32.

Os modelos de gaze (estimativa do olhar) são de duas etapas e não têm detector
próprio, então `face_detector` é obrigatório para eles. `gallery` só se aplica a
modelos cuja tarefa é `embed`; passar isso para qualquer outro sai com
`config_unsupported`.

O stdout carrega os resultados e nada mais; progresso, avisos e erros vão para o
stderr. `json=true` imprime um objeto JSON por invocação, ou um por frame
quando em streaming, cada um carregando `schema_version`. `quiet=true` silencia
o stderr. Os dois juntos dão a um leitor automático um stream de stdout limpo.

O código de saída é `0` em caso de sucesso, `2` para um erro de uso ou de
configuração, `3` quando a fonte não é encontrada, `4` quando o modelo não pode
ser carregado e `1` para outras falhas em tempo de execução.

`help_json=true` imprime os parâmetros, tipos, valores padrão e flags do comando
como JSON sem rodar nada, o que é a forma confiável de ler esta tabela a partir
de uma versão instalada.

Relacionado: [`libreyolo val`](/docs/cli/val) para métricas medidas sobre um
dataset, [`libreyolo export`](/docs/cli/export) para produzir os artefatos de
runtime citados acima.
