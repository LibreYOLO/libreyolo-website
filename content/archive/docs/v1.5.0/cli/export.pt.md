---
title: libreyolo export
seo_title: referência do comando libreyolo export
description: >-
  Exporte um checkpoint para um formato de deploy: cada argumento com seu valor
  padrão, onde o artefato é gravado e as combinações que o comando recusa.
lead: >-
  Converte um checkpoint em um formato de deploy e grava o artefato em weights/.
  O formato decide quais dos argumentos abaixo se aplicam.
keywords:
  - libreyolo export cli
  - exportar yolo onnx
  - comando libreyolo export
  - exportar yolo tensorrt
  - argumentos libreyolo export
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo export
    mono: true
  - label: Obrigatório
    value: model
    mono: true
  - label: Saída
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Grava weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS dentro do grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Executar o artefato
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # A factory se orienta pelo sufixo do arquivo, então a exportação
        carrega como um checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Sinopse

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`format=onnx` e `--format onnx` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Pesos do modelo `.pt`. Obrigatório |
| `format` | `onnx` | Formato de exportação: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Plataforma de destino do RKNN, hoje apenas `rk3588`. Recusado com qualquer outro formato |
| `imgsz` | | Tamanho da imagem de entrada: `640` ou `480x640` (altura x largura). `480,640` também é aceito. O tamanho do próprio modelo quando não definido |
| `batch` | `1` | Tamanho de batch da exportação |
| `half` | `false` | Precisão FP16 |
| `int8` | `false` | Quantização INT8 |
| `dynamic` | `false` | Formas de entrada dinâmicas (ONNX) |
| `simplify` | `true` | Simplificação do grafo ONNX |
| `nms` | `false` | Embute o NMS no modelo. Apenas ONNX e CoreML |
| `conf` | `0.25` | Limiar de confiança para o NMS embutido |
| `iou` | `0.45` | Limiar de IoU para o NMS embutido |
| `max_det` | `300` | Máximo de detecções para o NMS embutido do ONNX |
| `opset` | | Versão do opset do ONNX. Escolhida automaticamente quando não definida |
| `data` | | Dados de calibração para INT8 |
| `fraction` | `1.0` | Fração dos dados de calibração a usar |
| `device` | `auto` | Dispositivo para o tracing |
| `allow_download_scripts` | `false` | Permite Python embutido nos blocos de download do YAML do dataset |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `verbose` | `false` | Log detalhado da exportação |
| `verify` | `false` | Executa o simulador de PC do RKNN Toolkit2 e compara com o ONNX Runtime. Apenas RKNN |
| `help_json` | `false` | Despeja o schema do comando como JSON e sai |

`engine` é um alias de `tensorrt` e `litert` um alias de `tflite`. Os dois se
resolvem para o nome canônico antes de qualquer coisa ser gravada, então a saída
JSON e a linha de log sempre informam `tensorrt` ou `tflite`.

## Exemplos

<code-tabs name="examples" />

## Notas

### Onde o arquivo é gravado

O comando não aceita caminho de saída. O artefato é gravado em `weights/`, com o
nome-base do checkpoint de origem mais o sufixo do formato, e com `_fp16` ou
`_int8` inserido quando uma dessas precisões foi pedida. `LibreYOLO9s.pt`
exportado para ONNX em FP16 vira `weights/LibreYOLO9s_fp16.onnx`. O resultado
JSON traz o `output_path` resolvido, o tamanho do arquivo em MB e a forma de
entrada como `[batch, 3, height, width]`.

### Combinações que são recusadas

`nms=true` é aceito para ONNX e CoreML e recusado para todos os outros formatos
com `nms_unsupported_format`. No ONNX ele força `dynamic` para off, já que o
grafo embutido é fixo em batch 1, e avisa isso no stderr. No CoreML ele aceita
`conf` e `iou`, mas não `max_det`, então um `max_det` diferente do padrão junto
com `format=coreml nms=true` sai com `config_unsupported`.

`half=true` junto com `int8=true` não é erro. INT8 ganha, `half` é descartado e
um aviso vai para o stderr.

`name` e `verify` são opções do RKNN hoje. Passar qualquer uma delas com outro
formato sai com `config_unsupported` em vez de ser ignorada.

### Quais formatos cada família suporta

O suporte é por família e por tarefa, não global. `libreyolo formats
family=<family> task=<task>` imprime o nível de cada formato para essa
combinação, com o motivo e qualquer restrição associada. Veja
[`libreyolo formats`](/docs/cli/utilities) para os argumentos.

Alguns formatos precisam de uma instalação opcional e outros precisam de um
toolchain. Uma dependência Python faltando sai com `export_dep_missing`; uma
precisão que o formato não consegue produzir sai com
`format_precision_unsupported`.

### Rodar o que você exportou

Os artefatos exportados carregam pela mesma factory de modelos que os
checkpoints, guiada pelo sufixo do arquivo, então
`libreyolo predict model=weights/LibreYOLO9s.onnx` funciona sem nenhuma
conversão adicional. Três opções de predição são a exceção e são recusadas nos
backends de runtime: `tiling`, `overlap_ratio` e `output_file_format`.

Dois destinos de deploy têm página própria:
[NVIDIA DeepStream](/docs/export/deepstream) e
[NVIDIA Jetson](/docs/export/jetson).

### Saída e códigos de saída

O stdout leva o resultado; o progresso vai para o stderr. O código de saída é
`0` em caso de sucesso, `2` para um erro de uso ou de configuração, `4` quando o
modelo não pode ser carregado, `5` para um formato desconhecido, uma dependência
de exportação faltando, uma precisão não suportada ou um pedido de NMS embutido
recusado, e `1` para outras falhas em tempo de execução.

Relacionado: [`libreyolo quantize`](/docs/cli/quantize), que fica no PyTorch e
grava um checkpoint em vez de um artefato de deploy.
