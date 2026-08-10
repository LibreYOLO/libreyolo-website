---
title: utilitários do libreyolo
seo_title: "referência dos comandos utilitários da CLI do libreyolo"
description: "Os comandos pequenos do LibreYOLO: version, checks, models, formats, cfg, info, metadata, enroll e compare, cada um com seus argumentos e valores padrão."
lead: "Nove comandos que relatam ou inspecionam em vez de calcular. Eles imprimem dados do ambiente, o inventário de modelos e formatos, os valores padrão já resolvidos e os detalhes de um checkpoint, além de construir e consultar uma galeria de rostos."
keywords: [libreyolo version, libreyolo checks, listar modelos libreyolo, formatos de exportacao libreyolo, ver metadados de checkpoint yolo, galeria de rostos libreyolo enroll]
last_verified: "1.5.0"
meta:
  - label: Comandos
    value: version, checks, models, formats, cfg, info, metadata, enroll, compare
    mono: true
  - label: Saída
    value: "stdout, em texto ou com json=true como um único objeto que carrega schema_version"
snippets:
  examples:
    - label: Ambiente
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: O que está disponível
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Inspecionar um checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
---

## Sinopse

```bash
libreyolo <command> [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`model=x` e `--model x` são o mesmo argumento. Todo comando aqui escreve os
resultados no stdout e aceita `json=true` e `quiet=true`.

O comando raiz tem uma flag própria, `libreyolo --version`, que imprime a string
de versão e sai. Essa saída é menor que a do comando `version` abaixo.

## version

Imprime a versão do LibreYOLO mais as versões de Python, torch e CUDA com as
quais ele está rodando.

```bash
libreyolo version
```

| Argumento | Padrão | Significado |
|---|---|---|
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

## checks

Imprime o ambiente com mais detalhes: Python, torch, CUDA, cuDNN, cada GPU
detectada com seu nome e sua memória, e a versão instalada de cada pacote
opcional que os caminhos de exportação usam.

```bash
libreyolo checks
```

| Argumento | Padrão | Significado |
|---|---|---|
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

A lista de pacotes cobre `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` e `scipy`. Um pacote que não está instalado é reportado como tal
em vez de ser omitido, então uma exportação que falhou pode ser rastreada até uma
dependência ausente a partir desse único comando.

## models

Lista cada família de modelos com suas tarefas, seus tamanhos, os nomes de CLI
que resolvem para seus checkpoints e a resolução de entrada de cada tamanho.

```bash
libreyolo models
```

| Argumento | Padrão | Significado |
|---|---|---|
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

Uma família cuja dependência opcional não está instalada é listada como
indisponível junto com a linha de `pip install` que a tornaria disponível. Os
nomes de CLI são o que `model=` aceita como atalho: `yolox-s` resolve para
`LibreYOLOXs.pt`, e as tarefas que não são de detecção carregam o sufixo da sua
tarefa.

## formats

Lista os formatos de exportação que o ambiente instalado consegue produzir, com
a extensão de arquivo de cada formato e se ele suporta FP16 e INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `family` | | Mostra os níveis para uma família de modelos. `model=` é aceito como a mesma opção |
| `task` | | Tarefa canônica do modelo. A tarefa padrão da família quando não definida |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

Sem `family`, a saída é apenas o inventário de formatos. Com ela, cada formato
ganha o nível de suporte para aquela família e tarefa, o motivo por trás do
nível e qualquer restrição associada a ele. Uma família desconhecida, ou uma
tarefa que a família não suporta, é um erro de uso.

Os aliases de formato aparecem ao lado do nome canônico: `engine` para
`tensorrt`, `litert` para `tflite`.

## cfg

Imprime a configuração padrão já resolvida: os padrões de treinamento, os
padrões de validação, os padrões de predição e as sobrescritas por família.

```bash
libreyolo cfg
```

| Argumento | Padrão | Significado |
|---|---|---|
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

Os valores são lidos das dataclasses de configuração, não de uma cópia, então
esta é a autoridade sobre o que uma execução de treinamento vai usar quando você
não passa um argumento. `family_overrides` é a seção que responde por que uma
família treinou com configurações que você não pediu. Veja
[`libreyolo train`](/docs/cli/train) para saber como essas sobrescritas são
aplicadas.

## info

Carrega um modelo na CPU e relata sua família, seu tamanho, a contagem de
parâmetros, as classes e o nível de exportação de cada formato.

```bash
libreyolo info model=<name|path>
```

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Nome do modelo ou caminho para os pesos. Obrigatório |
| `detailed` | `false` | Inclui detalhes por parâmetro |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

## metadata

Lê os metadados de um checkpoint sem construir um modelo, e os valida contra o
schema de checkpoint do LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argumento | Padrão | Significado |
|---|---|---|
| `path` | | Caminho para um checkpoint `.pt`. Obrigatório |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

Entradas grandes que carregam tensores são resumidas em vez de impressas, então
a saída continua legível em um checkpoint completo de treinamento. Um checkpoint
que não existe sai com `checkpoint_not_found`, e um cujos metadados falham na
validação imprime os erros e sai com `1`.

## enroll

Constrói uma galeria de rostos a partir de uma árvore com uma pasta por pessoa,
para que predições posteriores consigam nomear os rostos que encontram.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Modelo de embedding facial, caminho ou nome. Obrigatório |
| `source` | | Árvore com uma pasta por pessoa, `source/<identity>/*.jpg`. Obrigatório |
| `gallery` | | Arquivo `.npz` de galeria de saída. Estendido no lugar se já existir. Obrigatório |
| `face_detector` | | Detector de rostos: um `.onnx` YuNet ou um detector LibreYOLO. O detector padrão da família quando não definido |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

```bash
# people/ guarda uma pasta por identidade; o nome da pasta vira a identidade.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

O nome da subpasta é a identidade. Uma imagem de referência sem rosto detectável
é pulada com uma linha no stderr e o resto continua; um source sem subpastas de
identidade, ou um em que nenhum rosto foi encontrado, é um erro.

Passe o arquivo resultante para
[`libreyolo predict`](/docs/cli/predict) como `gallery=people.npz` para que as
detecções carreguem uma identidade e uma pontuação de correspondência.

## compare

Relata a similaridade de cosseno entre duas imagens de rosto e se ela supera o
limiar de mesma identidade.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Modelo de embedding facial, caminho ou nome. Obrigatório |
| `source` | | Primeira imagem. Obrigatório |
| `source2` | | Segunda imagem para comparar. Obrigatório |
| `face_detector` | | Detector de rostos: um `.onnx` YuNet ou um detector LibreYOLO |
| `threshold` | `0.4` | Limiar de similaridade de cosseno para a decisão de mesma identidade |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` está registrado como um segundo nome para esse comando e
recebe os mesmos argumentos.

Tanto `compare` quanto `enroll` precisam de um modelo cuja tarefa seja embedding
facial. Qualquer outra coisa sai com `config_unsupported`. Caminhos locais de
imagem e URLs `http` ou `https` são aceitos como fontes.

## Exemplos

<code-tabs name="examples" />

## Notas

O stdout carrega o resultado; o progresso e os avisos vão para o stderr.
`json=true` imprime um único objeto com `schema_version`, que é a forma a ser
lida a partir de um script. A saída em texto é o padrão e é feita para ser lida
por uma pessoa.

Os códigos de saída seguem o mesmo mapa do resto da CLI: `0` em caso de sucesso,
`2` para um erro de uso ou de configuração, `3` quando um source não pode ser
encontrado, `4` quando um modelo ou checkpoint não pode ser carregado, e `1`
para outras falhas em runtime.

Relacionados: [`libreyolo doctor`](/docs/cli/doctor), que é o comando de
inspeção do lado do dataset, e [`libreyolo profile`](/docs/cli/profile), o do
lado de desempenho.
