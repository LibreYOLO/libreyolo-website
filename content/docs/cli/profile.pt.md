---
title: libreyolo profile
seo_title: "referência do comando libreyolo profile"
description: "Meça a velocidade do treinamento e da inferência e leia o resultado: cada subcomando de profile, seus argumentos e valores padrão, e o que cada ângulo de leitura informa."
lead: "Um grupo de comandos que mede para onde vai o tempo em um passo de treinamento ou em uma chamada de inferência, escreve um perfil autocontido e lê esse perfil de volta por vários ângulos."
keywords: [libreyolo profile cli, profiling de treinamento yolo, medir latência de inferência yolo, profiling de kernels gpu pytorch, comparar desempenho libreyolo]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo profile
    mono: true
  - label: Saída
    value: "profile.json e profile_trace.json em runs/profile"
    mono: true
snippets:
  examples:
    - label: Medir a inferência
      language: bash
      code: |
        # Sem argumento source, usa a imagem de exemplo incluída.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Ler o veredito
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Comparar duas medições
      language: bash
      code: |
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project runs/profile/a
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4 --project runs/profile/b

        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
---

## Sinopse

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Este grupo não aceita argumentos `key=value`. Seus subcomandos usam argumentos
posicionais e flags POSIX, ou seja, é `--weights LibreYOLO9t.pt`, não
`weights=LibreYOLO9t.pt`. Rodar `libreyolo profile` sem subcomando imprime a
lista.

Dois subcomandos medem e escrevem um perfil; os demais leem um. `run` e `infer`
emitem o mesmo `profile.json` autocontido, então todo subcomando de leitura
funciona com qualquer um dos dois.

## profile run

Roda um treinamento curto com profiling e escreve um perfil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `data` | | Posicional. YAML ou nome do dataset, por exemplo `coco128`. Obrigatório |
| `--weights` | `LibreYOLO9t.pt` | Arquivo de pesos ou nome do modelo |
| `--size` | `t` | Variante de tamanho do modelo |
| `--batch` | `16` | Micro-batch. `-1` ajusta automaticamente para cerca de 70% da VRAM |
| `--imgsz` | `640` | Tamanho da imagem de treinamento |
| `--workers` | `8` | Workers do dataloader |
| `--amp` | `true` | Usa o caminho AMP da família. `--no-amp` desativa |
| `--steps` | `20` | Passos com profiling, isto é, medidos |
| `--warmup` | `5` | Passos de warmup antes de medir |
| `--repeat` | `1` | Repete N vezes para obter média e desvio padrão |
| `--device` | `0` | Dispositivo |
| `--project` | `runs/profile` | Raiz do diretório de saída |
| `--json` | `false` | Saída JSON no stdout |

A janela medida é `--warmup` mais `--steps` iterações. Um dataset pequeno
demais para preenchê-la não produz perfil algum e o comando termina com o
código `3`, apontando as três saídas possíveis: um dataset maior, menos passos
ou um batch menor.

`--repeat` acima de 1 escreve um `runs/profile/profile_repeat.json` agregado,
cujas métricas escalares são a média das tentativas, enquanto as listas de
kernels vêm da última tentativa. Ele também é pré-requisito para um veredito de
significância no `compare`: uma única execução não consegue fornecer um.

## profile infer

Faz profiling do caminho de inferência e escreve um perfil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `source` | | Posicional. Imagem ou diretório. A imagem de exemplo incluída quando omitido |
| `--weights` | `LibreYOLO9t.pt` | Arquivo de pesos ou nome do modelo |
| `--size` | `t` | Variante de tamanho do modelo |
| `--batch` | `1` | Imagens por passada forward |
| `--imgsz` | `640` | Tamanho da imagem de entrada |
| `--half` | `false` | Forward com autocast, apenas CUDA. `--no-half` desativa |
| `--amp-dtype` | `float16` | Dtype do autocast CUDA: `float16` ou `bfloat16` |
| `--warmup` | `20` | Iterações de warmup antes de medir |
| `--runs` | `100` | Iterações medidas |
| `--repeat` | `1` | Repete N vezes para obter média e desvio padrão |
| `--conf` | `0.25` | Limiar de confiança, que muda quanto trabalho o NMS faz |
| `--iou` | `0.45` | Limiar de IoU do NMS |
| `--max-det` | `300` | Máximo de detecções por imagem, que muda quanto trabalho o NMS faz |
| `--device` | `0` | Dispositivo |
| `--trace` | `true` | Emite um trace do Chrome para detalhar kernels e ops. `--no-trace` pula isso |
| `--project` | `runs/profile` | Raiz do diretório de saída |
| `--json` | `false` | Saída JSON no stdout |

Informa a latência em p50, p90 e p99, o throughput em imagens por segundo e a
divisão por etapas entre pré-processamento, forward e pós-processamento. Os três
argumentos de limiar estão aqui porque mexem no número do pós-processamento.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um `profile.json` ou `profile_trace.json`. Obrigatório |
| `--json` | `false` | Saída JSON no stdout |

A leitura de alto nível: tempo por passo, throughput, utilização da GPU,
participação dos Tensor Cores, pico de VRAM, overhead do host, lançamentos de
kernel por passo, o veredito sobre o gargalo com seu motivo, a mistura de
kernels por categoria e os principais kernels por passo. Em um perfil de
inferência, imprime também os percentis de latência e a divisão por etapas.

Um perfil tomado sob thrashing de VRAM é marcado, porque a utilização e o
throughput medidos ali não são confiáveis.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um perfil. Obrigatório |
| `field` | | Posicional. Nome da métrica. Omita para listar as métricas disponíveis |
| `--json` | `false` | Saída JSON no stdout |

Imprime uma métrica e nada mais, para loops em scripts. Um campo desconhecido
termina com o código `2` e aponta para a forma de listagem.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um perfil. Obrigatório |
| `--json` | `false` | Saída JSON no stdout |

Milissegundos de GPU, milissegundos de relógio, contagem de kernels e contagem
de ops por fase: forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um perfil. Obrigatório |
| `--top` | `20` | Mostra os N principais por tempo de GPU |
| `--category` | | Filtra por substring de categoria: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filtra por expressão regular sobre o nome do kernel |
| `--tensorcore` | `false` | Apenas kernels de Tensor Core |
| `--sort` | `time` | `time`, `count` ou `name` |
| `--phase` | | Restringe a uma fase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Saída JSON no stdout |

O fundo da análise: kernels de GPU individuais com sua fatia do tempo de GPU,
milissegundos por passo, invocações por passo e categoria. Um `--phase`
desconhecido termina com o código `2` e lista as fases que o perfil tem.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um perfil. Obrigatório |
| `--top` | `20` | Mostra os N principais por tempo de CPU |
| `--phase` | | Restringe a uma fase |
| `--json` | `false` | Saída JSON no stdout |

A visão do framework em vez da visão do dispositivo: ops de `aten` e do
autograd ordenadas por tempo de CPU, que é onde aparece o custo de lançamento
do host.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `before` | | Posicional. Perfil de referência. Obrigatório |
| `after` | | Posicional. Perfil novo. Obrigatório |
| `--json` | `false` | Saída JSON no stdout |

Compara throughput, milissegundos por imagem, utilização da GPU, overhead do
host, lançamentos de kernel por passo e o veredito sobre o gargalo.

O julgamento de significância precisa que os dois lados tenham sido medidos com
`--repeat` de pelo menos 2. Com isso, uma diferença conta como significativa
quando supera o dobro do erro padrão combinado, e a saída imprime a comparação
que fez. Sem isso, a linha diz que uma única execução não sustenta o
julgamento.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argumento | Padrão | Significado |
|---|---|---|
| `trace` | | Posicional. Caminho para um perfil. Obrigatório |
| `--remove-category` | | Projeta a remoção de uma categoria de kernels: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Projeta a remoção de N lançamentos de kernel por passo, por exemplo um ganho de fusão de ops |
| `--json` | `false` | Saída JSON no stdout |

Estima o que uma mudança traria antes de a mudança ser escrita. Uma das duas
opções é obrigatória; nenhuma delas termina com o código `2`.

A projeção segue o veredito do próprio perfil. Abaixo de 80% de utilização da
GPU, modela a economia como menos lançamentos vezes o custo de host por
lançamento medido; acima disso, como menos trabalho de GPU. O resultado carrega
um campo de ressalva, porque o custo por lançamento é uma aproximação e a única
prova é uma segunda medição.

## Exemplos

<code-tabs name="examples" />

## Notas

O profiler mede e informa. Ele não muda nada: ler o veredito, editar a
configuração ou o código, rodar de novo e comparar é o ciclo para o qual ele
foi construído.

`--device` vale `0` por padrão, que é o dispositivo CUDA 0. Passar
`--device cpu` mede na CPU e produz um perfil que os subcomandos de leitura
continuam aceitando, sem o detalhe de kernels de GPU.

Todo subcomando aceita `--json`, e os de leitura imprimem apenas no stdout, que
é o que torna o grupo utilizável a partir de um script.

Os códigos de saída aqui são os do próprio grupo: `2` para um arquivo que não
existe ou um argumento que não resolve, `3` quando `run` não produziu nenhum
perfil, e `1` quando um trace não pode ser analisado.

Relacionado: [`libreyolo train`](/docs/cli/train), cujos argumentos são o que um
perfil de treinamento costuma ser tomado para ajustar.
