---
title: libreyolo label
seo_title: Referência do comando libreyolo label
description: >-
  Abra a ferramenta local de anotação de bounding boxes: argumentos com seus
  padrões, a chave de assistência por IA e o que expor a ferramenta em uma
  interface de rede significa.
lead: >-
  Inicia uma ferramenta web local para desenhar e editar bounding boxes. Ela
  escreve arquivos de rótulos no formato nativo do LibreYOLO, então um dataset
  anotado aqui treina sem nenhuma etapa de conversão.
keywords:
  - libreyolo label cli
  - ferramenta de anotação bounding box
  - rotular dataset yolo
  - rotulagem automática cli
  - compartilhar libreyolo label
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo label
    mono: true
  - label: Saída
    value: >-
      Uma URL de servidor no stdout; os rótulos são escritos como labels/*.txt
      junto às imagens
snippets:
  examples:
    - label: Básico
      language: bash
      code: >
        # Abre a página inicial do projeto; escolha ou crie um dataset no
        navegador.

        libreyolo label
    - label: 'Somente manual, porta fixa'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Deixar os colegas entrarem
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Sinopse

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`port=9200` e `--port 9200` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `data` | | YAML ou pasta do dataset a abrir diretamente. Começa na página inicial do projeto quando não é informado |
| `host` | `127.0.0.1` | Host ou interface em que fazer o bind |
| `port` | `8000` | Porta em que fazer o bind. Passa para a próxima livre se estiver ocupada |
| `device` | `auto` | Dispositivo para a rotulagem automática por IA: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Desativa a rotulagem automática por IA, deixando um rotulador manual |
| `no_browser` | `false` | Não abrir o navegador automaticamente |
| `share` | `false` | Faz o bind em `0.0.0.0` para que colegas na sua rede possam entrar |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `verbose` | `false` | Saída detalhada no stderr |

## Exemplos

<code-tabs name="examples" />

## Notas

### O que ele escreve

Os boxes são salvos como arquivos `labels/*.txt` no formato nativo do LibreYOLO,
que é o formato que o `libreyolo train` lê, então nada precisa ser convertido
depois. Esta versão trata apenas de bounding boxes. As edições são salvas
conforme você navega entre as imagens.

### Abrir um dataset

Sem `data`, a ferramenta começa na página inicial do projeto e o dataset é
escolhido ou criado pelo navegador. Passar `data=path/to/data.yaml` abre esse
dataset direto, e a linha de inicialização informa a quantidade de imagens, a
quantidade de classes e se o dataset é gravável. Um dataset somente leitura
também abre e diz por que não é possível escrever nele.

### Compartilhamento, e o que o `host` faz

`share=true` faz o bind no endereço curinga, o que deixa outras máquinas da sua
rede alcançarem a ferramenta enquanto as ações administrativas (trocar ou
apagar projetos e iniciar computação) ficam nesta máquina.

Definir `host` como uma interface específica faz algo diferente e menos seguro:
o host passa a ser indistinguível de um cliente de rede, então todo cliente
ganha direitos administrativos. O comando imprime um aviso no stderr quando você
faz isso. Prefira `share=true`.

### Portas e encerramento

Uma porta ocupada passa para a seguinte, até vinte acima da solicitada. Se as
vinte falharem, sai com `io_error`. A URL impressa no stdout é a porta que foi
realmente vinculada. Com `share=true`, o resultado traz também `lan_url`, o
endereço que os colegas devem abrir.

O comando serve em primeiro plano até o Ctrl+C.

Relacionado: [`libreyolo doctor`](/docs/cli/doctor) para conferir o dataset
rotulado antes de treinar, e [`libreyolo train`](/docs/cli/train) para treinar
com ele.
