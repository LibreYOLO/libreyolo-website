---
title: libreyolo monitor
seo_title: "Referência do comando libreyolo monitor"
description: "Serve um dashboard ao vivo para execuções de treinamento: argumentos com seus padrões, o que o servidor lê do disco e como um servidor cobre várias execuções."
lead: "Serve um dashboard web para execuções de treinamento, lendo os artefatos que uma execução grava no disco. Ele nunca se conecta ao processo de treinamento, então execuções ao vivo, concluídas e que quebraram aparecem do mesmo jeito."
keywords: [libreyolo monitor cli, dashboard de treinamento, acompanhar treinamento yolo, libreyolo monitor porta, visualizar métricas de treinamento]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo monitor
    mono: true
  - label: Saída
    value: "A URL do servidor no stdout, e então o processo fica em primeiro plano"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Observa runs/ e lista todas as execuções dentro dele.
        libreyolo monitor
    - label: Outra raiz de execuções
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: Uma execução, porta fixa, sem navegador
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
---

## Sinopse

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

O diretório é posicional. Todo o resto é um par `key=value`, e a forma POSIX
também funciona, então `port=9100` e `--port 9100` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `run_dir` | `runs` | Posicional. Uma raiz de execuções para observar, ou um único diretório de execução para abrir direto. De qualquer forma, todas as execuções abaixo da raiz são listadas |
| `host` | `127.0.0.1` | Host ou interface para fazer o bind |
| `port` | `8420` | Porta para fazer o bind. Pula para a próxima livre se estiver ocupada |
| `no_browser` | `false` | Não abrir o navegador automaticamente |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Suprime o stderr |
| `verbose` | `false` | Saída detalhada no stderr |

## Exemplos

<code-tabs name="examples" />

## Notas

### Um servidor, várias execuções

O servidor observa uma raiz de execuções em vez de uma única execução, e endereça
cada execução por URL, então várias execuções na mesma máquina compartilham uma
porta. Abra a URL da raiz para ver o índice, ou uma aba por execução; o parâmetro
`?run=` em cada URL identifica qual é.

Apontar o comando para um único diretório de execução enraíza o servidor no
diretório pai dele, então as execuções irmãs continuam aparecendo no índice, e o
link vai direto para a que foi nomeada.

### O que ele lê

O dashboard é montado a partir dos arquivos que o `libreyolo train` grava:
`status.json`, `metrics.jsonl`, `train.log` e as imagens da execução. Nada é lido
do próprio processo de treinamento, então uma execução que terminou, ou que
morreu, aparece exatamente como uma ao vivo.

### Pré-condições e portas

Pelo menos uma execução já precisa existir. Sem argumento e sem um diretório
`runs/`, o comando sai com `source_not_found`; o mesmo acontece quando o diretório
informado não contém nenhuma execução.

Uma porta ocupada passa para a seguinte, até vinte além da que foi pedida. Se as
vinte falharem, o comando sai com `io_error`. A URL impressa no stdout é a porta
que foi realmente vinculada.

O comando serve em primeiro plano até Ctrl+C. `json=true` imprime a URL, a raiz
que está sendo observada e o número de execuções encontradas, como um único
objeto com `schema_version`.

Relacionado: [`libreyolo train`](/docs/cli/train), cujos argumentos `project` e
`name` decidem onde esses diretórios de execução ficam.
