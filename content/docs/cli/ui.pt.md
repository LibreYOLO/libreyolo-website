---
title: libreyolo ui
seo_title: "referência do comando libreyolo ui"
description: "Sobe a interface web local de inferência: endereço de bind, comportamento da porta, escolha de dispositivo e como o comando termina."
lead: "Sobe um servidor web local que aceita imagens arrastadas ou coladas, roda nelas o modelo que você escolher e mostra os resultados no navegador."
keywords: [libreyolo ui cli, interface web libreyolo, inferência local no navegador, inferência arrastando imagem, porta libreyolo ui]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo ui
    mono: true
  - label: Saída
    value: "Uma URL de servidor no stdout e, em seguida, o processo fica em primeiro plano"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo ui
    - label: Porta fixa, sem navegador
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: Na CPU, saída legível por máquina
      language: bash
      code: |
        libreyolo ui device=cpu json=true
---

## Sinopse

```bash
libreyolo ui [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`port=9000` e `--port 9000` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `host` | `127.0.0.1` | Host ou interface para fazer bind |
| `port` | `8000` | Porta para fazer bind. Sobe para a próxima livre se estiver ocupada |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Não abrir o navegador automaticamente |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silenciar o stderr |
| `verbose` | `false` | Saída detalhada no stderr |

## Exemplos

<code-tabs name="examples" />

## Notas

O bind padrão é loopback, então a interface só é acessível a partir desta
máquina.

Se a porta pedida estiver em uso, o comando tenta a seguinte e continua subindo
até vinte portas acima da que você pediu. Se todas as vinte falharem, ele sai
com `io_error` e a sugestão de passar outra porta. A URL impressa no stdout é a
da porta que foi realmente vinculada, então leia essa URL em vez de assumir a
porta que você pediu.

A menos que você use `no_browser=true`, uma aba do navegador abre nessa URL
logo depois do bind.

O comando então serve em primeiro plano até o Ctrl+C, que desliga o servidor de
forma limpa. Não existe modo desacoplado; mande o comando para segundo plano
com o seu shell se quiser o terminal de volta.

`json=true` imprime a URL e o dispositivo como um único objeto com
`schema_version` antes de o servidor subir, que é como um script descobre a
porta vinculada.

Relacionados: [`libreyolo label`](/docs/cli/label) para desenhar caixas e salvar
rótulos, [`libreyolo monitor`](/docs/cli/monitor) para acompanhar treinamentos.
Os dois são servidores web locais com o mesmo comportamento de porta e
navegador.
