---
title: libreyolo doctor
seo_title: referência do comando libreyolo doctor
description: >-
  Confira um dataset de detecção antes de treinar: os argumentos com seus
  valores padrão, as famílias de verificações que você pode pular ou selecionar
  e os códigos de saída sobre os quais o CI pode falhar.
lead: >-
  Roda um conjunto de verificações de saúde sobre um dataset de detecção e
  informa o que atrapalharia um treinamento: arquivos faltando, rótulos
  quebrados, imagens corrompidas, vazamento entre splits e desbalanceamento de
  classes.
keywords:
  - libreyolo doctor cli
  - verificar saúde de dataset yolo
  - validar dataset de detecção
  - vazamento entre splits dataset
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo doctor
    mono: true
  - label: Obrigatório
    value: data
    mono: true
  - label: Saída
    value: Um relatório de achados no stdout. Sai com 1 quando há erros
snippets:
  examples:
    - label: Básico
      language: bash
      code: >
        # download=true permite que o coco8.yaml incluído baixe suas imagens se
        estiverem faltando.

        libreyolo doctor coco8.yaml download=true
    - label: 'Passada rápida, sem decodificar imagens'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Gate de CI sobre verificações selecionadas
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Sinopse

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

O dataset é posicional, e `data=<path>` é aceito como alternativa. Passar os
dois com valores diferentes termina com `config_conflict`. Todo o resto são
pares `key=value`, e a forma POSIX também funciona, então `imgsz=1024` e
`--imgsz 1024` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `data` | | Posicional. YAML do dataset no formato de detecção YOLO, p. ex. `coco8.yaml`. Obrigatório |
| `imgsz` | `640` | Tamanho de imagem de treinamento usado nas verificações baseadas em pixels, como a de objetos minúsculos |
| `fast` | `false` | Pula a decodificação das imagens, o que descarta as verificações de corrupção, duplicatas e vazamento |
| `skip` | | Ids de verificação ou famílias separados por vírgula que serão pulados, p. ex. `images,labels.tiny_object` |
| `only` | | Ids de verificação ou famílias separados por vírgula que serão executados com exclusividade |
| `strict` | `false` | Os avisos também fazem o código de saída falhar, para gates de CI |
| `download` | `false` | Permite o download do dataset por URL se ele estiver faltando. Nunca scripts |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `help_json` | `false` | Despeja o esquema do comando como JSON e sai |

### Famílias de verificações

`skip` e `only` aceitam tanto um id de verificação completo quanto um prefixo de
família, então `images` seleciona todas as verificações `images.*`.

| Família | Cobre |
|---|---|
| `config` | O próprio YAML do dataset: `names` ausente, `nc` frente a `names`, splits faltando, `path` que não pode ser resolvido, nomes de classe duplicados |
| `files` | O pareamento entre imagens e rótulos: rótulos faltando, imagens faltando, rótulos órfãos, extensões não suportadas, colisões de maiúsculas e minúsculas |
| `labels` | O conteúdo dos rótulos: sintaxe, linhas de polígono, ids de classe fora do intervalo, coordenadas fora do intervalo, bounding boxes degenerados, objetos minúsculos, boxes enormes, proporções de aspecto extremas, boxes duplicados, imagens lotadas, arquivos idênticos |
| `images` | Os dados de pixel: arquivos corrompidos, orientação EXIF, modos de cor incomuns, dimensões minúsculas ou extremas, imagens uniformes, duplicatas exatas e aproximadas |
| `splits` | Vazamento entre splits, exato e aproximado |
| `balance` | A distribuição de classes: classes com zero ou poucas instâncias, desbalanceamento, cobertura dos splits, proporção de fundo, viés entre splits |

## Exemplos

<code-tabs name="examples" />

## Notas

### Códigos de saída

`0` quando nenhum erro foi encontrado, `1` quando algum achado é um erro. Com
`strict=true`, os avisos também elevam o código de saída para `1`, que é a
configuração que um gate de CI quer.

Problemas de uso têm seus próprios códigos: `2` para um id de verificação ou
família desconhecidos em `skip` ou `only`, `3` quando o dataset não pode ser
encontrado e `3` quando o dataset não tem forma de detecção.

### A seleção é resolvida antes da varredura

`skip` e `only` são resolvidos contra o registro de verificações antes de
qualquer coisa ser lida do disco, então um erro de digitação falha na hora em
vez de falhar depois de uma longa passada pelas imagens. Um seletor que não
casa com nada é um erro, e a mensagem lista as famílias conhecidas.

Se a combinação de `skip`, `only` e `fast` não deixar nenhuma verificação para
rodar, isso também é um erro, e não uma passada silenciosa.

### Downloads

O dataset não é baixado a menos que `download=true`, e só downloads por URL são
feitos. Um script de download em Python embutido no YAML de um dataset nunca é
executado por este comando, seja qual for o valor da flag.

### Escopo

As verificações foram escritas para datasets de detecção. Um dataset cujos
rótulos têm forma de pose, de segmentação ou de caixa orientada é detectado e
recusado com `data_invalid` em vez de ser avaliado com as regras erradas.

### Saída

O relatório legível por pessoas vai para o stdout, e `json=true` o substitui por
um objeto estruturado que carrega as contagens do resumo, as estatísticas do
dataset, todos os achados e a lista das verificações que foram puladas.

Relacionado: [`libreyolo train`](/docs/cli/train), a execução antes da qual este
comando foi feito para rodar.
