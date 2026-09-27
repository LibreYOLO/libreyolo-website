---
title: libreyolo quantize
seo_title: referência do comando libreyolo quantize
description: >-
  Quantize um checkpoint no PyTorch pela linha de comando: receitas, argumentos
  de calibração, valores padrão e as famílias que cada receita aceita.
lead: >-
  Substitui os módulos float de um modelo por módulos quantizados, calibra esses
  módulos em imagens sem rótulos quando a receita precisa de estatísticas e
  salva o resultado como um checkpoint do PyTorch.
keywords:
  - libreyolo quantize cli
  - quantizacao int8 linha de comando
  - quantizacao fp8
  - quantizacao pos treinamento yolo
  - argumentos libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo quantize
    mono: true
  - label: Obrigatório
    value: model
    mono: true
  - label: Saída
    value: 'O caminho de origem com -<recipe> antes do sufixo, ex. LibreYOLO9s-int8.pt'
    mono: true
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Calibra em coco128 e escreve LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Só o cast, sem calibração'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: Calibração mais ampla e depois recuperação
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # O treinamento consciente da quantização sobre o checkpoint quantizado
        recupera acurácia.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Sinopse

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Os argumentos são pares `key=value`, e a forma POSIX também funciona, então
`recipe=int8` e `--recipe int8` são o mesmo argumento.

## Argumentos

| Argumento | Padrão | Significado |
|---|---|---|
| `model` | | Pesos do modelo `.pt`. Obrigatório |
| `recipe` | `int8` | Receita de quantização: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Imagens de calibração: um YAML de dados ou o nome de um dataset embutido. Sem rótulos, apenas forward. `none` pula a calibração |
| `samples` | `128` | Máximo de imagens de calibração |
| `batch` | `8` | Tamanho de batch da calibração |
| `algorithm` | `auto` | Estimativa do intervalo das ativações: `auto`, que seleciona minmax, ou `minmax`, ou `percentile` |
| `out` | | Caminho do checkpoint de saída. Por padrão, o caminho de origem com `-<recipe>` antes do sufixo |
| `device` | `auto` | Dispositivo |
| `allow_download_scripts` | `false` | Permite Python embutido nos blocos de download do YAML do dataset |
| `json` | `false` | Saída JSON no stdout |
| `quiet` | `false` | Silencia o stderr |
| `help_json` | `false` | Despeja o esquema do comando como JSON e sai |

## Exemplos

<code-tabs name="examples" />

## Notas

### Quais famílias aceitam

A quantização cobre quatro famílias: `yolo9`, `rfdetr`, `birefnet` e
`feynobg`. Qualquer outra família sai com `quantize_failed`, que carrega a lista.

### O que cada receita toca

`fp16` e `bf16` são casts. Mudam só o dtype, não precisam de calibração e
`calib=none` é o ajuste certo para elas.

`int8` e `fp8` quantizam os módulos `Conv2d` e `Linear`, e é por isso que
combinam com as famílias convolucionais.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2` quantizam apenas `nn.Linear`, então
miram as famílias transformer. Pedir uma delas em `yolo9` é recusado com uma
explicação em vez de produzir em silêncio um modelo não quantizado, já que ali a
aceleração abaixo de 8 bits é só de GEMM e as convoluções ficariam em precisão
mais alta.

`int8`, `fp8`, `w4a8` e `int2` precisam de estatísticas de calibração para suas
ativações. `int2` ainda precisa de treinamento depois para se recuperar, por isso
é recusado em `birefnet` e `feynobg`, que não têm treinador.

Cada família mantém um conjunto de módulos em float seja qual for a receita: as
primeiras camadas, as cabeças de predição e, no YOLOv9, a convolução DFL, que é
um operador de esperança integral fixo que não pode ser quantizado.

### Dados de calibração não são dados de treinamento

`calib` aponta para um pequeno conjunto de imagens sem rótulos, usado apenas no
forward, para derivar os intervalos de ativação. Nada é avaliado contra ele e
seus rótulos nunca são lidos. O `coco128.yaml` padrão baixa no primeiro uso a
partir de uma URL, então não precisa de permissão extra; um YAML com um script
de download em Python embutido precisa de `allow_download_scripts=true`.

`algorithm=percentile` está disponível e pode reduzir a acurácia nas famílias
transformer, e é por isso que `auto` seleciona minmax.

### Recuperar a acurácia

A saída é um checkpoint normal do PyTorch, então
[`libreyolo train`](/docs/cli/train) o aceita diretamente. Treinar um checkpoint
quantizado é treinamento consciente da quantização; adicionar
`distill_model=<teacher>` transforma isso em destilação consciente da
quantização.

### Saída e códigos de saída

O resultado imprime o caminho salvo, a receita, o modo de execução, se a
calibração rodou e a contagem de módulos trocados por tipo. O código de saída é
`0` em caso de sucesso, `4` quando o modelo não pode ser carregado, `5` quando a
quantização ou o salvamento falha, e `1` para outras falhas em tempo de execução.

Relacionado: [`libreyolo export`](/docs/cli/export), que sai do PyTorch e escreve
um artefato de deploy em vez disso.
