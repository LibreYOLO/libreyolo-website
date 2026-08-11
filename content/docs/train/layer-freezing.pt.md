---
title: Congelamento de camadas
seo_title: Congele camadas durante o treinamento no LibreYOLO
description: >-
  Congele parte de um modelo para transfer learning: um número inteiro de grupos
  de congelamento da família, uma lista explícita de índices, ou seletores por
  nome de módulo e de parâmetro.
lead: >-
  O congelamento mantém fixos os pesos selecionados enquanto o resto do modelo
  treina. Os seletores apontam para os grupos de congelamento ordenados da
  própria família ou para os nomes dos seus módulos, e não para números brutos
  de camada de um grafo YAML.
keywords:
  - congelar camadas yolo
  - transfer learning yolo python
  - congelar backbone
  - frozen batchnorm
  - grupos de congelamento
  - treinar só a cabeça do modelo
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Os 10 primeiros grupos são todo o backbone do YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Por nome
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Vários seletores
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Listar em ordem os grupos de congelamento de uma família
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Congele alguma coisa

`freeze` é opcional e, por padrão, não congela nada.

<code-tabs name="train" />

O congelamento acontece depois que o modelo é construído e depois de qualquer
reconstrução da cabeça para um novo número de classes, e antes de o otimizador
ser criado, então o otimizador só recebe parâmetros treináveis.

## O que um seletor pode ser

| Valor | Significado |
|---|---|
| `None`, `False`, `""`, `"none"` | Treina todos os parâmetros |
| `10` ou `"10"` | Congela os dez primeiros grupos de congelamento da família |
| `[0, 3, 7]` | Congela esses grupos, com índice começando em zero |
| `"backbone"` | Congela o grupo, módulo ou prefixo de parâmetro correspondente |
| `["backbone", "neck"]` | Congela cada seletor da lista |
| `["backbone", 3]` | Listas mistas funcionam |

Uma string passa por parsing antes de ser interpretada, então a CLI e uma
configuração YAML aceitam os mesmos formatos que o Python. `freeze="[0, 3, 'head']"`
é lido como uma lista literal, `freeze="backbone,neck"` é dividido pela vírgula, e
uma string decimal simples vira uma contagem.

`freeze=True` é rejeitado por ser ambíguo.

Seletores por nome casam com o nome de um grupo de congelamento, o nome de um
módulo ou um prefixo de nome de parâmetro, e os caracteres de glob `*`, `?` e `[`
funcionam. Um `model.` no início é tratado com flexibilidade, então tanto
`backbone` quanto `model.backbone` acertam a grafia que a família usa
internamente.

## Os grupos são definidos pela família

Um inteiro aponta para a lista ordenada de grupos de congelamento da própria
família, não para uma posição em um grafo compartilhado. As famílias do LibreYOLO
não são todas um único modelo sequencial indexado por YAML, então um número bruto
de camada significaria algo diferente em cada uma delas.

O YOLOv9 ordena seus grupos a partir da entrada: dez estágios do backbone, depois
seis estágios do neck, depois a cabeça. É por isso que `freeze=10` é exatamente o
backbone. `backbone`, `neck` e `head` são seletores por nome estáveis sobre essa
ordem.

Os grupos do RF-DETR são `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` e `head`. Aqui os nomes são a melhor
escolha, porque componentes de transformer não correspondem a uma contagem de
camadas. `backbone` casa com os dois grupos de backbone por prefixo.

Famílias que não definem grupos semânticos recorrem a um padrão conservador: cada
filho direto do modelo que tenha ao menos um parâmetro, na ordem de declaração.
Isso costuma ser uma lista curta, então um inteiro grande não vai encontrar
grupos suficientes:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Para ver a lista real em vez de adivinhar:

<code-tabs name="groups" />

## As falhas são explícitas

Toda forma de errar isso levanta um erro em vez de treinar algo que você não
pediu.

Um seletor que não casa com nada levanta um erro, nomeando os seletores que não
acertaram:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Um congelamento que não deixaria nada treinável levanta um erro, tanto no momento
do congelamento quanto de novo quando o otimizador é construído:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Que é o que `freeze="all"` faz, já que `all` casa com todos os parâmetros.

Quando o congelamento dá certo, uma linha registra o que aconteceu:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## BatchNorm congelado para de atualizar

Um parâmetro congelado continua dentro de um módulo cujas estatísticas acumuladas
continuariam mudando. Todo módulo do tipo BatchNorm cujos parâmetros caem no
conjunto congelado é colocado em modo eval, e o trainer reaplica isso depois da
chamada de `model.train()` de cada época, então as estatísticas ficam fixas
durante toda a execução.

Isso vem ligado por padrão e é o que faz com que congelar um backbone realmente
congele o backbone.

## Combinando com LoRA

`freeze` e `lora=True` funcionam juntos. No RF-DETR, no DEIM e no ConvNeXt os
parâmetros dos adaptadores são preservados como treináveis mesmo quando o grupo
pai está congelado, que é a combinação que você quer: um backbone congelado com
adaptadores aprendendo em cima dele. Veja [Fine-tuning com LoRA](/docs/train/lora).

## Escopo

Este é um congelamento estático decidido na inicialização. Descongelamento
agendado e congelamento progressivo não fazem parte da interface.

## Relacionados

- [Hiperparâmetros](/docs/train/hyperparameters) para o resto de `train()`.
- [Destilação](/docs/train/distillation) para o outro jeito de levar o
  conhecimento de um modelo grande para um treinamento.
