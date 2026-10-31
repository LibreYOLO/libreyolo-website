---
title: API de vocabulário aberto
seo_title: 'API LibreOpenVocab: aliases e argumentos'
description: >-
  A factory LibreOpenVocab, suas quatro famílias e todos os aliases,
  set_classes, os padrões de conf por família e as regras de text_threshold e
  iou.
lead: >-
  LibreOpenVocab é a factory dos detectores condicionados por texto. A lista de
  classes é um prompt em vez de uma cabeça fixa, então o vocabulário é definido
  por set_classes e o modelo retorna Results de detecção comuns em relação a
  ele.
keywords:
  - LibreOpenVocab
  - detecção de vocabulário aberto python
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Aliases lidos de libreyolo/models/openvocab/__init__.py; repositórios,
  tamanhos e limiares de grounding_dino.py, owlv2.py, omdet_turbo.py e
  ov_deim.py; regras de chamada de libreyolo/models/openvocab/base.py, tudo na
  v1.5.0. Intenção de projeto em docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Instalação

Este tier precisa do extra `openvocab`.

<code-tabs name="install" />

## A factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` é um alias, não um caminho. Underscores viram hífens antes da busca,
então os nomes qualificados por família que o inventário da CLI imprime, como
`omdet_turbo-t` e `grounding_dino-t`, carregam do jeito que estão. Um alias
desconhecido levanta `ValueError` listando todos os aliases conhecidos.

O construtor aceita `size`, `nb_classes=80`, `names=None`, `device="auto"`,
`task=None` e `text_threshold=None`. Passar `names` é o mesmo que chamar
`set_classes` logo depois de carregar. Passar `text_threshold` para uma família
que não o suporta levanta `TypeError`.

<code-tabs name="usage" />

## Famílias e aliases

| Família | Aliases | Tamanhos | Pesos |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

O alias padrão é `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` e `LibreOMDetTurbo` são exportados no nível
do pacote e podem ser construídos diretamente com `size=`. O OV-DEIM é
acessível pelos aliases da factory acima.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Define o vocabulário para toda chamada `predict()` posterior e retorna o modelo
para que as chamadas possam ser encadeadas. A lista precisa ser não vazia,
conter apenas strings e ter entradas únicas quando comparadas sem diferenciar
maiúsculas de minúsculas; rótulos em branco são rejeitados. Passar uma string
solta levanta `TypeError`, porque ela seria enumerada em classes de um
caractere.

Depois da chamada, `model.names` mapeia `0..N-1` para os rótulos na ordem dada,
e `model.nb_classes` é `N`.

## Argumentos de chamada

Este tier reaproveita a superfície padrão de predict com três diferenças.

`conf` assume por padrão o valor da própria família, e não o 0.25
compartilhado:

| Família | conf padrão | Supressão |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Pós-processamento próprio, limiar 0.5, respeita `iou=` |
| OV-DEIM | 0.25 | Correspondência um-para-um com seleção top-K, sem supressão |

`iou=` só significa alguma coisa para uma família que roda supressão. O
OMDet-Turbo recebe o limiar como argumento e usa 0.5 por padrão quando `iou=`
não é definido. As outras três não suprimem nada, então passar `iou=` nelas
emite um aviso e é ignorado.

`text_threshold=` é exclusivo do Grounding DINO, onde o padrão é 0.25. Pode ser
passado na construção para um valor persistente, ou por chamada. Um valor por
chamada não pode ser combinado com `stream=True`, porque os resultados em
stream são gerados de forma preguiçosa; nesse caso, defina o valor no
construtor. Toda outra família levanta `TypeError` para ele.

`imgsz=` levanta `ValueError`: o pipeline de pré-processamento é o dono do
redimensionamento neste tier. `augment=True` também levanta, já que o data
augmentation em tempo de teste está fora de escopo aqui. Os tamanhos de entrada
são registrados por família apenas para referência: Grounding DINO 800, OWLv2
960 e 1008, OMDet-Turbo 640, OV-DEIM 640.

## Sem suporte

`train()`, `val()`, `track()` e `export()` levantam todos
`NotImplementedError`. Faça fine-tuning upstream e carregue os pesos
resultantes; rode `predict()` por quadro no lugar do tracking. A validação
precisaria de um validador dedicado, porque o validador de detecção
compartilhado chama o modelo com tensores de imagem, enquanto este tier exige
entradas condicionadas por texto.
