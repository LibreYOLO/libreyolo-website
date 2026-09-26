---
title: API de visão e linguagem
seo_title: 'API do LibreVLM: aliases, set_classes e chat'
description: >-
  A factory LibreVLM, todos os aliases, o vocabulário persistente de
  set_classes, set_task, a válvula de escape do chat e por que a confiança é um
  placeholder.
lead: >-
  O LibreVLM carrega um modelo generativo de visão e linguagem e o dirige como
  um detector de objetos. A lista de classes é um prompt em vez de uma cabeça
  fixa, e o modelo retorna os mesmos Results que qualquer outra família retorna.
keywords:
  - LibreVLM
  - detecção de objetos com modelo de visão e linguagem
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Aliases lidos de libreyolo/models/vlm/__init__.py; repositórios, tamanhos e
  listas de tarefas dos módulos de família em libreyolo/models/vlm/ mais
  libreyolo/models/sensenova/model.py; regras de chamada e exceções levantadas
  de libreyolo/models/vlm/base.py, tudo na v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Detectar um vocabulário aberto
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Fazer uma pergunta livre
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Instalação

O tier precisa do extra `vlm`.

<code-tabs name="install" />

## A factory

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` é um alias, não um caminho. `**kwargs` chega ao construtor da família,
que aceita `device`, `names` (o vocabulário inicial, equivalente a chamar
`set_classes` depois do carregamento), `prompt` (sobrescreve o prompt de
detecção) e `max_new_tokens`. Um alias desconhecido levanta `ValueError`
listando todos os aliases.

<code-tabs name="usage" />

## Aliases

| Família | Aliases | Tamanhos | Pesos |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot upstream fixado |

O alias padrão é `qwen3-vl-4b`. Os tamanhos do alias padrão de cada família são
os listados primeiro: `qwen3-vl` resolve para `4b`, `lfm2-vl` para `450m`,
`internvl3` para `2b`, `smolvlm2` para `2.2b`, `florence-2` para `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` e `LibreMODUS`
(também escrito `LibreModus`) são exportados no nível do pacote.

## Tarefas

A maioria das famílias serve só `detect`. Duas servem mais:

| Família | Tarefas suportadas |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Como a tarefa é dirigida por prompt em vez de embutida em um checkpoint, ela
pode ser trocada em um modelo já carregado:

```python
model.set_task(task: str) -> LibreVLMModel
```

A tarefa é validada contra a lista de suportadas da família, é persistente nas
chamadas posteriores de `predict()` e `track()`, e o modelo é retornado para que
as chamadas possam ser encadeadas.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Define o vocabulário aberto. Qualquer palavra funciona, porque o modelo é
instruído por prompt com elas em vez de ficar restrito a uma cabeça fixa. A
lista precisa ser não vazia, e as entradas precisam ser únicas quando comparadas
sem diferenciar maiúsculas de minúsculas. Passar uma string solta levanta
`TypeError`, porque ela seria enumerada em classes de um caractere só. O
vocabulário é persistente: defina uma vez depois de carregar e ele permanece até
ser definido de novo.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Geração multimodal crua: imagem e prompt entram, texto decodificado sai, na
íntegra. Esta é a válvula de escape sob a conveniência da detecção, para
perguntas livres, contagem ou um formato de saída que o wrapper de detecção não
cobre. `max_new_tokens` recai no `MAX_NEW_TOKENS` da família, que é 1024 na
classe base. A decodificação é gulosa, com uma penalidade de repetição leve.

## Confiança

A saída gerada não tem confiança calibrada por box. Esta versão atribui um
placeholder constante para que `predict`, o desenho e o `track` funcionem, o que
torna a filtragem por `conf=` e o mAP frouxos em vez de significativos. É também
por isso que `val()` levanta erro: mAP no estilo COCO sobre pontuações de
placeholder enganaria.

## Predict e track

A superfície padrão de predict se aplica, e `track()` funciona, então um
detector VLM se encaixa no mesmo pipeline que qualquer outra família. Duas
políticas de nível de classe diferem de um detector convolucional: o data
augmentation em tempo de teste fica desativado, porque augmentation em múltiplas
escalas não faz sentido para um gerador de resolução fixa, e o predict em batch
fica desligado, porque a geração é autorregressiva e o pré-processamento retorna
uma codificação de texto e imagem em vez de um tensor de imagem empilhável.

## Não suportado

`train()`, `val()` e `export()` levantam `NotImplementedError`. Faça o
fine-tuning upstream e carregue os pesos resultantes.

## Código remoto

Toda família distribuída carrega por uma classe de modelo nativa, então o
LibreYOLO não executa código de repositório de terceiros por padrão. Uma família
que realmente precise disso tem que optar por ele explicitamente e fixar uma
revisão de snapshot; o LocateAnything é a que faz isso, fixado no commit
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

O LibreMODUS é uma exceção explícita ao schema de checkpoint: o alias dele
resolve para um diretório de arquivos upstream fixados em vez de um `.pt` do
LibreYOLO, e o LibreYOLO não adiciona metadados v1.0 a ele nem o republica.
