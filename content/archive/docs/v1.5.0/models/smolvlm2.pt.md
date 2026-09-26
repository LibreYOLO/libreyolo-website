---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 no LibreYOLO: detecção de vocabulário aberto'
description: >-
  SmolVLM2 no LibreYOLO: instalação, definição de um vocabulário aberto e
  predição ou chat com o modelo de visão e linguagem Apache-2.0 da Hugging Face.
lead: >-
  O SmolVLM2 é o pequeno modelo de visão e linguagem da Hugging Face. O
  LibreYOLO o envolve como detector de objetos de vocabulário aberto e expõe seu
  chat livre diretamente: passe uma lista de classes para detectar, ou faça uma
  pergunta a ele.
keywords:
  - SmolVLM2
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - modelo multimodal pequeno
  - smolvlm2 python
  - detectar objetos sem treinar
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # A via de escape sob a comodidade da detecção: qualquer pergunta,
        # não apenas uma consulta de bounding box.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Instalação

O SmolVLM2 pertence ao nível VLM-como-detector do LibreYOLO, uma superfície de
produto separada das famílias baseadas em checkpoints e com sua própria factory.
Ele precisa do extra `vlm`, que também traz junto o `num2words`, uma dependência
do próprio processador do SmolVLM2.

```bash
pip install "libreyolo[vlm]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

Esta família é carregada pela factory `LibreVLM()`, não por `LibreYOLO()`: as
famílias VLM não declaram nenhum carregador de checkpoint, então o roteamento
por sufixo de arquivo descrito em outras páginas de modelos não se aplica aqui.
`set_classes()` define o vocabulário que o SmolVLM2 deve encontrar; ele é
persistente, então continua valendo em todas as chamadas posteriores a
`predict()`/`track()` até que você o defina de novo. O SmolVLM2 não precisa de
nenhum override de parser no LibreYOLO: ele segue a mesma saída de template de
chat mais JSON do padrão compartilhado do nível, então seu prompt de detecção e
seu formato de box não são específicos da família. Todas as detecções carregam a
mesma confiança de placeholder, então filtrar por `conf` é tudo ou nada em vez de
uma ordenação; `iou` tem efeito, e descarta um box posterior da mesma classe
assim que ele se sobrepõe a um já mantido acima do limiar, já que um gerador
repetitivo pode emitir boxes quase duplicados para um mesmo objeto. O SmolVLM2
também responde perguntas livres através de `chat()`, a mesma via de escape
documentada na factory `LibreVLM`. A CLI do LibreYOLO não cobre este nível: não
existe uma forma `libreyolo predict model=...` para ele. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Um único tamanho no registro: SmolVLM2-500M-Video-Instruct, carregado como
`LibreVLM("smolvlm2-500m")`. O SmolVLM2 é um detector mais fraco que os modelos
de grounding feitos sob medida deste nível; o próprio wrapper do LibreYOLO o
descreve como uma demonstração de que uma família nova não precisa de parsing de
caso especial para funcionar aqui, e não como sua opção mais forte de
vocabulário aberto.

O LibreYOLO não treina, valida nem exporta o SmolVLM2: `train()`, `val()` e
`export()` levantam `NotImplementedError` em todas as famílias deste nível (veja
o nível de suporte acima). Faça fine-tuning do SmolVLM2 upstream e carregue os
pesos resultantes se você precisar de um vocabulário personalizado embutido;
confira a olho a saída de `predict()` em vez de uma passada de validação no
estilo COCO, já que todas as detecções carregam a mesma confiança de placeholder.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
