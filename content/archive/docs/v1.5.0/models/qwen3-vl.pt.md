---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL no LibreYOLO: detecção de vocabulário aberto'
description: >-
  Qwen3-VL no LibreYOLO: instalação, definição de um vocabulário aberto e
  predição ou chat com o modelo de visão e linguagem Apache-2.0 da Alibaba.
lead: >-
  O Qwen3-VL é o modelo de visão e linguagem da Alibaba, com grounding 2D
  nativo. O LibreYOLO o envolve como detector de objetos de vocabulário aberto e
  expõe seu chat livre diretamente: passe uma lista de classes para detectar, ou
  faça uma pergunta a ele.
keywords:
  - Qwen3-VL
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - qwen3 vl python
  - detectar objetos sem treinar
  - grounding
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # A via de escape sob a comodidade da detecção: qualquer pergunta,

        # não apenas uma consulta de bounding box.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Instalação

O Qwen3-VL pertence ao nível VLM-como-detector do LibreYOLO, uma superfície de
produto separada das famílias baseadas em checkpoints e com sua própria factory.
Ele precisa do extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente. `LibreVLM()`, chamado sem nenhum argumento, usa o Qwen3-VL-4B por
padrão.

<code-tabs name="predict" />

Esta família é carregada pela factory `LibreVLM()`, não por `LibreYOLO()`: as
famílias VLM não declaram nenhum carregador de checkpoint, então o roteamento
por sufixo de arquivo descrito em outras páginas de modelos não se aplica aqui.
`set_classes()` define o vocabulário que o Qwen3-VL deve encontrar; ele é
persistente, então continua valendo em todas as chamadas posteriores a
`predict()`/`track()` até que você o defina de novo. Todas as detecções carregam
a mesma confiança de placeholder, então filtrar por `conf` é tudo ou nada em vez
de uma ordenação; `iou` tem efeito nesta família, e descarta um box posterior da
mesma classe assim que ele se sobrepõe a um já mantido acima do limiar, já que um
gerador repetitivo pode emitir boxes quase duplicados para um mesmo objeto.
Diferente do Florence-2 e do Kosmos-2, o Qwen3-VL também responde perguntas
livres através de `chat()`, a mesma via de escape documentada na factory
`LibreVLM`. A CLI do LibreYOLO não cobre este nível: não existe uma forma
`libreyolo predict model=...` para ele. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Três tamanhos: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct e Qwen3-VL-8B-Instruct,
carregados como `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` e
`LibreVLM("qwen3-vl-8b")`. Os três declaram uma entrada nominal de 1024 px, mas é
o próprio smart-resize do processador do Qwen que decide o canvas real passado
para a rede, então esse número não é uma resolução de trabalho fixa como é nas
outras famílias deste site. O LibreYOLO não publicou nenhum benchmark comparando
a acurácia entre os três tamanhos.

O LibreYOLO não treina, valida nem exporta o Qwen3-VL: `train()`, `val()` e
`export()` levantam `NotImplementedError` em todas as famílias deste nível (veja
o nível de suporte acima). Faça fine-tuning do Qwen3-VL upstream e carregue os
pesos resultantes se você precisar de um vocabulário personalizado embutido;
confira a olho a saída de `predict()` em vez de uma passada de validação no
estilo COCO, já que todas as detecções carregam a mesma confiança de placeholder.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
