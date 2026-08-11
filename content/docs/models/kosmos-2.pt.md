---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 no LibreYOLO: detecção de objetos com grounding'
description: >-
  Kosmos-2 no LibreYOLO: instale, defina um vocabulário aberto e preveja caixas
  com grounding usando o modelo da Microsoft licenciado sob MIT.
lead: >-
  O Kosmos-2 é o modelo de grounding da Microsoft: ele gera uma legenda para a
  imagem e depois localiza com uma caixa cada sintagma nominal dessa legenda. O
  LibreYOLO o envolve como um detector de objetos de vocabulário aberto: a lista
  de classes é informada na hora de prever.
keywords:
  - Kosmos-2
  - modelo de visão e linguagem
  - grounding
  - detecção de vocabulário aberto
  - detectar objetos sem treinar
  - kosmos 2 python
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice
        # de webcam, stream RTSP ou uma lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Instalação

O Kosmos-2 pertence ao nível VLM-como-detector do LibreYOLO, uma superfície de
produto separada das famílias baseadas em checkpoint e com sua própria factory.
Ele precisa do extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local. O
LibreYOLO carrega diretamente o repositório `microsoft/kosmos-2-patch14-224` da
própria Microsoft; ao contrário do Florence-2, aqui não é preciso nenhum reenvio
da comunidade.

<code-tabs name="predict" />

Esta família é carregada pela factory `LibreVLM()`, não por `LibreYOLO()`: as
famílias VLM não declaram nenhum carregador de checkpoint, então o roteamento
por sufixo de arquivo descrito em outras páginas de modelos não se aplica aqui.
`set_classes()` define o vocabulário que o Kosmos-2 deve encontrar; ele é
persistente, então continua valendo em todas as chamadas posteriores a
`predict()`/`track()` até que você o defina de novo. O Kosmos-2 faz o grounding
de sintagmas nominais em vez de casar exatamente com um rótulo, então o wrapper
do LibreYOLO aceita uma correspondência parcial: uma classe chamada `"boat"`
também casa com um sintagma gerado como "the boats". Todas as detecções carregam
a mesma confiança de placeholder, então filtrar por `conf` é tudo ou nada em vez
de uma ordenação, e `iou` não tem efeito aqui, já que o wrapper monta a lista de
detecções diretamente a partir das entidades com grounding, sem nenhuma etapa de
deduplicação. `chat()` levanta `NotImplementedError`, porque o Kosmos-2 é
controlado por um prompt `<grounding>` e não por um template de chat. A CLI do
LibreYOLO não cobre este nível: não existe uma forma
`libreyolo predict model=...` para ele. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Um único tamanho: `kosmos-2-patch14-224`, a 224 px, carregado como
`LibreVLM("kosmos-2")`. É um modelo da época de 2023, e o próprio wrapper do
LibreYOLO observa que seu grounding é mais grosseiro que o dos detectores mais
novos deste nível.

O LibreYOLO não treina, valida nem exporta o Kosmos-2: `train()`, `val()` e
`export()` levantam `NotImplementedError` em todas as famílias deste nível (veja
o nível de suporte acima). Faça fine-tuning do Kosmos-2 upstream e carregue os
pesos resultantes se precisar de um vocabulário personalizado embutido; confira
a saída de `predict()` no olho, em vez de uma passagem de validação no estilo
COCO, já que todas as detecções carregam a mesma confiança de placeholder.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
