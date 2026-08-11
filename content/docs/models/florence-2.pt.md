---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 no LibreYOLO: detecção de vocabulário aberto'
description: >-
  Florence-2 no LibreYOLO: instalação, definição de um vocabulário aberto e
  predição de caixas com o modelo de visão da Microsoft, licenciado sob MIT.
lead: >-
  O Florence-2 é o modelo fundacional de visão da Microsoft, guiado por um token
  de tarefa em vez de passar por uma cabeça de detecção fixa. O LibreYOLO o
  envolve como detector de objetos de vocabulário aberto: a lista de classes é
  informada na hora de predizer.
keywords:
  - Florence-2
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - detectar objetos sem treinar
  - florence 2 python
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice
        # de webcam, stream RTSP ou uma lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Instalação

O Florence-2 pertence ao nível VLM-como-detector do LibreYOLO, uma superfície de
produto separada das famílias baseadas em checkpoints e com sua própria fábrica.
Ele precisa do extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente. O LibreYOLO baixa o reenvio do checkpoint feito pela
florence-community em vez do repositório original `microsoft/Florence-2-*`; veja
Licenciamento para saber por quê.

<code-tabs name="predict" />

Esta família é carregada pela fábrica `LibreVLM()`, e não por `LibreYOLO()`: as
famílias VLM não declaram nenhum carregador de checkpoint, então o roteamento por
sufixo de arquivo descrito em outras páginas de modelos não se aplica aqui.
`set_classes()` define o vocabulário que o Florence-2 deve procurar na imagem;
ele é persistente, então continua valendo em todas as chamadas posteriores a
`predict()`/`track()` até você defini-lo de novo. O `Results` devolvido traz
`boxes` no mesmo formato de qualquer outra família, mas toda detecção carrega a
mesma confiança de placeholder, então filtrar por `conf` é tudo ou nada em vez de
uma ordenação, e `iou` não tem efeito: o wrapper do Florence-2 monta a lista de
detecções diretamente a partir da saída parseada do token de tarefa, sem nenhuma
etapa de deduplicação. Aqui `chat()` lança `NotImplementedError`, porque o
Florence-2 é guiado pelo token de tarefa `<OPEN_VOCABULARY_DETECTION>` e não por
um template de chat. A CLI do LibreYOLO não cobre este nível: não existe uma
forma `libreyolo predict model=...` para ele. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Dois tamanhos: Florence-2-base e Florence-2-large, ambos a 768 px, carregados
como `LibreVLM("florence-2-base")` ou `LibreVLM("florence-2-large")`. O LibreYOLO
não publicou nenhum benchmark comparando a acurácia entre eles.

O LibreYOLO não treina, valida nem exporta o Florence-2: `train()`, `val()` e
`export()` lançam `NotImplementedError` em todas as famílias deste nível (veja o
nível de suporte acima). Faça fine-tuning do Florence-2 upstream e carregue os
pesos resultantes se você precisar de um vocabulário personalizado embutido;
confira a olho a saída de `predict()` em vez de recorrer a uma passada de
validação no estilo COCO, já que toda detecção carrega a mesma confiança de
placeholder.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
