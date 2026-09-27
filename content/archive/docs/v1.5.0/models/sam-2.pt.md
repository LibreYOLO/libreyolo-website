---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: segmentação de imagens com prompts no LibreYOLO'
description: >-
  Use o SAM 2 no LibreYOLO para segmentação com prompts de ponto e de caixa.
  Instale e faça predições com os checkpoints tiny, small, base-plus e large,
  sob Apache-2.0.
lead: >-
  O SAM 2 estende o SAM com uma arquitetura de memória em streaming construída
  para vídeo, e transforma um clique de ponto ou de caixa em uma máscara de
  objeto. O LibreYOLO suporta seu caminho de segmentação de imagens por meio de
  uma factory LibreSAM dedicada, separada da factory de detectores LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - segmentação com prompt
  - segmentação interativa python
  - segmentar objeto com um clique
  - prompt de ponto
  - prompt de caixa
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de ponto e de caixa
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Aliases de tamanho: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (também as formas curtas
        "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


        # Um prompt de ponto: [x, y] em coordenadas de pixel, label 1 = primeiro
        plano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # um polígono por máscara

        print(result.boxes.xyxy)    # caixa ajustada derivada da máscara


        # Um prompt de caixa em vez de um ponto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sem prompt nenhum, a imagem inteira é segmentada (um gerador

        # automático de máscaras simplificado, não o exaustivo de referência).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifique uma vez, mande vários prompts'
      language: python
      code: |
        from libreyolo import LibreSAM2, SAMPLE_IMAGE

        # A classe específica da família recebe o tamanho sem o prefixo "sam2-".
        model = LibreSAM2("large")

        # O encoder de imagem é a parte cara. set_image() o roda uma vez;
        # toda chamada a predict() depois disso reusa o embedding em cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Instalação

O SAM 2 precisa do extra `sam`, que puxa `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predição

`LibreSAM(...)` (ou o `LibreSAM2(...)` específico da família) é um ponto de
entrada separado de `LibreYOLO(...)`: ele devolve um segmentador guiado por
prompts em vez de um detector, porque aqui um forward pass não significa nada
sem um prompt espacial. Não existe comando de CLI `libreyolo predict` para esta
família; use a API Python. Só a segmentação de imagens é suportada; o tracking
com memória de vídeo do SAM 2 está fora do escopo aqui.

<code-tabs name="predict" />

Um prompt de ponto aceita `[x, y]` para um objeto, `[[x, y], ...]` para vários,
ou arrays numpy; `labels` marca cada ponto com `1` (primeiro plano) ou `0`
(fundo) e, por padrão, todos são primeiro plano. Um prompt de caixa recebe
`[x1, y1, x2, y2]` ou uma lista de caixas, uma máscara por caixa. Omitir os dois
prompts segmenta a imagem inteira mandando uma grade densa de prompts e ficando
com as máscaras confiantes que não se sobrepõem; esse modo de "segmentar tudo" é
simplificado em relação ao gerador automático de máscaras de referência e pode
subsegmentar cenas cheias, então um prompt real de ponto ou de caixa é o caminho
preciso. `conf` filtra pela qualidade de máscara predita (IoU), não por uma
confiança de detecção: passe `0.0` para manter todos os candidatos.
`multimask=True` devolve as três máscaras de ambiguidade todo-versus-parte do
SAM para cada prompt, em vez de apenas a melhor. `device=` move o modelo e, se
houver uma sessão de `set_image()` ativa, o embedding em cache dela. Toda
máscara carrega o id de classe `0`, chamado `"object"`, já que uma máscara
guiada por prompt não tem um conjunto fixo de classes. `train()`, `val()`,
`export()` e `track()` todos levantam `NotImplementedError` nesta família: a
inferência sobre imagens é o que o LibreYOLO suporta aqui. Veja
[predição](/docs/predict) para os tipos de fonte.

## Variantes

Quatro tamanhos com backbone Hiera: tiny, small, base-plus e large, todos na
mesma resolução de entrada. Ainda não há benchmark de acurácia ou de latência
publicado para esta família, então escolher um tamanho troca diretamente peso
do encoder por qualidade de máscara: tiny é o mais rápido de codificar, large o
mais pesado.

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
