---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: segmentação leve guiada por prompts no LibreYOLO'
description: >-
  Use o MobileSAM no LibreYOLO para segmentação guiada por prompts de ponto e de
  caixa com um encoder TinyViT. Instale e faça predições com o checkpoint tiny
  sob Apache-2.0.
lead: >-
  O MobileSAM troca o encoder de imagem ViT-H do SAM por um encoder TinyViT
  destilado, então o mesmo fluxo de trabalho com prompts de ponto e de caixa
  roda em hardware mais leve. O LibreYOLO traz um port nativo dele através de
  uma fábrica LibreSAM dedicada, separada da fábrica de detectores LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - segmentação com prompts
  - segmentação interativa python
  - segmentar objeto com um clique
  - prompt de ponto
  - segmentação leve
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de ponto e de caixa
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # O MobileSAM tem um único tamanho, "tiny", então nenhum outro alias é
        preciso.

        model = LibreSAM("mobilesam")


        # Um prompt de ponto: [x, y] em coordenadas de pixel, label 1 = primeiro
        plano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # um polígono por máscara

        print(result.boxes.xyxy)    # caixa ajustada derivada da máscara


        # Um prompt de caixa em vez de um ponto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sem prompt nenhum, a imagem inteira é segmentada (um gerador
        automático

        # de máscaras simplificado, não o exaustivo de referência).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifique uma vez, use vários prompts'
      language: python
      code: |
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE

        model = LibreMobileSAM()

        # O encoder de imagem é a parte cara. set_image() o executa uma vez;
        # toda chamada a predict() depois disso reutiliza o embedding em cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Instalação

O MobileSAM precisa do extra `sam`: o download de pesos do próprio LibreYOLO
ainda passa pelas ferramentas de snapshot do Hugging Face do `transformers`,
mesmo que a inferência rode em um decoder nativo que não usa `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predição

`LibreSAM(...)` (ou o `LibreMobileSAM(...)` específico da família) é um ponto
de entrada separado do `LibreYOLO(...)`: ele devolve um segmentador guiado por
prompts em vez de um detector, porque aqui um forward pass não significa nada
sem um prompt espacial. Não existe comando de CLI `libreyolo predict` para
essa família; use a API Python.

<code-tabs name="predict" />

Um prompt de ponto aceita `[x, y]` para um objeto, `[[x, y], ...]` para vários,
ou arrays numpy; `labels` marca cada ponto com `1` (primeiro plano) ou `0`
(fundo) e, por padrão, todos são primeiro plano. Um prompt de caixa recebe
`[x1, y1, x2, y2]` ou uma lista de caixas, uma máscara por caixa. Omitir os
dois prompts segmenta a imagem inteira lançando uma grade densa de prompts e
ficando com as máscaras confiantes que não se sobrepõem; esse modo de
"segmentar tudo" é simplificado em relação ao gerador automático de máscaras de
referência e pode subsegmentar cenas cheias, então um prompt real de ponto ou
de caixa é o caminho preciso. `conf` filtra pela qualidade de máscara predita
(IoU), não por uma confiança de detecção: passe `0.0` para manter todos os
candidatos. `multimask=True` devolve as três máscaras de ambiguidade
todo-versus-parte do SAM por prompt, em vez de apenas a melhor. `device=` move
o modelo e, se houver uma sessão `set_image()` ativa, o embedding em cache
dela. Toda máscara carrega o id de classe `0`, com nome `"object"`, já que uma
máscara guiada por prompt não tem um conjunto fixo de classes. `train()`,
`val()`, `export()` e `track()` levantam todos `NotImplementedError` nessa
família: no LibreYOLO, o MobileSAM é apenas de predição. Veja
[predição](/docs/predict) para os tipos de fonte.

## Variantes

Um único tamanho, tiny, com entrada fixa de 1024 px: o MobileSAM traz um único
encoder TinyViT em vez da escada base/large/huge que o SAM-1 oferece.

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
