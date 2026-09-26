---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): prever máscaras no LibreYOLO'
description: >-
  Use o SAM no LibreYOLO para segmentação guiada por prompt de ponto e de caixa.
  Instale e faça predições com os checkpoints base, large e huge sob Apache-2.0.
lead: >-
  O SAM (Segment Anything) transforma um clique de ponto ou de caixa na máscara
  de um objeto. O LibreYOLO o carrega por meio de uma factory LibreSAM dedicada,
  separada da factory de detectores LibreYOLO(), porque um modelo guiado por
  prompt precisa de outro formato de chamada.
keywords:
  - SAM
  - Segment Anything
  - segmentação com prompt
  - segmentação interativa python
  - segmentar objeto com um clique
  - prompt de ponto
  - prompt de caixa
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de ponto e de caixa
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" baixa automaticamente facebook/sam-vit-base no primeiro uso.

        # Outros tamanhos: "large", "huge" (também "b"/"l"/"h").

        model = LibreSAM("base")


        # Um prompt de ponto: [x, y] em coordenadas de pixel, label 1 = primeiro
        plano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # um polígono por máscara

        print(result.boxes.xyxy)    # caixa justa derivada da máscara


        # Um prompt de caixa em vez de um ponto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sem nenhum prompt, a imagem inteira é segmentada (um gerador

        # automático de máscaras simplificado, não o exaustivo de referência).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifique uma vez, use vários prompts'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # O encoder de imagem é a parte cara. set_image() o executa uma vez;
        # toda chamada a predict() depois disso reutiliza o embedding em cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f8904d241ef8a929
---

## Instalação

O SAM precisa do extra `sam`, que instala junto `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predição

`LibreSAM(...)` é um ponto de entrada separado de `LibreYOLO(...)`: ele devolve
um segmentador guiado por prompt em vez de um detector, porque aqui um forward
pass não significa nada sem um prompt espacial. Não existe comando de CLI
`libreyolo predict` para esta família; use a API do Python.

<code-tabs name="predict" />

Um prompt de ponto aceita `[x, y]` para um objeto, `[[x, y], ...]` para vários,
ou arrays do numpy; `labels` marca cada ponto com `1` (primeiro plano) ou `0`
(fundo) e, por padrão, todos são primeiro plano. Um prompt de caixa recebe
`[x1, y1, x2, y2]` ou uma lista de caixas, uma máscara por caixa. Omitir os dois
prompts segmenta a imagem inteira lançando uma grade densa de prompts e ficando
com as máscaras confiantes que não se sobrepõem; esse modo "segmentar tudo" é
simplificado em relação ao gerador automático de máscaras de referência e pode
subsegmentar cenas cheias, então um prompt real de ponto ou de caixa é o caminho
preciso. `conf` filtra pela qualidade de máscara prevista (IoU), não por uma
confiança de detecção: passe `0.0` para manter todos os candidatos.
`multimask=True` devolve as três máscaras de ambiguidade todo-versus-parte do
SAM para cada prompt, em vez de apenas a melhor. `device=` move o modelo e, se
houver uma sessão de `set_image()` ativa, o embedding em cache dela. Toda
máscara carrega o id de classe `0`, com nome `"object"`, já que uma máscara
guiada por prompt não tem um conjunto fixo de classes. `train()`, `val()`,
`export()` e `track()` levantam `NotImplementedError` nesta família: o SAM é
somente predição no LibreYOLO, e o rastreamento em vídeo está fora do escopo.
Veja [predição](/docs/predict) para os tipos de fonte.

## Variantes

Três tamanhos de encoder de imagem ViT: base, large e huge, todos com entrada
fixa de 1024 px. Ainda não há nenhum benchmark de acurácia ou de latência
publicado para esta família, então escolher um tamanho troca diretamente peso do
encoder por qualidade de máscara: base é o mais rápido de codificar, huge o mais
pesado.

## Licenciamento

<provenance-box>

O LibreYOLO não hospeda sua própria cópia dos pesos do SAM-1.
`LibreSAM("base")`, `"large"` e `"huge"` baixam direto dos repositórios
`facebook/sam-vit-base`, `facebook/sam-vit-large` e `facebook/sam-vit-huge` da
própria Meta no Hugging Face, cada um marcado como Apache-2.0 lá, de forma
independente do LibreYOLO.

</provenance-box>

## Citação

<citation-block />
