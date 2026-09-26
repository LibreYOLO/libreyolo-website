---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: segmentação guiada por prompts e por conceito no LibreYOLO'
description: >-
  Use o SAM 3 no LibreYOLO para segmentação por ponto, por caixa e por conceito
  de texto. Instale e faça predições com o checkpoint large, restrito sob a SAM
  License da Meta.
lead: >-
  O SAM 3 estende o SAM com um prompt de conceito em texto além dos pontos e
  caixas de sempre, então uma frase como "yellow school bus" devolve todas as
  instâncias correspondentes. O LibreYOLO dá suporte ao caminho de imagem dele
  através de uma fábrica LibreSAM dedicada, separada da fábrica de detectores
  LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - segmentação com prompts
  - segmentação por conceito
  - segmentar com texto python
  - prompt de texto
  - prompt de ponto
  - segmentar objeto com um clique
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de ponto e de caixa
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "sam3" é o único tamanho ("large"); aliases: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Um prompt de ponto: [x, y] em coordenadas de pixel, label 1 = primeiro
        plano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # um polígono por máscara

        print(result.boxes.xyxy)    # caixa ajustada derivada da máscara


        # Um prompt de caixa em vez de um ponto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt de texto (conceito)
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Encontra todas as instâncias que casam com a frase, não apenas um
        objeto.

        # text= é mutuamente exclusivo com points, bboxes, labels e masks.

        result = model.predict(SAMPLE_IMAGE, text="a person")

        print(result.names)         # {0: "a person"}

        print(result.boxes.conf)    # o score de detecção PCS por instância
    - label: 'Codifique uma vez, use vários prompts'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # O encoder de imagem é a parte cara. set_image() o executa uma vez;

        # toda chamada a predict() depois disso reutiliza o embedding em

        # cache. Uma chamada com text= recodifica internamente, já que o

        # tracker e o encoder de segmentação por conceito não compartilham
        cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Instalação

O SAM 3 precisa do extra `sam`, que traz junto `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

Os pesos são restritos: acesse
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), aceite a
SAM License da Meta e então rode `hf auth login` (ou defina `HF_TOKEN`) antes do
primeiro download. O LibreYOLO registra um aviso de licença na primeira vez que
baixa essa família.

## Predição

`LibreSAM(...)` (ou o `LibreSAM3(...)` específico da família) é um ponto de
entrada separado do `LibreYOLO(...)`: ele devolve um segmentador guiado por
prompts em vez de um detector, porque aqui um forward pass não significa nada
sem um prompt. Não existe comando de CLI `libreyolo predict` para essa família;
use a API Python. Só há suporte a inferência sobre imagens; os modelos de vídeo
do SAM 3 estão fora do escopo aqui.

<code-tabs name="predict" />

O caminho de ponto e caixa é igual ao do resto da família SAM: um prompt de
ponto aceita `[x, y]` para um objeto ou `[[x, y], ...]` para vários, `labels`
marca cada ponto com `1` (primeiro plano) ou `0` (fundo), e um prompt de caixa
recebe `[x1, y1, x2, y2]` ou uma lista de caixas. Nesse caminho, `conf` filtra
pela qualidade de máscara predita (IoU), não por uma confiança de detecção.

O caminho do `text=` é o que o SAM 3 acrescenta: uma string de conceito devolve
todas as instâncias correspondentes na imagem através da Promptable Concept
Segmentation, e não pode ser combinada com pontos, caixas, labels ou máscaras.
Ali `conf` é o score de detecção PCS em vez do IoU da máscara; deixá-lo no valor
padrão aplica o limiar de 0.3 do próprio modelo, e `conf=0.0` mantém todos os
candidatos. O `names` devolvido associa o id de classe `0` à string de conceito
solicitada, já que, fora isso, uma máscara guiada por prompt não tem um conjunto
fixo de classes. `device=` move o modelo e, se houver uma sessão `set_image()`
ativa, o embedding em cache dela. `train()`, `val()`, `export()` e `track()`
levantam todos `NotImplementedError` nessa família: no LibreYOLO, o SAM 3 é
apenas de predição, e o rastreamento em vídeo está fora do escopo. Veja
[predição](/docs/predict) para os tipos de fonte.

## Variantes

Um único tamanho, large, com entrada fixa de 1008 px. O SAM 3.1 não tem
suporte: sua implementação carrega uma licença própria que não pode ser
incorporada a este repositório MIT, e a versão do Transformers da qual o
LibreYOLO depende ainda não carrega o formato de checkpoint dele.

## Licenciamento

<provenance-box>

O LibreYOLO não hospeda uma cópia própria dos pesos do SAM 3 e não os
redistribui. `LibreSAM("sam3")` baixa diretamente do repositório restrito
`facebook/sam3` da Meta no Hugging Face, que exige aceitar a SAM License da Meta
e se autenticar antes do primeiro download.

</provenance-box>

## Citação

<citation-block />
