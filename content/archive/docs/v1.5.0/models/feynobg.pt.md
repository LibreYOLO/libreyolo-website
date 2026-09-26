---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: remoção de fundo no LibreYOLO'
description: >-
  Use o FeyNobg no LibreYOLO para remoção de fundo e alpha matting, uma variante
  aprofundada do BiRefNet feita pela Feyn Inc. Instale, faça predições e valide.
lead: >-
  Um modelo de remoção de fundo da Feyn Inc. que aprofunda a arquitetura do
  BiRefNet e a retreina. O LibreYOLO inclui inferência e validação para a tarefa
  de matte do FeyNobg.
keywords:
  - FeyNobg
  - remover fundo de imagem
  - tirar fundo de foto python
  - segmentação dicotômica de imagens
  - alpha matte
  - image matting
  - recorte com fundo transparente
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Recorte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: o RGB de origem mais o matte como canal alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Um diretório contendo images/ e um diretório de mattes detectado
        # automaticamente (mattes/, matte/, gt/, masks/, mask/ ou alpha/)
        # também funciona no lugar de um YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Instalação

O FeyNobg não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

O checkpoint é baixado da organização LibreYOLO no Hugging Face no primeiro uso
e fica em cache local, como em qualquer outra família, embora ainda não esteja
listado na tabela de Checkpoints desta página.

<code-tabs name="predict" />

Um resultado de matte não carrega boxes; `result.matte` é um array denso
`(H, W)` float32 em `[0, 1]`, com 1 totalmente primeiro plano e 0 totalmente
fundo. Diferente de uma máscara binária, o matte suave preserva o detalhe das
bordas com antialiasing, como cabelo e pelo. `result.cutout()` compõe a imagem
de origem com esse canal alfa em um array RGBA, e `result.save(path)` (ou
`save=True` na chamada de predição) grava direto em um PNG de fundo
transparente. O modelo roda em um canvas nativo fixo de 1024x1024; outra
resolução não é suportada, porque as tabelas de posição relativa do backbone
Swin estão atreladas a ela, e uma divergência as interpola mal em vez de gerar
um erro. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Variantes

Um único tamanho publicado, `l`, um backbone do nível Swin-L. O FeyNobg pega a
arquitetura do BiRefNet e aprofunda seu terceiro estágio Swin de 18 para 24
blocos antes de retreinar, então o port para LibreYOLO reaproveita o forward
path, o pré-processamento e o contrato de saída de logit único do BiRefNet;
predição, validação e o tratamento de checkpoints se comportam igual à família
`birefnet`.

## Validação

`val()` reporta duas métricas sobre uma pasta pareada de imagens e mattes,
ambas em `[0, 1]` e independentes da resolução: MAE, o erro absoluto médio em
relação ao alfa do ground truth (quanto menor, melhor), e S-measure (Fan et
al., ICCV 2017), uma similaridade estrutural que valoriza preservar a forma e
os buracos do sujeito, algo que o MAE por pixel sozinho não capta (quanto
maior, melhor). A validação passa pelo próprio `predict` do modelo, então usa
exatamente o pré-processamento da família.

<code-tabs name="val" />

A validação é somente inferência. A biblioteca `nobg` upstream inclui código de
treinamento sob Apache-2.0; hoje, fazer fine-tuning significa treinar lá e
converter o resultado com o script de conversão do próprio LibreYOLO, não
chamar `train()` nesta família, que gera um erro em vez de rodar um trainer
parcial.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
