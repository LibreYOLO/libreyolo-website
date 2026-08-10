---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: remoção de fundo e matting no LibreYOLO'
description: >-
  Use o BiRefNet no LibreYOLO para remoção de fundo e segmentação dicotômica de
  imagens. Instale, faça predições, valide e exporte o checkpoint geral.
lead: >-
  Uma rede de referência bilateral que prediz um alpha matte suave separando o
  sujeito do fundo. O LibreYOLO inclui inferência e validação para a tarefa de
  matte do BiRefNet.
keywords:
  - BiRefNet
  - remover fundo de imagem
  - tirar fundo de foto python
  - segmentação dicotômica de imagens
  - alpha matte
  - image matting
  - recorte com fundo transparente
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Recorte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: o RGB de origem mais o matte como canal alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Um diretório contendo images/ e um diretório de mattes detectado
        # automaticamente (mattes/, matte/, gt/, masks/, mask/ ou alpha/)
        # também funciona no lugar de um YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Instalação

O BiRefNet não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

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

Um único checkpoint publicado, `l`, o modelo BiRefNet-general do nível Swin-L
e o padrão de qualidade no projeto original. O código da família também
suporta um nível lite Swin-T, `t`, mas ainda não há nenhuma conversão dele
para LibreYOLO publicada.

## Validação

`val()` reporta duas métricas sobre uma pasta pareada de imagens e mattes,
ambas em `[0, 1]` e independentes da resolução: MAE, o erro absoluto médio em
relação ao alfa do ground truth (quanto menor, melhor), e S-measure (Fan et
al., ICCV 2017), uma similaridade estrutural que valoriza preservar a forma e
os buracos do sujeito, algo que o MAE por pixel sozinho não capta (quanto
maior, melhor). A validação passa pelo próprio `predict` do modelo, então usa
exatamente o pré-processamento da família.

<code-tabs name="val" />

A validação é somente inferência; o fine-tuning é um desdobramento documentado
e não um recurso já incluído (veja Predição para a restrição exata de
resolução que qualquer trainer futuro herdaria).

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` se comporta como um checkpoint e devolve o
mesmo `Results`. TorchScript é o caminho validado; a conversão para ONNX roda,
mas não passou pelo mesmo nível de paridade. [Exportação](/docs/export) lista
os argumentos que todo formato aceita e os extras que alguns poucos adicionam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
