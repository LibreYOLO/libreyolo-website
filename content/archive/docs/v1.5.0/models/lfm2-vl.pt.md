---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: detecção de vocabulário aberto no LibreYOLO'
description: >-
  Use o LFM2-VL no LibreYOLO para detecção de objetos de vocabulário aberto no
  dispositivo. Faça predições com qualquer rótulo de texto; treinamento,
  validação e exportação não são suportados.
lead: >-
  O LFM2-VL é um modelo de visão e linguagem compacto, feito para rodar no
  dispositivo, lançado pela Liquid AI. O LibreYOLO o envolve como um detector de
  objetos de vocabulário aberto: qualquer lista de rótulos de texto vira o
  conjunto de classes, sem cabeça fixa e sem precisar de fine-tuning.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - detectar objetos sem treinar
  - VLM python
  - VLM no dispositivo
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Vocabulário aberto: qualquer palavra vale, não uma cabeça fixa.
        # Persiste nas chamadas a predict()/track() até ser definido de novo.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat direto
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # A saída de escape por baixo da conveniência de detecção: perguntas
        # livres, contagem ou qualquer prompt que o wrapper de caixas não cobre.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Instalação

O LFM2-VL precisa do extra `vlm`, que traz junto o `transformers` para o
backbone de template de chat.

```bash
pip install "libreyolo[vlm]"
```

## Predição

`LibreLFM2VL` é uma classe Python, não um checkpoint `.pt`: ela não é
carregada pela fábrica `LibreYOLO()`, e a CLI do `libreyolo` não a resolve. A
fábrica `LibreVLM(...)` (`from libreyolo import LibreVLM`) também alcança esta
família por alias, por exemplo `LibreVLM("lfm2-vl-450m")`; a classe usada
abaixo é o que ela constrói. Os pesos vêm do próprio repositório da Liquid AI
no Hugging Face, e não de um mirror do LibreYOLO; a primeira chamada faz o
download e os deixa em cache localmente, e registra no log um aviso de licença
uma única vez antes disso.

<code-tabs name="predict" />

`result.boxes` traz as detecções parseadas como em qualquer outra família. A
confiança é um placeholder: o LFM2-VL não emite score por caixa, então toda
detecção recebe a mesma confiança constante, e `conf=` apenas descarta as
linhas abaixo dessa constante, sem ordená-las. `iou` descarta caixas quase
duplicadas da mesma classe acima da sobreposição informada, um efeito colateral
de o decoding guloso repetir um objeto; não é uma passada de NMS por classe.
Pule `set_classes()` e o vocabulário assume por padrão os nomes do COCO-80.
Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Variantes

Dois tamanhos: 450m e 1.6b, ambos da versão LFM2.5-VL da Liquid AI, feitos para
deploy no dispositivo. O harness de benchmark do LibreYOLO não mediu esta
família, então não há números de acurácia publicados para compará-los; escolha
um tamanho de acordo com o seu próprio orçamento de computação.

O LibreYOLO expõe esta família apenas para predição. `train()`, `val()` e
`export()` lançam `NotImplementedError`: faça fine-tuning upstream e carregue o
resultado, a validação em dataset fica de fora porque uma confiança de
placeholder tornaria o mAP do COCO enganoso, e a exportação está fora de escopo
para um modelo generativo sem state dict para rastrear.

## Licenciamento

<provenance-box>

A LFM Open License v1.0 permite uso comercial, reprodução e modificação, mas
apenas abaixo de um limiar de US$ 10 milhões de receita anual; uma pessoa
jurídica nesse limiar ou acima dele não é licenciada por este acordo de forma
alguma para uso comercial, e precisa entrar em contato diretamente com a Liquid
AI. Organizações sem fins lucrativos qualificadas estão isentas do limiar para
uso não comercial ou de pesquisa. O LibreYOLO não distribui nenhum código-fonte
da LiquidAI, já que o modelo é carregado através da biblioteca `transformers`,
sob Apache-2.0, e não hospeda nem redistribui os pesos: `LibreLFM2VL` baixa o
tamanho correspondente diretamente do próprio repositório da Liquid AI no
Hugging Face na primeira vez que roda, e registra no log um aviso uma única vez
antes desse download.

</provenance-box>

## Citação

<citation-block />
