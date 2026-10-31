---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: detecção de vocabulário aberto no LibreYOLO'
description: >-
  Use o InternVL3 no LibreYOLO para detecção de objetos de vocabulário aberto.
  Faça predições com qualquer rótulo de texto; treinamento, validação e
  exportação não têm suporte.
lead: >-
  O InternVL3 é um modelo de linguagem grande multimodal nativo lançado pela
  OpenGVLab que aprende visão e linguagem de forma conjunta em uma única etapa
  de pré-treinamento. O LibreYOLO o envolve como detector de objetos de
  vocabulário aberto: qualquer lista de rótulos de texto vira o conjunto de
  classes, sem cabeça fixa e sem precisar de fine-tuning.
keywords:
  - InternVL3
  - InternVL
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - detectar objetos com texto
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Vocabulário aberto: vale qualquer palavra, não uma cabeça de
        # classes fixa. Persiste em cada predict()/track() posterior até
        # ser definido de novo.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat direto
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # A via de escape sob a comodidade da detecção: perguntas livres,
        # contagens ou qualquer prompt que o wrapper de boxes não cubra.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Instalação

O InternVL3 precisa do extra `vlm`, que traz junto o `transformers` para o
backbone de templates de chat.

```bash
pip install "libreyolo[vlm]"
```

## Predição

`LibreInternVL3` é uma classe Python, não um checkpoint `.pt`: ele não é
carregado pela factory `LibreYOLO()`, e a CLI `libreyolo` não o resolve. A
factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) também alcança esta
família por alias, por exemplo `LibreVLM("internvl3-2b")`; a classe usada
abaixo é a que ela constrói. Os pesos vêm dos próprios repositórios `-hf` da
OpenGVLab no Hugging Face, não de um mirror do LibreYOLO; a primeira chamada
baixa e guarda em cache localmente, e antes disso registra um aviso de licença
único sobre os pesos Qwen, que são de acesso restrito.

<code-tabs name="predict" />

`result.boxes` carrega as detecções já parseadas como em qualquer outra
família. A confiança é um marcador de posição: o InternVL3 não emite nenhuma
pontuação por box, então toda detecção recebe a mesma confiança constante, e
`conf=` só descarta as linhas abaixo dessa constante, ele não as ordena. `iou`
descarta boxes quase duplicados da mesma classe acima da sobreposição indicada,
um efeito colateral de a decodificação gulosa repetir um objeto; não é uma
passada de NMS por classe. Pule `set_classes()` e o vocabulário cai nos nomes
do COCO-80 por padrão. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

Três tamanhos: 1b, 2b e 8b, todos checkpoints `-hf` nativos da OpenGVLab (um
backbone de LLM Qwen, não a arquitetura de torre dupla que o paper original do
InternVL descreve). O harness de benchmark do LibreYOLO não mediu esta família,
então não há números de acurácia publicados para compará-los; escolha um
tamanho de acordo com o seu próprio orçamento de computação.

O LibreYOLO expõe esta família apenas para predição. `train()`, `val()` e
`export()` levantam todos `NotImplementedError`: faça o fine-tuning upstream e
carregue o resultado, a validação sobre dataset é ignorada porque uma confiança
de marcador de posição tornaria o mAP do COCO enganoso, e a exportação está
fora de escopo para um modelo generativo sem state dict para traçar.

## Licenciamento

<provenance-box>

O código do próprio InternVL3 é MIT, permissivo e utilizável em produtos
comerciais e de código fechado. Os checkpoints `-hf` que esta família carrega
trazem um backbone de LLM Qwen e são licenciados à parte, sob a Qwen License da
Alibaba Cloud: livres para usar, modificar e redistribuir com a exigência de
atribuição por meio de um "Built with Qwen" ou "Improved using Qwen", e com um
teto de 100 milhões de usuários ativos mensais no uso comercial, acima do qual
é necessária a autorização da própria Alibaba. O LibreYOLO não hospeda nem
redistribui esses pesos: `LibreInternVL3` baixa o tamanho correspondente
diretamente de `OpenGVLab/InternVL3-<size>-hf` no Hugging Face na primeira vez
que roda, e registra um aviso único sobre a Qwen License antes desse download.

</provenance-box>

## Citação

<citation-block />
