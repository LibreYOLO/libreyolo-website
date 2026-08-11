---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: detecção de vocabulário aberto e localização por pontos'
description: >-
  Use o LocateAnything no LibreYOLO para detecção de vocabulário aberto e
  localização por pontos. Faça predições com qualquer rótulo de texto;
  treinamento, validação e exportação não têm suporte.
lead: >-
  O LocateAnything é um modelo de grounding de visão e linguagem lançado pela
  NVIDIA que decodifica bounding boxes e pontos em paralelo, em vez de ir um
  token de coordenada por vez. O LibreYOLO o envolve como detector e localizador
  por pontos de vocabulário aberto: qualquer lista de rótulos de texto vira o
  conjunto de classes, sem cabeça fixa e sem precisar de fine-tuning.
keywords:
  - LocateAnything
  - NVIDIA
  - modelo de visão e linguagem
  - detecção de vocabulário aberto
  - detectar objetos com texto
  - detecção de pontos
  - VLM
  - grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Vocabulário aberto: vale qualquer palavra, não uma cabeça de
        # classes fixa. Persiste em cada predict()/track() posterior até
        # ser definido de novo.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt de ponto
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" devolve um ponto por objeto encontrado, em vez de
        # uma caixa. Troque de tarefa em um modelo já carregado com
        # model.set_task("point").
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Chat direto
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # A via de escape sob a comodidade da detecção: perguntas livres,
        # contagens ou qualquer prompt que o wrapper de boxes não cubra.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Instalação

O LocateAnything precisa do extra `vlm`, que traz junto o `transformers` mais
os pacotes `decord`, `lmdb` e `peft` que o código remoto dele no Hugging Face
importa no carregamento.

```bash
pip install "libreyolo[vlm]"
```

## Predição

`LibreLocateAnything` é uma classe Python, não um checkpoint `.pt`: ele não é
carregado pela factory `LibreYOLO()`, e a CLI `libreyolo` não o resolve. A
factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) também alcança esta
família por alias, por exemplo `LibreVLM("locate-anything")`; a classe usada
abaixo é a que ela constrói. Carregá-lo baixa e executa o código remoto do
próprio modelo da NVIDIA no Hugging Face, então o LibreYOLO fixa o download em
uma revisão de commit fixa em vez do branch mutável `main`, e registra um aviso
de licença único antes do primeiro download.

<code-tabs name="predict" />

`result.boxes` (tarefa `detect`) e `result.points` (tarefa `point`) carregam a
saída já parseada como em qualquer outra família. A confiança é um marcador de
posição: o LocateAnything não emite nenhuma pontuação por box, então toda
detecção recebe a mesma confiança constante, e `conf=` só descarta as linhas
abaixo dessa constante, ele não as ordena. Pule `set_classes()` e o vocabulário
cai nos nomes do COCO-80 por padrão. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Um único tamanho publicado, 3b. Duas tarefas compartilham os mesmos pesos:
`detect` (a padrão) devolve boxes, e `task="point"` devolve no lugar um único
ponto por objeto encontrado, em `result.points`; alterne entre elas em um
modelo já carregado com `model.set_task("point")`. O harness de benchmark do
LibreYOLO não mediu esta família, então não há números de acurácia publicados
para comparação.

O LibreYOLO expõe esta família apenas para predição. `train()`, `val()` e
`export()` levantam todos `NotImplementedError`: faça o fine-tuning upstream e
carregue o resultado, a validação sobre dataset é ignorada porque uma confiança
de marcador de posição tornaria o mAP do COCO enganoso, e a exportação está
fora de escopo para um modelo generativo sem state dict para traçar.

## Licenciamento

<provenance-box>

A NVIDIA License permite uso, reprodução e modificação, mas restringe o modelo
e qualquer derivado a uso não comercial, de pesquisa ou de avaliação apenas,
para qualquer um que não seja a NVIDIA e suas afiliadas: não há limite de
receita nem exceção paga. O LocateAnything-3B também combina outros dois
componentes licenciados: um backbone de linguagem Qwen2.5-3B-Instruct sob a
Qwen Research License, e um codificador de visão MoonViT-SO-400M sob MIT. O
LibreYOLO não hospeda, espelha nem redistribui nada disso:
`LibreLocateAnything` baixa os pesos e o código remoto necessário diretamente
de `nvidia/LocateAnything-3B` no Hugging Face, fixado em um único commit, na
primeira vez que roda.

</provenance-box>

## Citação

<citation-block />
