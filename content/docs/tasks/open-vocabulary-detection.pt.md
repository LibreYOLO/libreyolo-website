---
title: Detecção de vocabulário aberto
seo_title: Detecção de vocabulário aberto no LibreYOLO
description: >-
  Detecte objetos a partir de um vocabulário de texto no LibreYOLO. Carregue
  Grounding DINO, OWLv2, OMDet-Turbo ou OV-DEIM pelo LibreOpenVocab e defina as
  classes em runtime.
lead: >-
  A detecção de vocabulário aberto substitui a lista fixa de classes de um
  checkpoint por palavras que você escolhe na hora da chamada. No LibreYOLO ela
  não é uma tarefa separada: é a tarefa detect atendida por um tier de modelos
  separado, carregado pela factory LibreOpenVocab em vez do LibreYOLO.
keywords:
  - detecção de vocabulário aberto
  - detecção de objetos zero-shot
  - open-set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - detectar objetos por texto
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Trocar o vocabulário
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("owlv2-b16")


        # set_classes é persistente: vale até a próxima chamada dele.

        # Os rótulos precisam ser únicos depois de passar para minúsculas e
        perder os artigos.

        model.set_classes(["a red backpack", "traffic cone"])

        result = model.predict(SAMPLE_IMAGE)


        model.set_classes(["bicycle wheel"])

        result = model.predict(SAMPLE_IMAGE)
    - label: Limiar de texto do Grounding DINO
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtra pelo score da caixa, e text_threshold pelo score de token
        # da frase decodificada. Ambos assumem 0.25 quando não são definidos.
        # Só o Grounding DINO aceita text_threshold; os outros levantam erro.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definição

A detecção de vocabulário aberto retorna `Results` de detecção comuns: caixas,
confianças e índices de classe, com `result.names` mapeando esses índices de
volta para as strings que você pediu. O que muda é de onde vem a lista de
classes. Um detector convencional é treinado contra um conjunto fixo de
categorias e nunca consegue emitir uma categoria fora dele. Esses modelos
recebem o vocabulário como texto no momento da inferência, então
`set_classes(["forklift", "safety cone"])` basta para que essas sejam as
classes.

O LibreYOLO não tem uma chave de tarefa `open-vocabulary`. Esses modelos
declaram `SUPPORTED_TASKS = ("detect",)` como qualquer outro detector. O que os
separa é o caminho de carregamento: são snapshots da Hugging Face em vez de
checkpoints de state-dict do LibreYOLO, então ficam de fora da factory
`LibreYOLO()` e são construídos por `LibreOpenVocab()`. Essa factory é irmã de
`LibreSAM()` e `LibreVLM()`, não uma substituta de `LibreYOLO()`.

Os scores são scores de detecção reais, não uma legenda gerada e interpretada
depois. Cada família pontua regiões da imagem contra o embedding de texto de
cada prompt.

## Modelos

Quatro famílias formam o tier, todas elas somente de predição. Carregue qualquer
uma delas por alias através do `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino), da IDEA Research, nos tamanhos
`t` e `b`. É o padrão do tier e a única família que aceita `text_threshold`, um
segundo corte sobre o score de token da frase decodificada.

[OWLv2](/docs/models/owlv2), do Google Research, nos tamanhos `b16` e `l14`.
Ele pontua regiões da imagem contra embeddings de texto de um codificador no
estilo CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo), do Om AI Lab, em um único tamanho `t`.
Ele desacopla os embeddings de classe de um prompt de tarefa em linguagem
natural, e é a única família aqui que suprime caixas sobrepostas dentro do seu
próprio pós-processamento, então `iou=` é respeitado.

[OV-DEIM](/docs/models/ov-deim), nos tamanhos `s`, `m` e `l`, um detector no
estilo DETR que casa as queries do decodificador com embeddings de texto de uma
torre de texto MobileCLIP embutida. É correspondência um-para-um com seleção
top-K, então nenhum NMS roda em lugar nenhum.

Os pesos do OV-DEIM são o caso restrito deste tier. Os pesos do detector são CC
BY-NC 4.0, não comerciais. A torre de texto embutida está sob a licença Machine
Learning Research Model da Apple, apenas para uso em pesquisa. O checkpoint `l`
acrescenta um fine-tuning de backbone DINOv3-S sob a DINOv3 License da Meta. Os
três textos de licença acompanham o repositório de pesos, e a biblioteca
registra no log o mesmo resumo quando resolve os pesos, antes de o modelo ser
construído. Leia [OV-DEIM](/docs/models/ov-deim) antes de fazer deploy dele.

O tier precisa de um extra:

```bash
pip install "libreyolo[openvocab]"
```

Isso cobre `transformers` e `timm` para as três famílias encapsuladas, e os
pacotes `huggingface_hub`, `safetensors`, `regex` e `ftfy` de que o OV-DEIM
precisa por ser um port nativo.

Um segundo tier também aceita um vocabulário de texto: `LibreVLM()` carrega
modelos generativos de visão e linguagem, como
[Qwen3-VL](/docs/models/qwen3-vl) e [Florence-2](/docs/models/florence-2), e
transforma a saída deles nos mesmos `Results`. Ele compartilha a mesma
superfície `set_classes()`. A diferença está no que produz as caixas: as
famílias desta página são detectores discriminativos que emitem scores
diretamente, enquanto o tier de VLMs as gera.

## Predição

<code-tabs name="predict" />

`set_classes()` recebe uma lista não vazia de strings de rótulo e vale até ser
chamada de novo. Os rótulos precisam ser únicos depois de passar para minúsculas
e perder os artigos iniciais, então `"a bus"` e `"bus"` não podem coexistir em
um mesmo vocabulário. Frases de várias palavras são rótulos como quaisquer
outros, e cada família transforma a lista na sua própria entrada de texto antes
de tokenizar, então `"traffic cone"` é uma query diferente de `"cone"`.

Três argumentos de predição se comportam de forma diferente aqui em relação a um
detector nativo. `imgsz=` é rejeitado, porque o processador é quem cuida do
redimensionamento nessas famílias. `augment=True` é rejeitado, já que o data
augmentation em tempo de teste está fora do escopo do tier. `iou=` vale apenas
para a família cujo processador roda a própria supressão; onde nada é suprimido,
passá-lo emite um aviso e é ignorado.

Quando não é definido, `conf` assume o padrão da própria família carregada em
vez do 0.25 usual de `predict()`, e esse padrão não é o mesmo em todo o tier.
Defina o valor explicitamente ao comparar duas famílias na mesma imagem.

`track()` levanta erro em todo o tier. Rode `predict()` quadro a quadro no
lugar. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Treinamento

Nenhuma família deste tier treina dentro do LibreYOLO. `train()` levanta erro:
faça fine-tuning upstream e carregue os pesos resultantes. O vocabulário passado
para `set_classes()` é a única configuração que muda o que um modelo carregado
detecta.

## Validação

Não existe validador para este tier, e `val()` levanta erro. A validação de
vocabulário aberto precisa de um validador dedicado, porque o validador de
detecção padrão alimenta o modelo diretamente com tensores de imagem, enquanto
essas famílias exigem entradas condicionadas por texto construídas junto com
eles.

## Exportação

A exportação está fora do escopo do tier e `export()` levanta erro. Esses
modelos rodam por `predict()` no PyTorch.
