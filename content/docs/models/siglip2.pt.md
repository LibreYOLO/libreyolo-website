---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 no LibreYOLO: classificação e embeddings zero-shot'
description: >-
  Use o SigLIP2 no LibreYOLO para classificação de imagens zero-shot e embedding
  de imagem e texto, com pontuação multi-rótulo por sigmoide. Sem treinamento.
lead: >-
  O SigLIP2 é um modelo de torre dupla que pontua uma imagem contra prompts de
  texto com uma sigmoide independente por classe, em vez de um softmax
  compartilhado sobre um conjunto fixo de rótulos. O LibreYOLO oferece suporte a
  ele para classificação zero-shot e embedding de imagem e texto, sem nenhuma
  etapa de treinamento.
keywords:
  - SigLIP2
  - SigLIP 2
  - classificação zero-shot
  - classificar imagens sem treinar
  - embedding de imagem python
  - embedding de texto
  - vocabulário aberto
  - modelo multilíngue
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Sem uma chamada a set_classes(), o predict da CLI usa os 1.000

        # nomes de classe do ImageNet que o modelo carrega por padrão.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pontuação sigmoide multi-rótulo
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # Probabilidades independentes por classe: mais de uma, ou nenhuma,
        # pode pontuar alto ao mesmo tempo. Já o softmax (o padrão) normaliza
        # tudo em uma distribuição de rótulo único, igual ao LibreCLIP.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embedding de imagem e texto
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Ambos são normalizados por L2, então um simples produto escalar é a
        similaridade de cosseno.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data é uma raiz ImageFolder com um split train/; os nomes das pastas
        # viram os prompts de classe zero-shot desta execução.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Os rótulos atuais de set_classes() e a resolução de entrada ficam
        # fixados no grafo. Exporte de novo depois de mudar qualquer um dos
        # dois. multi_label precisa ser False (o padrão) na hora de exportar.
    - label: CLI
      language: bash
      code: |
        # Aqui não há chamada a set_classes(), então isso fixa as 1.000
        # classes padrão do ImageNet com as quais o modelo carrega.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Exportação de embeddings
      language: python
      code: >
        from libreyolo import LibreYOLO


        # task="embed" traça apenas a torre de imagem; nenhuma classe é
        necessária.

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Instalação

O SigLIP2 precisa do seu próprio extra, que instala o pacote SentencePiece usado pelo seu tokenizador multilíngue.

```bash
pip install "libreyolo[siglip2]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache localmente.

<code-tabs name="predict" />

`set_classes()` é a primitiva que transforma isso em um classificador de vocabulário aberto: ela insere cada rótulo em todos os templates de prompt, codifica e faz a média dos resultados, e guarda em cache a matriz `[K, D]` resultante como cabeça do classificador, de modo que ela não é recalculada a cada imagem. Basta chamá-la de novo para mudar as classes a qualquer momento. Sem nenhuma chamada, o LibreSigLIP2 carrega já com os 1.000 nomes de classe do ImageNet-1k definidos.

O SigLIP pontua cada classe de forma independente: `logit = scale * (image . text) + bias`. Por padrão, esse conjunto de logits ainda passa por um softmax, o que dá uma distribuição de rótulo único que corresponde ao comportamento de `top1`/`top5` do LibreCLIP. Passar `multi_label=True` para `set_classes()` (ou na construção) troca isso por probabilidades sigmoides independentes, de modo que mais de uma classe, ou nenhuma, pode pontuar alto na mesma imagem. O tokenizador é um modelo SentencePiece multilíngue (vocabulário do Gemma), então nomes de classe em idiomas diferentes do inglês funcionam da mesma forma.

Com `task="embed"`, a predição devolve um vetor de imagem normalizado por L2 para cada entrada em vez de probabilidades de classe, e `embed_text()` devolve linhas de texto normalizadas no mesmo espaço vetorial, então um simples produto escalar entre eles é a similaridade de cosseno. `iou` não tem efeito em nenhuma das duas tarefas; não existe etapa de NMS. Veja [predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Validação

`val()` lê os nomes das pastas de classe sob o split `train/` de um ImageFolder, chama `set_classes()` com eles e então mede a acurácia zero-shot top-1 e top-5 com pontuação por softmax. A acurácia depende de como os nomes das classes funcionam como prompts, não de qualquer atualização de pesos, já que não há nada para treinar. A validação cobre apenas `task="classify"`; `task="embed"` não tem validador de dataset.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação fixa o estado atual do modelo em um grafo estático. Para `task="classify"`, os últimos rótulos definidos por `set_classes()` e a resolução no momento da exportação ficam fixados em uma camada linear final com o `scale` e o `bias` aprendidos, de modo que o grafo exportado é um classificador de imagens `[B, K]` comum, sem torre de texto e sem tokenizador; exporte de novo depois de mudar as classes ou o tamanho. A exportação no modo `multi_label=True` não está implementada; volte o valor para `False` antes. A exportação com `task="embed"` traça apenas a torre de imagem. Ambas precisam do opset 14 do ONNX ou superior, que o exportador define por padrão.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família. Ambos são convertidos a partir dos checkpoints Apache-2.0 `siglip2-base-patch16-256` e `siglip2-so400m-patch14-384` do Google, não de nenhum treinamento em COCO.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
