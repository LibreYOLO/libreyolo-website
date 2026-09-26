---
title: CLIP
families:
  - clip
seo_title: 'CLIP no LibreYOLO: classificação e embeddings zero-shot'
description: >-
  Use CLIP no LibreYOLO para classificação de imagens zero-shot e embeddings de
  imagem e texto. Sem treinamento: set_classes() define o conjunto de rótulos em
  tempo de execução.
lead: >-
  CLIP é um modelo de torre dupla que pontua uma imagem contra prompts de texto
  em vez de um conjunto fixo de rótulos. O LibreYOLO oferece suporte a ele para
  classificação zero-shot e embeddings de imagem e texto, sem nenhuma etapa de
  treinamento.
keywords:
  - CLIP
  - OpenCLIP
  - classificação zero-shot
  - classificar imagens sem treinar
  - embedding de imagem python
  - embedding de texto
  - busca por similaridade imagem texto
  - vocabulário aberto
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Sem uma chamada a set_classes(), o predict da CLI usa os 1.000

        # nomes de classe do ImageNet que o modelo carrega por padrão.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embedding de imagem e texto
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

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

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data é uma raiz ImageFolder com um split train/; os nomes das pastas
        # viram os prompts de classe zero-shot desta execução.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Os rótulos atuais de set_classes() e a resolução de entrada ficam

        # fixados no grafo. Exporte de novo depois de mudar qualquer um dos
        dois.
    - label: CLI
      language: bash
      code: |
        # Aqui não há chamada a set_classes(), então isso fixa as 1.000
        # classes padrão do ImageNet com as quais o modelo carrega.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Exportação de embeddings
      language: python
      code: >
        from libreyolo import LibreYOLO


        # task="embed" traça apenas a torre de imagem; nenhuma classe é
        necessária.

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Instalação

O CLIP precisa do seu próprio extra, que instala os pacotes usados pelo tokenizador BPE embutido para reproduzir exatamente os mesmos ids de token.

```bash
pip install "libreyolo[clip]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache localmente.

<code-tabs name="predict" />

`set_classes()` é a primitiva que transforma isso em um classificador de vocabulário aberto: ela insere cada rótulo em todos os templates de prompt, codifica e faz a média dos resultados, e guarda em cache a matriz `[K, D]` resultante como cabeça do classificador, de modo que ela não é recalculada a cada imagem. Basta chamá-la de novo para mudar as classes a qualquer momento. Sem nenhuma chamada, o LibreCLIP carrega já com os 1.000 nomes de classe do ImageNet-1k definidos.

Com `task="embed"`, a predição devolve um vetor de imagem normalizado por L2 para cada entrada em vez de probabilidades de classe, e `embed_text()` devolve linhas de texto normalizadas no mesmo espaço vetorial, então um simples produto escalar entre eles é a similaridade de cosseno. `iou` não tem efeito em nenhuma das duas tarefas; não existe etapa de NMS. Veja [predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Validação

`val()` lê os nomes das pastas de classe sob o split `train/` de um ImageFolder, chama `set_classes()` com eles e então mede a acurácia zero-shot top-1 e top-5. A acurácia depende de como os nomes das classes funcionam como prompts, não de qualquer atualização de pesos, já que não há nada para treinar. A validação cobre apenas `task="classify"`; `task="embed"` não tem validador de dataset.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação fixa o estado atual do modelo em um grafo estático. Para `task="classify"`, os últimos rótulos definidos por `set_classes()` e a resolução no momento da exportação ficam fixados em uma camada linear final, de modo que o grafo ONNX ou TensorRT exportado é um classificador de imagens `[B, K]` comum, sem torre de texto e sem tokenizador; exporte de novo depois de mudar as classes ou o tamanho. A exportação com `task="embed"` traça apenas a torre de imagem. Ambas precisam do opset 14 do ONNX ou superior, que o exportador define por padrão.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família. Ambos são convertidos a partir dos checkpoints do OpenCLIP treinados com LAION-2B (`ViT-B-32` e `ViT-B-16`), não de nenhum treinamento em COCO.

<checkpoint-table />

Os dados de treinamento do LAION-2B têm um histórico documentado de conteúdo CSAM (Stanford Internet Observatory, dezembro de 2023). Desde então a LAION publicou o Re-LAION, uma reedição limpa; se você for re-hospedar esses pesos por conta própria, prefira os checkpoints derivados do Re-LAION quando estiverem disponíveis.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
