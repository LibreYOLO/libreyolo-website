---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 no LibreYOLO: segmentação semântica, classificação e embeddings'
description: >-
  Use o DINOv2 no LibreYOLO para segmentação semântica, classificação e
  embedding de imagem inteira sobre o backbone DINOv2-with-Registers. Apache-2.0
  do começo ao fim.
lead: >-
  O DINOv2 é um vision transformer auto-supervisionado treinado pela Meta AI
  para produzir features de imagem de uso geral sem rótulos. O LibreYOLO
  empacota seu backbone DINOv2-with-Registers para três tarefas: segmentação
  semântica, classificação e embedding de imagem inteira.
keywords:
  - DINOv2
  - DINOv2 with registers
  - aprendizado auto-supervisionado
  - vision transformer
  - segmentação semântica python
  - embedding de imagem
  - extração de características de imagem
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Semântica
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Não existe checkpoint hospedado pelo LibreYOLO para esta família:
        # isto baixa o backbone Apache-2.0 DINOv2-with-Registers-small da org
        # da Meta no Hugging Face. A cabeça densa começa com inicialização
        # aleatória até você treiná-la (veja Treinamento abaixo).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Classificação
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= é a contagem de classes do seu dataset; a cabeça linear
        # começa com inicialização aleatória até você treiná-la.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Ignora todas as cabeças de tarefa: o backbone sozinho já basta,
        # então isso não precisa de fine-tuning para ser útil.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), normalizado por L2
    - label: Embedding de um batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Wrapper de conveniência: roda predict() e empilha cada linha em um
        # único tensor (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semântica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Classificação
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semântica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Classificação
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semântica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Classificação
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results. A
        # exportação nomeia o arquivo pela tarefa, aqui LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Instalação

O LibreDINOv2 só é registrado quando `transformers` está instalado, a mesma
dependência opcional que o RF-DETR precisa para seu backbone DINOv2, então ele
precisa do mesmo extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Predição

O LibreYOLO não publica um checkpoint LibreDINOv2. Construa o wrapper
diretamente em vez de carregar um arquivo: `model_path=None` (o padrão) baixa
o backbone Apache-2.0 `facebook/dinov2-with-registers-small` da Meta no
Hugging Face no primeiro uso. `task=` seleciona o que roda em cima dele.

<code-tabs name="predict" />

`task="semantic"` e `task="classify"` adicionam uma cabeça densa ou linear em
cima do backbone; essa cabeça é inicializada aleatoriamente e só é útil depois
que você a treina (veja [Treinamento](#train)). `task="embed"` pula todas as
cabeças e devolve o token CLS final normalizado do backbone como uma única
linha por imagem inteira em `result.embeddings`, então não precisa de
treinamento nenhum. `result.boxes` é sempre `None`: nenhuma das três tarefas
produz detecções por instância. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

`size` seleciona a largura do projetor no estilo RF-DETR sobreposto ao
backbone, não o backbone em si: todos os tamanhos compartilham o mesmo encoder
DINOv2-S (small). A segmentação semântica roda na grade quadrada nativa de
patches do DINOv2; a classificação e o embedding rodam na resolução de
classificação menor usada para treinar o linear probe.

## Treinamento

`task="semantic"` e `task="classify"` treinam; `task="embed"` não tem cabeça
dependente de classes para ajustar e levanta `NotImplementedError` se você
chamar `train()` nele.

<code-tabs name="train" />

Os argumentos nomeados principais aqui são `batch_size` e `lr`, não `batch` e
`lr0` usados pela maioria das outras famílias; `batch` e `lr0` continuam sendo
aceitos e mapeados para eles, mas passar os dois levanta um erro de conflito.
`output_dir=` (padrão `"runs/train"`) substitui `project=`/`name=` como forma
principal de posicionar uma execução, embora passar `project=`/`name=`
diretamente ainda funcione. Veja [treinamento](/docs/train) para datasets, data
augmentation, multi-GPU e loggers.

## Validação

`val()` devolve um dicionário de chaves `metrics/`: mIoU e acurácia por pixel
para `task="semantic"`, acurácia top-1 e top-5 para `task="classify"`.
`task="embed"` não tem ground truth contra o qual pontuar e levanta
`NotImplementedError` se você chamar `val()` nele.

<code-tabs name="val" />

## Exportação

<export-matrix />

Cada tarefa suporta um subconjunto diferente de formatos, mostrado acima. Um
artefato exportado é recarregado por `LibreYOLO()` pelo sufixo do arquivo, então
um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e devolve o
mesmo `Results`. [Exportação](/docs/export) lista os argumentos que cada
formato aceita.

<code-tabs name="export" />

## Licenciamento

<provenance-box>

A linha "Weights" acima nomeia a licença que se aplica, Apache-2.0, mas nada é
de fato republicado sob a org do LibreYOLO no Hugging Face para esta família:
o LibreYOLO não hospeda nenhum checkpoint LibreDINOv2 próprio. O que
`LibreDINOv2(model_path=None)` baixa é o repositório
`facebook/dinov2-with-registers-small` da própria Meta, intocado.

</provenance-box>

## Citação

<citation-block />
