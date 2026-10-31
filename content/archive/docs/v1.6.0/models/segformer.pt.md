---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: segmentação semântica no LibreYOLO'
description: >-
  Use o SegFormer no LibreYOLO para segmentação semântica ADE20K nos tamanhos b0
  a b5. Instale, faça predições, treine e exporte; os pesos pré-treinados são
  não comerciais.
lead: >-
  O SegFormer é um transformer de segmentação semântica que junta um encoder
  hierárquico Mix Transformer (MiT) a uma cabeça de decode leve, toda em MLP,
  dispensando os decoders pesados e as codificações posicionais fixas de que os
  transformers de segmentação anteriores precisavam. O LibreYOLO o suporta para
  uma tarefa, segmentação semântica, em seis tamanhos.
keywords:
  - SegFormer
  - segmentação semântica
  - Mix Transformer
  - MiT
  - ADE20K
  - segformer pytorch
  - segmentação semântica python
  - transformer para segmentação
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (fine-tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Do zero
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # Sem model_path: init aleatório, nada é baixado. A única rota para
        # pesos livres do termo não comercial dos checkpoints pré-treinados.
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Instalação

O SegFormer não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

`result.semantic_mask` traz o mapa denso de classes: `.data` é um tensor
`(H, W)` de ids de classe no tamanho original da imagem, e `.classes` lista os
ids de classe realmente presentes. `result.boxes` é `None`, já que não há
detecções por instância. `conf` e `iou` são aceitos por paridade de API, mas não
mudam a saída: o modelo devolve uma classe por pixel, não detecções por
instância para filtrar ou desduplicar. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

Seis tamanhos, de b0 a b5, alargando e aprofundando o encoder Mix Transformer a
cada passo, mantendo o mesmo design de cabeça de decode toda em MLP.

<checkpoint-table />

## Treinamento

`train()` faz fine-tuning de um checkpoint publicado por padrão. Em vez disso,
não passe nenhum `model_path` para `LibreSegformer(...)` e ele constrói o modelo
com encoder e cabeça inicializados aleatoriamente, treinando do zero — a única
rota para pesos que não carregam nenhuma das restrições não comerciais dos
checkpoints pré-treinados (veja [Licenciamento](#licensing)).

<code-tabs name="train" />

Sem mexer em nada, o trainer segue a receita do artigo do SegFormer para o
ADE20K: AdamW com um learning rate base no backbone e a cabeça de decode
treinada a 10x essa taxa, weight decay em tudo menos no LayerNorm e na
convolução posicional do Mix-FFN, e um schedule de decaimento linear com
warmup. A convergência dos tamanhos maiores, de b3 a b5, não foi validada de
ponta a ponta.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário de chaves `metrics/`: mIoU e pixel accuracy,
medidos sobre qualquer dataset no formato em que você treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O encoder e a cabeça de decode do LibreSegformer são um port para PyTorch da
implementação do SegFormer do Hugging Face Transformers, sob Apache-2.0, e não
do NVlabs/SegFormer: o repositório original da NVIDIA nunca foi lido nem
copiado, e é creditado aqui apenas para atribuição aos autores do artigo. Só os
checkpoints pré-treinados acima carregam a restrição não comercial da NVIDIA; a
arquitetura e o código do próprio LibreYOLO seguem MIT do começo ao fim.

</provenance-box>

## Citação

<citation-block />
