---
title: Classificação de imagens
seo_title: Classificação de imagens no LibreYOLO
description: >-
  Rotule uma imagem inteira no LibreYOLO: as famílias que atendem à tarefa, o
  layout de dataset ImageFolder e as chamadas de predição, treinamento,
  validação e exportação.
lead: >-
  A classificação de imagens atribui uma distribuição de rótulos à imagem
  inteira e não localiza nada dentro dela. A chave da tarefa é classify.
keywords:
  - classificação de imagens python
  - treinar classificador de imagens
  - dataset ImageFolder
  - acurácia top-1
  - classificação zero-shot
  - biblioteca de classificação de imagens MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -cls no nome do arquivo seleciona a tarefa, então não é
        # preciso passar o argumento task.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: A distribuição completa
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data é o vetor completo (C,); top5/top5conf são visões ordenadas.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, sem treinamento'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O CLIP pontua a imagem em relação a prompts de texto, então o conjunto
        de

        # rótulos é definido na hora da chamada em vez de embutido no
        checkpoint.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # imagenette160 é um nome de dataset conhecido e é baixado no primeiro
        uso.

        # Passe um diretório com um split train/ para seus próprios dados.

        model = LibreYOLO("LibreResNet50-cls.pt")

        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() retorna um dict simples, não um objeto.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # é carregado como um checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definição

A classificação de imagens produz uma pontuação por classe para a imagem
inteira e nenhuma coordenada. Ela responde o que está na imagem, nunca onde, e
é isso que a separa da [detecção de objetos](/docs/tasks/object-detection).

`classify` é a chave canônica da tarefa, e o sufixo `-cls` no nome do arquivo
de um checkpoint a seleciona. Esse sufixo é obrigatório, não opcional, nas
famílias de classificação, então `LibreResNet50.pt` não é lido como
classificador e só `LibreResNet50-cls.pt` é.

`predict()` preenche `result.probs` e deixa `boxes` vazio. `.data` é o vetor
completo de pontuações, `.top1` o índice da maior pontuação e `.top1conf` o
valor dela, `.top5` os cinco maiores índices em ordem decrescente e `.top5conf`
as pontuações deles. Os índices apontam para `result.names`. Fatiar um objeto
`Results` nunca trunca `probs`, porque o vetor pertence à imagem, e não a uma
linha.

## Modelos

Cinco famílias treinam e fazem predição: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) e
[DINOv2](/docs/models/dinov2). As quatro primeiras rodam no pacote base e
trazem pesos publicados. O DINOv2 precisa de `pip install "libreyolo[rfdetr]"`
e não tem checkpoint hospedado pelo LibreYOLO: ele carrega o backbone original
com uma cabeça linear inicializada aleatoriamente, então é um ponto de partida
para fine-tuning, e não um preditor pronto.

Outras cinco fazem predição, validação e exportação, mas o `train()` delas levanta
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) e
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) e [SigLIP2](/docs/models/siglip2) classificam sem um
conjunto fixo de rótulos. Eles pontuam a imagem em relação a prompts de texto,
então `set_classes()` define as classes na hora da chamada e não existe nenhuma
etapa de treinamento para um novo conjunto de rótulos. Ambos também atendem à
tarefa `embed`.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

`conf`, `iou` e `max_det` não têm efeito aqui: não há candidatos para filtrar
por limiar ou suprimir, apenas uma distribuição. Veja
[predição](/docs/predict) para fontes, streaming e tratamento dos resultados.

## Formato do dataset

A classificação usa uma árvore de diretórios, não arquivos de rótulos nem um
YAML. `data` é a raiz do dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` é obrigatório para o treinamento e define o mapeamento de classe para
índice pelo nome das pastas em ordem alfabética, então a primeira pasta na
ordem alfabética vira a classe 0. `val/` é obrigatório para a validação. Um
split `test/` pode existir, e os comandos padrão de treinamento e validação não
o usam. Qualquer split que não seja `train` precisa conter os mesmos nomes de
pastas de classe do conjunto de classes esperado, que é o que faz uma
divergência falhar em alto e bom som em vez de contar como uma predição errada.
As extensões de imagem aceitas são `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`,
`.tif` e `.tiff`.

`data` aceita três coisas: um caminho para um diretório que contenha um split
`train/`, uma URL `.zip`, ou um dos nomes de dataset conhecidos,
`imagenette160` e `smoke10`, que são baixados e ficam em cache no primeiro uso.

O loader canônico é `libreyolo.data.classify_dataset`.

## Treinamento

<code-tabs name="train" />

Não há `nc` para declarar: a contagem de classes vem dos nomes das pastas
dentro de `train/`, e a camada linear final é reconstruída para bater com ela
enquanto o backbone é transferido sem alteração. Veja
[treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário simples de chaves `metrics/`, calculado sobre o
split `val/` da raiz do dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` é a proporção de imagens cuja classe de maior pontuação
é a verdadeira, e é o número principal, o que o treinamento usa para escolher a
melhor época. `metrics/accuracy_top5` é a proporção cuja classe verdadeira
aparece em qualquer posição entre as cinco classes de maior pontuação, o que
diz menos quanto menos classes o dataset tiver. O dicionário também traz
`fitness`, uma cópia do valor top-1.

## Exportação

<code-tabs name="export" />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` a partir do
sufixo do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e retorna o mesmo `Results`. A cobertura de formatos varia por
família; a matriz em cada página de modelo é gerada a partir do conjunto
validado, e não digitada à mão. Veja [exportação e deploy](/docs/export) para os formatos, seus extras e
suas restrições.
