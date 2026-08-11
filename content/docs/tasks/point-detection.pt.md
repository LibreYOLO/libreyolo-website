---
title: Detecção de pontos
seo_title: Detecção de pontos e contagem no LibreYOLO
description: >-
  Localize objetos como pontos únicos em vez de caixas no LibreYOLO. Preveja
  centroides, conte objetos, treine o FOMO e leia as métricas de ponto.
lead: >-
  A detecção de pontos devolve uma localização x, y por objeto em vez de um
  bounding box. O LibreYOLO expõe isso como a tarefa point, e uma predição
  carrega uma linha de x, y, classe e confiança por objeto.
keywords:
  - detecção de pontos python
  - contar objetos em imagem python
  - detecção de centroide
  - contagem de objetos visão computacional
  - FOMO localização de pontos
  - localizar objetos por ponto python
last_verified: 1.5.0
snippets:
  predict:
    - label: Prever pontos e contá-los
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Os pesos do LibreFOMO não têm download automático. Baixe antes um

        # checkpoint de https://huggingface.co/LibreYOLO e carregue pelo caminho
        local.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # contagem de objetos

        print(points.xy)       # (N, 2) centros em pixels da imagem original

        print(points.cls, points.conf)
    - label: Coordenadas normalizadas e contagens por classe
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # os mesmos centros em [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Treinar o FOMO em um dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Prever com o checkpoint treinado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() recarrega o melhor checkpoint no mesmo objeto, então o
        # modelo prevê com os pesos treinados quando a chamada retorna.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validar e ler as chaves das métricas
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # erro médio de localização

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # erro de
        contagem
    - label: Mudar os limiares de distância
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Os limites da varredura fazem parte do texto da chave, então uma

        # varredura personalizada renomeia as chaves de mAP que ela produz.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Rodar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definição

A tarefa `point` localiza cada objeto com uma única coordenada x, y e uma classe,
sem largura, altura ou máscara. Como uma predição é uma lista plana de objetos, a
contagem de linhas é a contagem de objetos, e é isso que faz desta a tarefa de
contagem.

Uma predição preenche `result.points`, um payload `Points` que envolve um array
`(N, 4)` de linhas `x, y, classe, confiança` em pixels da imagem original. `.xy`
devolve as coordenadas, `.xyn` as mesmas coordenadas divididas pelo tamanho da
imagem, `.cls` os índices de classe e `.conf` os scores; `len()` devolve o número
de pontos. `result.boxes` fica vazio, então `iou` e `max_det` não têm sobre o que
agir.

## Modelos

Três famílias atendem `point`, e elas não são intercambiáveis.

[FOMO](/docs/models/fomo) é a opção de vocabulário fixo: um classificador de
grade que rotula cada célula de uma grade de baixa resolução como fundo ou centro
de objeto. É a única família de pontos que o LibreYOLO consegue treinar, e a
única que exporta.

[LocateAnything](/docs/models/locate-anything) recebe texto em vez de um índice de
classe, então o vocabulário é qualquer frase que você escrever. Ele precisa do
extra `vlm`, é construído como `LibreLocateAnything` e não pela factory
`LibreYOLO()`, e seus pesos são restritos a uso não comercial. Os termos exatos, e
as duas licenças adicionais que o checkpoint compõe, estão na página dele.

[SenseNova-Vision](/docs/models/sensenova-vision) chega a `point` pelo mesmo
checkpoint de geração por prompt que usa para outras seis tarefas, carregado com
`LibreVLM("sensenova-vision", task="point")`. Ele precisa do extra `sensenova`, e
cada predição é uma passagem de geração sobre um modelo de 7B, então espere uma
latência por imagem bem mais alta que a de um detector feito sob medida. Seus
pesos são não comerciais; a licença está na página dele.

## Predição

Os pesos do LibreFOMO são a única exceção ao download automático neste site.
`LibreYOLO("LibreFOMOs-point.pt")` procura esse arquivo no disco e levanta um
`ValueError` nomeando-o em vez de baixá-lo. Baixe antes um checkpoint da
[organização LibreYOLO](https://huggingface.co/LibreYOLO) no Hugging Face e
carregue pelo caminho local, ou treine o seu.

<code-tabs name="predict" />

O nome do arquivo precisa carregar o sufixo de tarefa `-point` para o loader
reconhecê-lo. `predict(..., nms_radius=1)` controla quantas células de grade duas
detecções do FOMO precisam ter entre si para que ambas sobrevivam. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

`point` não tem formato de rótulos próprio. As famílias de pontos leem o layout
padrão de detecção do YOLO e derivam um centro de cada linha de caixa, então
`cx cy` é o ponto e `w h` só decidem se a linha é válida.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Cada arquivo de rótulos contém uma linha por objeto, com coordenadas
normalizadas:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Um arquivo de rótulos ausente ou vazio significa nenhum objeto. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

O FOMO é a única família de pontos com implementação de treinamento. `train()` no
LocateAnything e no SenseNova-Vision levanta `NotImplementedError`; faça
fine-tuning desses modelos upstream e carregue o resultado.

<code-tabs name="train" />

`imgsz` não é uma escolha livre no FOMO: ele assume por padrão a resolução nativa
do checkpoint carregado, e passar um valor diferente levanta `ValueError`
nomeando o tamanho que ele espera. Veja [treinamento](/docs/train) para datasets,
loggers e multi-GPU, e a [página do FOMO](/docs/models/fomo) para os padrões
desta família.

## Validação

`val()` casa os pontos previstos com os pontos de ground truth um a um pelo
algoritmo húngaro, ao longo de uma varredura de limiares de distância. Um limiar
é uma distância euclidiana em coordenadas normalizadas da imagem, e a varredura
padrão são dez valores de 0.01 a 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` e `metrics/f1` são médias macro sobre as
classes no limiar mais estrito da varredura, 0.01 por padrão.
`metrics/mAP@0.01` é a precisão média nesse mesmo limiar, e
`metrics/mAP@[0.01:0.10]` é a média sobre toda a varredura. Esse valor da
varredura também é o `fitness`, o número que a seleção do melhor checkpoint lê.
Ambas as chaves de mAP são construídas a partir dos limiares em uso, então passar
`dist_thresholds=` as renomeia.

`metrics/MLE` é a distância média entre os pares casados no limiar mais estrito,
nas mesmas unidades normalizadas. `metrics/MAE` e `metrics/RMSE` são métricas de
contagem, não de localização: elas medem a diferença, por imagem, entre o número
de pontos previstos e o de pontos do ground truth.

O FOMO acrescenta um segundo grupo, em nível de grade, além desses. Ele varre a
confiança e o `nms_radius` e publica a combinação de melhor F1 como
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` e
`metrics/grid_FN`, com as configurações que a produziram em `decode/threshold` e
`decode/nms_radius`.

## Exportação

O FOMO exporta pelo caminho de exportação compartilhado, e um artefato exportado
é carregado de volta por `LibreYOLO()` a partir do sufixo do arquivo, então um
arquivo `.onnx` ou `.engine` se comporta como um checkpoint e devolve o mesmo
`Results`.

<code-tabs name="export" />

A cobertura por formato está na [página do FOMO](/docs/models/fomo) e na
[matriz completa de exportação](/docs/reference/export-matrix). LocateAnything e
SenseNova-Vision não exportam: `export()` levanta erro em ambos, porque um modelo
generativo não tem grafo de detecção rastreável.
