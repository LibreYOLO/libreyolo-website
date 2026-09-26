---
title: Estimativa de profundidade
seo_title: Estimativa monocular de profundidade no LibreYOLO
description: >-
  Preveja um mapa denso de profundidade relativa a partir de uma única imagem no
  LibreYOLO. Compare as famílias de profundidade, interprete as métricas de
  profundidade e exporte um modelo de profundidade.
lead: >-
  A estimativa de profundidade prevê a que distância cada pixel está da câmera
  usando uma única imagem. O LibreYOLO expõe isso como a tarefa depth, que
  devolve um mapa denso de profundidade inversa relativa sobre o canvas da
  imagem original.
keywords:
  - estimativa de profundidade monocular python
  - mapa de profundidade de uma imagem
  - modelo de profundidade relativa
  - depth anything libreyolo
  - calcular profundidade com python
last_verified: 1.5.0
snippets:
  predict:
    - label: Prever um mapa de profundidade
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) no canvas original
        print(depth.min, depth.max, depth.mean)
    - label: Trabalhar com os valores
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # maior é mais perto; sem unidade métrica, sem
        escala

        gray = depth.normalized() # reescalado para [0, 1] para visualização

        print(raw.shape, float(gray.max()))
    - label: Uma alternativa compacta
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Mesmo contrato de tarefa, uma rede bem menor feita para runtimes de
        borda (edge).

        model = LibreYOLO("LibreZipDepthb-depth.pt")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
  val:
    - label: Validar e ler as chaves das métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Rodar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Definição

A tarefa `depth` prevê um valor por pixel a partir de uma única imagem RGB. O
LibreYOLO define esse valor como profundidade inversa relativa: maior significa
mais perto da câmera, e os números não carregam unidade métrica nem uma escala
que se mantenha entre duas imagens. Comparar a profundidade entre dois pixels da
mesma predição faz sentido; comparar um valor com o valor de outra imagem, não.

Uma predição preenche `result.depth_map`, um payload `DepthMap` que guarda um
array `(H, W)` sobre o canvas da imagem original. `.min`, `.max` e `.mean` leem
os valores finitos, e `.normalized()` reescala o mapa para `[0, 1]` para
exibição. `result.boxes` fica vazio, então `conf`, `iou` e `max_det` não têm
efeito, e `save=True` grava uma imagem do mapa com mapa de cores aplicado, em
vez de uma foto anotada.

## Modelos

Seis famílias atendem `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) combina um encoder DINOv2 com
um decoder DPT e é a opção padrão de uso geral aqui. O licenciamento decide o
tamanho tanto quanto a acurácia: o checkpoint Small é Apache-2.0, enquanto Base e
Large são de uso não comercial, então confira a tabela de checkpoints na página
dele antes de escolher um.

[Depth Anything 3](/docs/models/depth-anything-3) porta o checkpoint
DA3MONO-LARGE, um transformer comum, sem nenhuma especialização arquitetural para
profundidade.

[ZipDepth](/docs/models/zipdepth) é o nível compacto: uma CNN reparametrizável
destilada do Depth Anything V2 Large, com um segundo checkpoint cujo decoder
evita operações de gather e unfold para compiladores de NPU que não as têm.

[MiDaS](/docs/models/midas) é a linha de trabalho que estabeleceu o protocolo de
profundidade relativa zero-shot com que as outras famílias são medidas. É a única
família de profundidade que o LibreYOLO não republica: pedir um checkpoint baixa
o asset oficial do release do GitHub dos autores e confere um SHA-256 fixado.

[LibreMODUS](/docs/models/libremodus) chega à profundidade como um dos alvos de
um modelo any-to-any, e não como uma cabeça dedicada. Precisa do extra `modus` e
da sua própria conta autenticada no Hugging Face, e não oferece nem `val()` nem
`export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) gera o mapa de profundidade
como uma imagem por meio de um decode por difusão, a partir do mesmo checkpoint
de 7B que atende suas outras seis tarefas. Precisa do extra `sensenova`, e seus
pesos são restritos a uso não comercial; a licença está na página dele.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente, exceto nas duas famílias citadas acima.

<code-tabs name="predict" />

A resolução de entrada é limitada por família. Depth Anything V2 e Depth Anything
3 se apoiam em uma grade de patches DINOv2, então `imgsz` precisa ser divisível
por 14, o que o LibreYOLO confere antes de rodar. `Results.plot()` não cobre esta
tarefa; está definido apenas para normais de superfície e bordas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

A validação de profundidade emparelha cada imagem com um mapa de profundidade
denso de canal único e mesma resolução, encontrado ao substituir o diretório de
profundidade no caminho da imagem.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Os mapas são PNG ou TIF de canal único, ou `.npy`. Os valores são profundidade
pura em uma unidade que o dataset mantém consistente, e pixels `0`, negativos,
NaN e infinitos marcam amostras inválidas, que ficam de fora das métricas. Mapas
inteiros são divididos por `depth_scale`, cujo padrão é `256.0`, a convenção dos
PNG de 16 bits; mapas `.npy` em ponto flutuante são usados como estão.
`depth_stem_suffix` e `depth_mask_suffix` cobrem datasets que nomeiam de outro
jeito seus arquivos de profundidade ou suas máscaras de validade. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

Nenhuma família de profundidade no LibreYOLO tem implementação de treinamento:
`train()` levanta `NotImplementedError` nas seis. A página de cada modelo indica o
script de conversão que transforma um checkpoint treinado upstream em um que o
LibreYOLO consegue carregar.

## Validação

`val()` roda o validador de profundidade compartilhado. A profundidade relativa
não tem escala absoluta, então cada predição é primeiro ajustada ao inverso do
seu ground truth com uma escala e um deslocamento de mínimos quadrados calculados
por imagem, e depois invertida de volta para profundidade. Todas as métricas
abaixo são calculadas por imagem sobre esse mapa alinhado, com média sobre o
dataset, contando apenas os pixels que o dataset marca como válidos.

<code-tabs name="val" />

`metrics/abs_rel` é o erro relativo absoluto médio, o resíduo dividido pela
profundidade do ground truth, e quanto menor, melhor. `metrics/rmse` é a raiz do
erro quadrático médio na própria unidade de profundidade do dataset, também
quanto menor, melhor. `metrics/delta1`, `metrics/delta2` e `metrics/delta3` são
as acurácias por limiar: a fração de pixels válidos cuja razão em relação ao
ground truth, tomada na direção que for maior, fica abaixo de 1.25, de 1.25 ao
quadrado e de 1.25 ao cubo, então quanto maior, melhor. `metrics/delta1` também é
o `fitness`, o número que a seleção do melhor checkpoint lê.

## Exportação

Um modelo de profundidade exportado é carregado de volta por `LibreYOLO()` pelo
sufixo do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e devolve o mesmo `Results`, com `depth_map` no lugar dos boxes.

<code-tabs name="export" />

A cobertura varia por família, e o Depth Anything 3 rejeita qualquer formato fora
do seu conjunto validado em vez de tentar uma conversão não validada. Confira a
página do modelo e a
[matriz completa de exportação](/docs/reference/export-matrix) antes de se
comprometer com um alvo. LibreMODUS e SenseNova-Vision não exportam de jeito
nenhum. [Exportação](/docs/export) lista os argumentos que cada formato aceita.
