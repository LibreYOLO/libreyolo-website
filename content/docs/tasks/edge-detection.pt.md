---
title: Detecção de bordas
seo_title: Detecção de bordas no LibreYOLO
description: >-
  Prediga um mapa denso de probabilidade de bordas a partir de uma imagem no
  LibreYOLO. Converta um checkpoint, aplique um limiar ao mapa, valide com ODS e
  OIS, e exporte.
lead: >-
  A detecção de bordas prediz o quanto é provável que cada pixel esteja sobre a
  borda de um objeto. O LibreYOLO expõe isso como a tarefa edge, que retorna um
  mapa denso de probabilidade no canvas da imagem original em vez de um conjunto
  de segmentos de reta.
keywords:
  - detecção de bordas python
  - detecção de contornos deep learning
  - mapa de probabilidade de bordas
  - ODS OIS F-measure
  - DexiNed TEED python
last_verified: 1.5.0
snippets:
  predict:
    - label: Predizer um mapa de bordas
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nenhum checkpoint de bordas acompanha o LibreYOLO; converta um antes
        (abaixo).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 em [0, 1]

        print(edges.binary(0.5).sum())    # contagem de pixels de borda em 0.5
    - label: Escolher o seu próprio limiar
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # O mapa contínuo é mantido para que o limiar continue sendo sua
        decisão.

        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Salvar a visualização
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # plot() desenha o mapa; ele é definido para resultados de borda e de
        normais.

        result.plot().save("edges.png")
  val:
    - label: Validar e ler as chaves de métrica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Mudar a varredura e a tolerância de pareamento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado é
        # carregado como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Definição

A tarefa `edge` prediz uma probabilidade por pixel a partir de uma única imagem
RGB: `0` significa não-borda e `1` significa borda. O mapa permanece contínuo,
então escolher o limiar que o transforma em uma imagem binária de bordas fica a
cargo de quem chama, e o limiar certo depende do dataset e do uso posterior.

Uma predição preenche `result.edges`, um payload `EdgeMap` que guarda um array
float32 `(H, W)` em `[0, 1]` no canvas da imagem original. `.array` retorna esse
mapa como NumPy e `.binary(threshold)` retorna uma máscara booleana.
`result.boxes` fica vazio, então `conf`, `iou` e `max_det` não têm efeito.
`Results.plot()` cobre essa tarefa e desenha o mapa diretamente.

## Modelos

Três famílias atendem a tarefa `edge`.

O [DexiNed](/docs/models/dexined), a Dense Extreme Inception Network, funde
várias saídas laterais em um único mapa de probabilidade e roda em 352 px
nativos.

O [TEED](/docs/models/teed), o Tiny and Efficient Edge Detector, é uma rede
pequena nos mesmos 352 px nativos, com stride de downsample de 4 contra os 16 do
DexiNed, então ele aceita mais valores de `imgsz`.

O [LibreMODUS](/docs/models/libremodus) produz bordas no estilo Canny como um
dos alvos de um modelo any-to-any. Ele precisa do extra `modus` e da sua própria
conta autenticada no Hugging Face, e não oferece nem `val()` nem `export()`,
então não participa das seções de validação e exportação abaixo.

## Predição

O LibreYOLO não publica nenhum checkpoint de bordas. Os pesos de DexiNed e TEED
lançados oficialmente são treinados no BIPED, cujos termos publicados de dataset
restringem o uso a fins não comerciais, então o LibreYOLO não os espelha.
Converta um checkpoint que você tenha licença para usar e depois carregue o
arquivo convertido pelo caminho:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

O nome do arquivo precisa carregar o sufixo de tarefa `-edge` para o loader
reconhecê-lo. `imgsz` precisa ser divisível pelo stride de downsample da rede, e
o LibreYOLO levanta um erro claro informando o divisor quando não é. Veja
[predição](/docs/predict) para fontes, streaming e tratamento dos resultados.

## Formato do dataset

A validação de bordas pareia cada imagem RGB com um mapa de canal único de mesmo
nome-base e mesma resolução, mais uma máscara de validade opcional.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

O alvo é um PNG ou TIF de canal único, não uma visualização RGB. Mapas inteiros
são divididos pelo máximo do seu dtype; mapas float já precisam ser finitos e
estar em `[0, 1]`. Pixels da máscara contam como válidos quando são diferentes de
zero, e pixels de padding nunca contribuem para uma métrica. `edge_invert: true`
cobre fontes que armazenam bordas pretas sobre branco. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

Nenhuma família de bordas no LibreYOLO tem implementação de treinamento:
`train()` levanta `NotImplementedError` nas três. Cada página de modelo indica o
script de conversão que transforma um checkpoint treinado em outro lugar em um
que o LibreYOLO consegue carregar.

## Validação

`val()` reporta as F-measures no estilo BSDS. As predições contínuas são
primeiro afinadas com supressão de não-máximos de gradiente em quatro direções, e
depois os pixels de borda preditos e os do ground truth são pareados um a um
dentro de uma tolerância de distância.

<code-tabs name="val" />

`metrics/ODS` é a F-measure na escala ótima do dataset: as contagens de
pareamento são agregadas em todo o dataset a cada limiar, e a melhor dessas
F-measures agregadas é reportada. Ela também é o `fitness`, o número que a
seleção do melhor checkpoint lê. `metrics/OIS` é a F-measure na escala ótima da
imagem, a média sobre as imagens da melhor F-measure de cada imagem, então ela
deixa cada imagem escolher o próprio limiar. `metrics/best_threshold` é o limiar
único que produziu o ODS, que é o que se deve reutilizar em `edges.binary()` na
inferência.

Dois argumentos moldam a varredura. `edge_thresholds` é o conjunto de limiares
testados, com padrão de 0.01 a 0.99 em centésimos. `edge_max_dist` é a
tolerância de pareamento como fração da diagonal da imagem, com padrão
`0.0075`; um par mais distante que isso não é um pareamento.

## Exportação

Um modelo de bordas exportado é carregado de volta pelo `LibreYOLO()` a partir
do sufixo do arquivo, então um arquivo `.onnx` se comporta como um checkpoint e
retorna os mesmos `Results`.

<code-tabs name="export" />

A exportação de bordas usa um contrato de runtime de resolução fixa e batch 1:
`dynamic` e um `batch` diferente de 1 são rejeitados, e o grafo exportado gera um
único mapa de probabilidade fundido. A cobertura por formato está nas páginas do
[DexiNed](/docs/models/dexined) e do [TEED](/docs/models/teed) e na
[matriz completa de exportação](/docs/reference/export-matrix).
[Exportação](/docs/export) lista os argumentos que cada formato aceita.
