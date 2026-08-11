---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: prediga, valide e exporte normais de superfície'
description: >-
  Use o MoGe-2 no LibreYOLO para predição densa de normais de superfície.
  Instale, faça predições, valide e exporte os checkpoints oficiais ViT-S, ViT-B
  e ViT-L.
lead: >-
  O MoGe-2 é um modelo de geometria monocular de uma única passagem que prediz
  um campo denso de normais de superfície a partir de uma imagem RGB. O
  LibreYOLO oferece suporte a ele apenas para estimativa de normais, através dos
  checkpoints oficiais ViT-S, ViT-B e ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - normais de superfície
  - estimativa de normais de superfície
  - mapa de normais python
  - geometria monocular
  - surface normal estimation
  - predição densa
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # vetores unitários (H, W, 3) float32
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # graus
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # porcentagem de pixels
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Instalação

O MoGe-2 não precisa de nenhum extra opcional. Tudo o que ele importa está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados automaticamente no primeiro uso: o LibreYOLO busca o
tamanho correspondente direto dos checkpoints oficiais e guarda em cache local.

<code-tabs name="predict" />

O MoGe-2 devolve um campo denso em vez de um conjunto de detecções, então
`result.boxes` fica vazio e `conf`, `iou` e `max_det` não têm efeito.
`result.normal_map` carrega o resultado: um array `(H, W, 3)` de vetores
unitários no referencial de câmera do OpenCV, onde `+x` é para a direita, `+y` é
para baixo, `+z` entra na cena, e uma superfície voltada para a câmera lê
`(0, 0, -1)`. Predizer uma lista de imagens roda um forward pass por imagem;
esta família não tem caminho rápido de batch empilhado. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Três tamanhos de encoder são publicados como checkpoints separados: ViT-S,
ViT-B e ViT-L, todos na mesma resolução de entrada. O harness de benchmark do
LibreYOLO não mediu esta família, então não há números de acurácia publicados
para compará-los; escolha um tamanho de acordo com o seu próprio orçamento de
computação.

## Validação

`val()` mede o erro angular contra um dataset pareado de mapas de normais:
imagens ao lado de PNGs de normais de 16 bits com o mesmo nome-base, com uma
máscara de validade opcional para que pixels de padding e inválidos nunca
contem. Ele devolve o erro angular médio e mediano em graus, além da
porcentagem de pixels dentro de 11.25, 22.5 e 30 graus.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação de normais usa um contrato de runtime de resolução fixa e batch 1:
`dynamic` e um `batch` diferente de 1 são rejeitados, e `imgsz` precisa ser
divisível pelo tamanho de patch do encoder ViT, o que o LibreYOLO confere antes
de a execução começar. Um artefato exportado é carregado de volta por
`LibreYOLO()` pelo sufixo do arquivo, então um arquivo `.onnx` se comporta como
um checkpoint e devolve o mesmo `Results`.

<code-tabs name="export" />

## Licenciamento

<provenance-box>

O LibreYOLO não copia esses checkpoints para a sua própria organização.
`LibreYOLO("LibreMoGe2s-normal.pt")` baixa o tamanho correspondente direto dos
repositórios oficiais no Hugging Face em uma revisão fixada, e verifica o
arquivo contra um checksum SHA-256 registrado antes de usar.

</provenance-box>

## Citação

<citation-block />
