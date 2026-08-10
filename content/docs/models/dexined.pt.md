---
title: DexiNed
families: [dexined]
seo_title: "DexiNed: detecção de bordas, com o seu próprio checkpoint"
description: "Use o DexiNed no LibreYOLO para predição densa de probabilidade de bordas. Converta um checkpoint licenciado e depois faça predições, valide e exporte."
lead: "O DexiNed (Dense Extreme Inception Network) é uma rede convolucional que prediz um mapa denso de probabilidade de bordas a partir de uma imagem RGB. O LibreYOLO envolve a arquitetura dele apenas para detecção de bordas; nenhum checkpoint acompanha a biblioteca."
keywords: [DexiNed, "Dense Extreme Inception Network", "detecção de bordas", "detecção de bordas python", "edge detection", BIPED, "mapa de bordas de imagem"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 em [0, 1]
        print(edges.binary(0.5).sum())  # contagem de pixels de borda após o limiar
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreDexiNedb-edge.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-measure na escala ótima do dataset
        print(metrics["metrics/OIS"])   # F-measure na escala ótima da imagem
    - label: CLI
      language: bash
      code: |
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx imgsz=352
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt imgsz=352 half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
---

## Instalação

O DexiNed não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base.

```bash
pip install libreyolo
```

## Predição

O LibreYOLO não inclui nenhum checkpoint do DexiNed. Os pesos lançados
oficialmente são treinados no BIPED, cujos termos publicados de dataset
restringem o uso a fins não comerciais, então o LibreYOLO não os espelha.
Converta um checkpoint que você tenha licença para usar com
`weights/convert_dexined_weights.py`, que confere as chaves dos tensores
contra a arquitetura de runtime antes de escrever um arquivo que o LibreYOLO
consegue carregar diretamente:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` guarda o resultado: um array float32 `(H, W)` em `[0, 1]`, com
`.binary(threshold)` retornando uma máscara booleana de bordas. Não há
bounding boxes, então `conf`, `iou` e `max_det` não têm efeito. Veja
[predição](/docs/predict) para fontes, streaming e tratamento dos resultados.

## Variantes

O DexiNed vem em um único tamanho no LibreYOLO. O harness de benchmark do
LibreYOLO não mediu essa família, então não há números publicados para
comparação.

## Validação

`val()` reporta as F-measures ODS e OIS no estilo BSDS contra um dataset
pareado de bordas: imagens ao lado de mapas de bordas de mesmo nome-base, com
uma máscara de validade opcional para que pixels de padding nunca contem.
`imgsz` precisa ser divisível pelo stride de downsample da rede, e o LibreYOLO
levanta um erro claro se não for.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação de bordas usa um contrato de runtime de resolução fixa e batch 1:
`dynamic` e um `batch` diferente de 1 são rejeitados, e o grafo exportado gera
um único mapa de probabilidade fundido. Um artefato exportado é carregado de
volta pelo `LibreYOLO()` a partir do sufixo do arquivo, então um arquivo
`.onnx` se comporta como um checkpoint e retorna os mesmos `Results`.

<code-tabs name="export" />

## Licenciamento

<provenance-box>

O LibreYOLO não publica nenhum checkpoint do DexiNed. Nada é espelhado na
organização LibreYOLO; em vez disso, converta um checkpoint para o qual você
tenha licença com `weights/convert_dexined_weights.py`.

</provenance-box>

## Citação

<citation-block />
