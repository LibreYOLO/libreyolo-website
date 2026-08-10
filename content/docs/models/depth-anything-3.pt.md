---
title: Depth Anything 3
families: [depth_anything3]
seo_title: "Depth Anything 3: prediga profundidade monocular no LibreYOLO"
description: "Use o Depth Anything 3 no LibreYOLO para estimativa de profundidade monocular. Instale, faça predições, valide e exporte o checkpoint DA3MONO-LARGE, Apache-2.0."
lead: "O Depth Anything 3 é um transformer DINOv2 comum treinado para predizer profundidade e geometria de câmera a partir de uma ou mais vistas, sem nenhuma especialização arquitetural. O LibreYOLO porta seu checkpoint DA3MONO-LARGE para a tarefa de profundidade: predição e validação zero-shot, sem caminho de treinamento."
keywords: [Depth Anything 3, DA3, "estimativa de profundidade monocular", DINOv2, "mapa de profundidade python", "profundidade relativa", "depth estimation python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnything3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Ler o mapa de profundidade
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: denso (H, W), maior = mais perto
        raw = depth.data                # tensor, sem unidade métrica nem escala entre imagens
        normalized = depth.normalized() # reescalado para [0, 1] para visualização
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx
        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Instalação

O Depth Anything 3 não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

`result.depth_map` carrega um mapa denso de profundidade inversa relativa:
valores maiores significam mais perto da câmera, e os valores não têm unidade
métrica nem escala entre imagens. O checkpoint original emite profundidade
relativa positiva; o wrapper de rede do LibreYOLO a inverte e reproduz o
tratamento oficial de céu, para que a saída siga o contrato de profundidade
compartilhado do LibreYOLO. `save=True` grava em disco uma visualização com
mapa de cores desse mapa; `Results.plot()` não cobre esta família, já que está
definido apenas para normais de superfície e bordas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Um único tamanho, `l`, em uma resolução de entrada fixa. O DA3 original também
publica checkpoints Small e Base de qualquer vista (any-view), um checkpoint de
profundidade métrica e checkpoints Nested e Giant; o LibreYOLO não expõe nenhum
deles. A profundidade métrica precisa de um contrato público diferente da tarefa
de profundidade inversa relativa do LibreYOLO, e os checkpoints any-view e
Nested precisam de uma API de câmera multi-imagem que o LibreYOLO não oferece.
Os checkpoints any-view Large e Giant também são CC-BY-NC-4.0 e não são
referenciados por nenhum caminho de download do LibreYOLO.

O treinamento não é oferecido para esta família. `LibreDepthAnything3.train()`
levanta `NotImplementedError` incondicionalmente; treine no projeto original e
converta um checkpoint DA3MONO-LARGE compatível com
`weights/convert_depth_anything3_weights.py`.

## Validação

`val()` roda o validador de profundidade compartilhado: ele alinha cada predição
ao seu ground truth com uma escala e um deslocamento de mínimos quadrados por
imagem, e então reporta as métricas padrão de profundidade relativa zero-shot,
AbsRel, RMSE e os três limiares delta.

<code-tabs name="val" />

## Exportação

<export-matrix />

A exportação é restrita a cinco formatos nesta família: ONNX, TorchScript,
ExecuTorch, TensorRT e OpenVINO. Pedir qualquer outro formato levanta
`NotImplementedError` em vez de tentar uma conversão não validada. Um artefato
exportado é carregado de volta por `LibreYOLO()` pelo sufixo do arquivo, então
um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e devolve o mesmo
`Results`, com `depth_map` no lugar dos boxes.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
