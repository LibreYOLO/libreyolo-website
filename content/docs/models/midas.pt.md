---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estimativa de profundidade monocular no LibreYOLO'
description: >-
  Use o MiDaS no LibreYOLO para estimativa de profundidade monocular. Instale,
  faça predições, valide e exporte duas variantes sob licença MIT, baixadas da
  isl-org.
lead: >-
  O MiDaS é estimativa de profundidade relativa monocular treinada com uma loss
  invariante a escala e deslocamento sobre datasets mistos, a linha de trabalho
  que estabeleceu o protocolo de transferência zero-shot de profundidade que as
  famílias seguintes reutilizam. O LibreYOLO oferece suporte a ele na tarefa de
  profundidade: predição e validação zero-shot, sem caminho de treinamento.
keywords:
  - MiDaS
  - estimativa de profundidade monocular
  - mapa de profundidade python
  - profundidade a partir de uma imagem
  - depth estimation python
  - DPT
  - profundidade relativa
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ainda não está em disco: o LibreYOLO baixa do release oficial

        # isl-org/MiDaS no GitHub e confere contra um SHA-256 fixado antes de
        usar.

        model = LibreYOLO("LibreMiDaSl-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)


        depth = result.depth_map

        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Variante Small
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encoder EfficientNet-Lite3, menor e mais rápido que o tamanho l,
        DPT-Large.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Instalação

O MiDaS não precisa de nenhum extra opcional. Tudo o que ele importa está na instalação base.

```bash
pip install libreyolo
```

## Predição

O MiDaS é a única família de profundidade que o LibreYOLO não republica na sua
própria organização no Hugging Face. Pedir um checkpoint pelo nome de arquivo do
LibreYOLO baixa o asset oficial correspondente direto dos releases do
`isl-org/MiDaS` no GitHub, confere contra um SHA-256 fixado e o embrulha com os
metadados de checkpoint do LibreYOLO antes do primeiro uso; as execuções
seguintes reutilizam o arquivo local em cache. Veja Licenciamento para saber por
quê.

<code-tabs name="predict" />

`result.depth_map` carrega um mapa denso de profundidade inversa relativa:
valores maiores significam mais perto da câmera, e os valores não têm unidade
métrica nem escala entre imagens. `save=True` grava em disco uma visualização
desse mapa com mapa de cores; `Results.plot()` não cobre esta família, já que
está definido apenas para normais de superfície e bordas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Duas variantes com encoders diferentes, não apenas escalas diferentes do mesmo
encoder. `s` é o MiDaS v2.1 Small, um encoder EfficientNet-Lite3. `l` é o
DPT-Large, um encoder ViT-L/16 com o decoder DPT que o MiDaS introduziu para
predição densa. Elas também fazem o pré-processamento de forma diferente: `s`
usa um redimensionamento de proporção com limite superior e normalização por
média/desvio padrão do ImageNet, `l` usa um redimensionamento de proporção
mínimo com média e desvio padrão de 0.5. Escolha `s` para uma CNN mais leve, `l`
para a acurácia do decoder transformer.

Treinamento não é oferecido para esta família. `LibreMiDaS.train()` levanta
`NotImplementedError` incondicionalmente.

## Validação

`val()` roda o validador de profundidade compartilhado: ele alinha cada predição
ao seu ground truth com uma escala e um deslocamento de mínimos quadrados por
imagem, e depois reporta as métricas padrão de profundidade relativa zero-shot,
AbsRel, RMSE e os três limiares delta.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`, com `depth_map` no lugar dos boxes.

<code-tabs name="export" />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
