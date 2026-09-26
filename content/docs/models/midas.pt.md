---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estimativa de profundidade monocular no LibreYOLO'
description: >-
  Rode inferência de profundidade relativa com MiDaS no LibreYOLO. Os
  checkpoints s e l usam espelhos do LibreYOLO sob a licença MIT do publicador.
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
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Baixa o checkpoint espelhado no primeiro uso.
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
source_hash: 64537aa3ad3a6f8f
---

## Instalação

MiDaS precisa do extra `midas` para seus encoders timm.

```bash
pip install "libreyolo[midas]"
```

## Predição

Os checkpoints s e l são baixados dos espelhos do LibreYOLO sob a licença MIT do publicador e ficam em cache localmente.

<code-tabs name="predict" />

`result.depth_map` contém um mapa denso de profundidade inversa relativa: valores maiores indicam maior proximidade da câmera, e os valores não têm unidade métrica nem escala entre imagens. `save=True` grava no disco uma visualização desse mapa com cores; `Results.plot()` renderiza o mapa de profundidade. Veja [predição](/docs/predict) para fontes, streaming e tratamento de resultados.

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
