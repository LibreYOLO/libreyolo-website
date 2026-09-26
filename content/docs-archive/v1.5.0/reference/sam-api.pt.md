---
title: API de segmentação com prompts
seo_title: 'API do LibreSAM: prompts, aliases e assinaturas'
description: >-
  A fábrica LibreSAM, seus aliases de tamanho, os tipos de prompt de ponto, de
  caixa e de conceito em texto, o ciclo de vida do set_image que codifica uma
  vez só, e o que o nível não suporta.
lead: >-
  O LibreSAM é a fábrica da segmentação com prompts. Um forward pass precisa de
  um prompt por imagem informado na hora da chamada, então o nível tem a sua
  própria superfície de predict em vez de passar pelo runner de inferência sem
  prompts.
keywords:
  - LibreSAM
  - segmentação com prompts python
  - prompt de ponto SAM
  - prompt de caixa SAM
  - set_image
  - segmentar tudo SAM
  - libreyolo extra sam
last_verified: 1.5.0
verification: >-
  Aliases, tamanhos e repositórios da fábrica lidos de
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py e libreyolo/models/picosam3/model.py.
  Contrato de prompts e valores padrão lidos de libreyolo/models/sam/base.py.
  Intenção de projeto vinda de docs/adr/0007-libresam-contract.md, tudo na
  v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompts de ponto e de caixa
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Codifique uma vez, mande vários prompts'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Instalação

O nível precisa do extra `sam`.

<code-tabs name="install" />

## A fábrica

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` é um alias de tamanho, e não um caminho. `**kwargs` chega ao construtor
da família, que aceita `device` e `multimask`. Um alias desconhecido levanta
`ValueError`, e a mensagem lista todos os aliases conhecidos.

<code-tabs name="usage" />

## Aliases

| Família | Aliases | Tamanhos | Pesos |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, e as formas curtas `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

O padrão é `base`. SAM-1, SAM-2, EdgeTAM e MobileSAM rodam em uma tela nominal
de 1024 pixels, o SAM 3 em 1008 e o PicoSAM3 em 96.

Os pesos do SAM 3 são restritos. Eles são baixados de `facebook/sam3` sob a SAM
License própria da Meta, que não é nem MIT nem Apache-2.0 e não é redistribuída
pelo LibreYOLO. Aceite os termos na página do repositório e faça login no
Hugging Face antes de carregar; o loader registra o aviso antes de tudo.

As classes de família também são exportadas, então `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` e `LibrePicoSAM3` podem ser
construídos diretamente com `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argumento | Padrão | Significado |
|---|---|---|
| `source` | `None` | Imagem a segmentar; `None` reaproveita a imagem guardada em cache por `set_image()` |
| `points` | `None` | Prompt de ponto em coordenadas de pixel |
| `bboxes` | `None` | Prompt de caixa como `[x1, y1, x2, y2]`, ou uma lista deles para uma máscara por caixa |
| `labels` | `None` | Labels dos pontos, `1` positivo e `0` negativo, no formato que casa com `points`; todos positivos quando omitido |
| `masks` | `None` | Reservado; passar uma levanta `NotImplementedError` |
| `text` | `None` | Prompt de conceito; só no SAM 3 |
| `conf` | `None` | Piso do IoU de máscara predito |
| `multimask` | `None` | Devolve todas as máscaras de ambiguidade por prompt; o padrão é a configuração da construção |
| `max_det` | `300` | Teto de máscaras devolvidas |
| `device` | `None` | Move o modelo para esta chamada e as seguintes, invalidando os embeddings em cache |
| `color_format` | `"auto"` | Dica de formato de cor para arrays em memória |
| `points_per_side` | `None` | Densidade da grade do segmentar tudo; o padrão é 32 |

O retorno é um `Results` comum carregando `masks`, mais `boxes` justas
derivadas dessas máscaras, com a classe `0` chamada `"object"`.

## Formatos de prompt

`points` aceita as formas aninhadas `[x, y]` para um objeto, `[[x, y], ...]`
para N objetos e `[[[x, y], ...], ...]` para pontos agrupados por objeto. Arrays
numpy funcionam em todo lugar em que uma lista funciona. As coordenadas são
pixels simples na imagem de origem.

Omitir todos os prompts espaciais roda o segmentar tudo, um gerador automático
de máscaras em grade com um limiar de IoU predito e deduplicação por IoU de
caixa. O `points_per_side` padrão de 32 roda por volta de 1024 passagens do
decoder, o que é lento na CPU; baixe esse valor para uso interativo. O gerador
não faz filtragem por stability score, multi-crop nem deduplicação por IoU de
máscara, então ele é uma aproximação do caminho com prompt, e não um
equivalente exato.

## Confiança

`conf` filtra pelo IoU de máscara predito, que é uma pontuação de qualidade de
máscara e não uma confiança de detecção. `None` mantém todas as máscaras no
caminho com prompt e aplica o limiar de grade da família no segmentar tudo.
`0.0` desliga a filtragem nos dois modos.

No caminho de texto do SAM 3, `conf` passa a ser a pontuação de detecção da
Promptable Concept Segmentation. Ali, `None` significa o limiar padrão de 0.3, e
`0.0` mantém todos os candidatos.

## Prompts de texto

`text=` é só do SAM 3; todas as famílias de prompt espacial levantam
`NotImplementedError` para ele. Texto é mutuamente exclusivo com pontos e
caixas. O `names` devolvido associa a classe `0` ao conceito pedido. Uma chamada
de texto com `source=None` recodifica a imagem em cache, porque o tracker e o
encoder de conceito não compartilham cache.

O argumento `exemplars=` está reservado para uma futura extensão de exemplares
em imagem e não está implementado.

## O ciclo de vida de codificar uma vez

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` roda o encoder pesado de imagem uma única vez e guarda os embeddings
em cache, então todo `predict()` posterior com `source=None` sai barato. Os dois
métodos devolvem o modelo, de modo que as chamadas podem ser encadeadas. Passar
`device=` para o `predict` move o modelo e invalida o cache.

## PicoSAM3

O PicoSAM3 aceita apenas `bboxes=`. Prompts de ponto, de texto, de máscara,
multimask e de segmentar tudo levantam erro. A caixa é expandida em 10 por cento
e passa por uma rede de ROI de 96 pixels, e o PicoSAM3 é a única família do
nível que exporta, e só para ONNX.

## Sem suporte

`train()`, `val()` e `track()` levantam `NotImplementedError` em todas as
famílias do nível. Máscaras guiadas por prompt não têm um conjunto fixo de
classes contra o qual pontuar, então mAP não tem significado aqui. `export()`
levanta erro para SAM-1, SAM-2, SAM 3, EdgeTAM e MobileSAM.

Os caminhos de vídeo e de memória do SAM-2, do SAM 3 e do EdgeTAM estão fora do
escopo desta versão, assim como os exemplares em imagem e os prompts de máscara
do SAM 3.
