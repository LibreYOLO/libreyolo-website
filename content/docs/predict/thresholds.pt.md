---
title: Limiares e filtragem
seo_title: 'conf, iou e max_det no LibreYOLO'
description: >-
  O que conf, iou, max_det, classes e agnostic_nms fazem na hora da predição, e
  quais famílias ignoram iou porque não rodam NMS próprio.
lead: >-
  Quatro argumentos decidem quais predições sobrevivem: conf, iou, max_det e
  classes. Só dois deles se aplicam a todas as famílias, porque um preditor de
  conjuntos decodifica um conjunto fixo de queries e nunca roda NMS.
keywords:
  - limiar de confiança yolo
  - conf yolo python
  - limiar iou nms
  - max_det yolo
  - filtrar classes detecção python
  - agnostic nms
  - detr sem nms
  - filtragem por classes inferência
last_verified: 1.6.0
verification: >-
  Valores padrão retirados de InferenceRunner.__call__ em
  libreyolo/models/base/inference.py. Comportamento do NMS por família lido de
  todos os módulos em libreyolo/postprocess/ e conferido contra
  _is_nms_free_family em libreyolo/backends/base.py. Filtragem por classes de
  InferenceRunner._apply_classes_filter e _wrap_results. agnostic_nms a partir
  de check_agnostic_nms e postprocess_max_det em libreyolo/utils/predict_args.py
  e do passo agnóstico em InferenceRunner._wrap_results. Tratamento de
  vocabulário aberto a partir de NMS_THRESHOLD em
  libreyolo/models/openvocab/base.py. Padrões de validação de BaseModel.val.
snippets:
  basic:
    - label: Os quatro argumentos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # mantém as predições com esta pontuação ou acima
            iou=0.45,       # limiar de sobreposição do NMS, onde o NMS roda
            max_det=300,    # teto por imagem
            classes=None,   # ou uma lista de ids de classe
        )
        print(len(result.boxes))
    - label: Varredura de conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtrar por classes específicas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Os ids de classe indexam model.names. No COCO, 0 é person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Descobrir o id de um nome
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou em uma família que não roda NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O RF-DETR decodifica um conjunto fixo de queries, então iou não muda
        nada aqui.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Mesma contagem nos dois casos. conf e max_det são os controles que
        funcionam.

        print(len(loose.boxes), len(tight.boxes))
  agnostic:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        per_class = model(SAMPLE_IMAGE)
        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)

        # Caixas sobrepostas de classes diferentes colapsam na de maior score.
        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
---

## Os quatro argumentos

| Argumento | Padrão | Aplica-se a |
|---|---|---|
| `conf` | `0.25` | Todas as famílias |
| `iou` | `0.45` | Famílias que rodam supressão de não máximos |
| `max_det` | `300` | Todas as famílias |
| `classes` | `None` | Todas as famílias |

<code-tabs name="basic" />

Dois deles são universais e dois não são, e essa é a coisa mais útil de saber
antes de ajustar qualquer outra.

A validação usa padrões diferentes de propósito: `val()` roda com `conf=0.001` e
`iou=0.6`, porque a precisão média é calculada sobre uma curva de
precisão-recall completa e um corte em 0.25 a truncaria.

## conf

`conf` é a pontuação abaixo da qual uma predição é descartada. Ela se aplica a
todas as famílias, inclusive as que nunca rodam NMS, e é o primeiro controle a
buscar quando há detecções demais ou de menos.

O padrão de `0.25` serve para olhar imagens. Alimentar um sistema a jusante
geralmente pede um valor mais alto; medir acurácia pede um valor bem mais baixo.

A validação de detecção expõe `metrics/best_conf` e limiares por classe obtidos de F1 em IoU 0.50. Use-os como pontos de partida medidos para os limiares de deploy; veja [validação](/docs/train/validation). A segmentação não expõe essas chaves.

## iou

`iou` é a sobreposição acima da qual a supressão de não máximos remove, entre
duas caixas da mesma classe, a de menor pontuação. Só significa alguma coisa se
a família rodar supressão.

Um preditor de conjuntos decodifica um número fixo de queries e fica com as de
maior pontuação. As duplicatas são suprimidas dentro da arquitetura durante o
treinamento, não por uma etapa de pós-processamento, então não há limiar para
girar. Estas famílias aceitam `iou` por paridade de API e o ignoram:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR e a cabeça end-to-end do
YOLOv9. Variantes construídas sobre esses decodificadores herdam o
comportamento.

<code-tabs name="nmsfree" />

A maioria delas diz isso nas docstrings de pós-processamento, mas nenhum aviso é
emitido em tempo de execução, então uma varredura de `iou` no RF-DETR produz uma
linha reta em vez de um erro. Faster R-CNN e Mask R-CNN são um caso um pouco
diferente: os dois já rodaram NMS dentro do modelo, com um limiar fixo lá em
cima que `iou` não tem como mudar de forma suportada.

Estas famílias usam sim: YOLOv1 até YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet e SSD.

Três opções de tempo de predição fazem `iou` importar até para um preditor de
conjuntos, porque as três agem sobre as caixas depois que o modelo terminou:

- `tiling=True` reconcilia tiles sobrepostos com NMS por classe em `iou`
- `augment=True` mescla visões espelhadas com NMS por classe em `iou`
- `agnostic_nms=True` suprime caixas sobrepostas entre classes em `iou`,
  descrito [abaixo](#agnostic_nms)

As duas primeiras são cobertas em [Desempenho de inferência](/docs/predict/performance).

Detectores de vocabulário aberto têm a própria regra. Uma família cujo
processador roda NMS declara o próprio limiar padrão e respeita `iou`, que é o
caso do OMDet-Turbo. Famílias que não suprimem nada, Grounding DINO, OWLv2 e
OV-DEIM, emitem um aviso quando `iou` é passado. Esse aviso é o único do gênero
na biblioteca.

## max_det

`max_det` limita quantas predições voltam para uma imagem. Ele se aplica em
todo lugar, mas por mecanismos diferentes: uma família com NMS trunca depois da
supressão, um preditor de conjuntos usa o valor como o tamanho da sua seleção
top-k.

Algumas famílias limitam abaixo do que você pedir, porque a configuração de
referência original delas faz isso. O SSD limita em 200, e o FCOS no próprio limite de
detecções por imagem. Aumentar `max_det` acima desses valores não tem efeito.

`max_det` é aplicado de forma centralizada e não por família em dois casos: a
inferência por blocos (tiles), onde a lista mesclada é truncada depois que os
tiles são reconciliados, e as chamadas com `classes` ou `agnostic_nms=True`, em
que o corte vem depois do filtro ou da supressão.

## Filtragem por classes

<code-tabs name="classes" />

`classes` recebe uma lista de ids de classe e mantém apenas as predições cuja
classe está nela. Os ids indexam `result.names`, e a forma mais segura de obter
um é ler `names` de um resultado em vez de supor a ordenação de um dataset.

A filtragem acontece de forma centralizada, depois do pós-processamento de cada
família, no único funil pelo qual passa todo caminho de predição. Isso tem duas
consequências que vale conhecer. Funciona em todas as famílias, inclusive nas
que não têm NMS. E também filtra os payloads alinhados com as caixas, então
máscaras, keypoints e caixas orientadas são cortados junto em vez de ficarem
desalinhados.

Na linha de comando, `classes` aceita um inteiro solto, uma lista ou uma string
separada por vírgulas:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filtrar não é acurácia de graça. O modelo continua gastando seu orçamento
prevendo classes que você depois descarta. Com `classes` definido, a família
mantém um orçamento de candidatos de 300 caixas, ou `max_det` quando ele é
maior, e o corte para `max_det` vem depois do filtro, então um `max_det` pequeno
não espreme a sua classe para fora. Os backends exportados usam o mesmo
orçamento, inclusive os engines TensorRT em batch. Uma imagem com mais de 300
caixas de maior score de classes indesejadas ainda pode encher o orçamento antes
de chegar na sua classe; aumente `max_det` acima de 300 se isso acontecer.

## agnostic_nms

O NMS de toda família leva a classe em conta, então, por padrão, duas caixas
sobrepostas de classes diferentes sobrevivem as duas, em qualquer `iou`.
`agnostic_nms=True` acrescenta um passo agnóstico a classes: entre as caixas que
se sobrepõem acima de `iou`, só a de maior score sobrevive, quaisquer que sejam
as classes.

<code-tabs name="agnostic" />

O passo roda sobre as detecções já prontas da família, depois do filtro de
`classes` e antes do corte para `max_det`. Por isso ele se comporta igual em
preditores de conjuntos que não rodam NMS próprio, em backends exportados, e com
`augment=True` e `tiling=True`. Máscaras e keypoints são descartados junto com a
sua caixa, e caixas orientadas são comparadas por IoU rotacionado. Ele cobre as
tarefas que retornam caixas: detect, segment, pose e obb.

A família mantém um orçamento de candidatos de 300 caixas, ou `max_det` quando
ele é maior, para o passo trabalhar. Com `max_det` abaixo de 300, candidatos de
menor score reabastecem as vagas que a supressão libera. Com 300 ou mais, o que
inclui o padrão, essas vagas ficam vazias.

Onde ele não se aplica:

- No caminho padrão do predict, uma tarefa sem caixas levanta `ValueError`.
- `LibreEnsemble` e [HRNet](/docs/models/hrnet) levantam `NotImplementedError`.
- Os modelos `LibreSAM` e os [detectores 3D](/docs/tasks/3d-object-detection),
  WildDet3D, 3D-MOOD, FCOS3D e DetAny3D, não recebem esse argumento e levantam
  `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body) e
  [LibreFaceRec](/docs/models/librefacerec) rodam o próprio caminho de predict e
  o ignoram.

`val(agnostic_nms=True)` aplica o mesmo passo à validação de detect e segment;
veja [Validação e métricas](/docs/train/validation).

## O que o predict rejeita

Dois argumentos levantam erro em vez de avisar: `visualize` e `embed` levantam
`NotImplementedError`. Para embeddings, carregue o modelo com `task="embed"` e
chame `predict` ou `embed` normalmente. `agnostic_nms=True` levanta erro onde não
se aplica, como listado acima.

Qualquer coisa não reconhecida levanta `TypeError` nomeando as opções
suportadas, então um erro de digitação falha na hora em vez de ser ignorado em
silêncio.

Estes são aceitos, geram aviso e são descartados: `boxes`, `dnn`, `half`,
`line_width`, `retina_masks`, `show_conf`, `show_labels` e `verbose`.
