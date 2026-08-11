---
title: Segmentação panóptica
seo_title: Segmentação panóptica no LibreYOLO
description: >-
  Atribua um único segmento a cada pixel no LibreYOLO: as famílias que atendem à
  tarefa, o formato de dataset COCO-panoptic e as chamadas de predição e
  validação.
lead: >-
  A segmentação panóptica atribui cada pixel a exatamente um segmento, sem
  sobreposição, unificando instâncias de objetos contáveis com regiões amorfas
  de fundo. A chave da tarefa é panoptic.
keywords:
  - segmentação panóptica python
  - panoptic quality PQ
  - things and stuff segmentação
  - formato COCO panoptic
  - mapa de ids de segmento
  - métrica PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -panoptic no nome do arquivo seleciona a tarefa, então
        # nenhum argumento de tarefa é necessário.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ids de segmento
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Um segmento por vez
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # booleano (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Um checkpoint menor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() retorna um dict simples, não um objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Definição

A segmentação panóptica é a união das outras duas tarefas de segmentação. Cada
pixel recebe exatamente um segmento, os segmentos nunca se sobrepõem, e um
segmento é ou um *thing*, uma instância de objeto contável, ou *stuff*, uma
região amorfa como o céu ou a estrada. Isso a torna mais estrita que a
[segmentação de instâncias](/docs/tasks/instance-segmentation), que deixa os
pixels de fundo sem atribuição e permite que as máscaras se sobreponham, e mais
estrita que a [segmentação semântica](/docs/tasks/semantic-segmentation), que
rotula cada pixel mas funde instâncias vizinhas de uma mesma classe.

`panoptic` é a chave canônica da tarefa, e o sufixo `-panoptic` no nome do
arquivo de um checkpoint a seleciona, então `task=` não é necessário ao carregar
pesos publicados.

`predict()` preenche `result.panoptic`. `.data` é um mapa inteiro `(H, W)` de
ids de segmento sobre o canvas da imagem original. `.segments_info` é uma lista
de dicts, um por segmento, cada um carregando ao menos `{"id", "category_id"}`,
onde `id` corresponde a um valor no mapa e `category_id` indexa `result.names`.
`.segment_ids` lista os ids presentes em ordem crescente e `.segment_mask(id)`
retorna a seleção booleana `(H, W)` de um segmento. O id de segmento `0` é o
valor void: pixels sem rótulo, excluídos da métrica e deixados de fora de
`.segment_ids`.

Ser *thing* ou *stuff* é uma propriedade da categoria, não do segmento
individual. Isso vai nos metadados de categoria do conjunto de rótulos, e um
payload de predição pode copiar essa informação para cada segmento como
`"isthing"` por conveniência, mas os metadados de categoria continuam sendo a
fonte autoritativa.

## Modelos

[EoMT](/docs/models/eomt) é a família que atende a esta tarefa através de
`LibreYOLO()`. Roda no pacote base e traz checkpoints panópticos em três
tamanhos, s, b e l, treinados no COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) também emite mapas panópticos.
É um modelo generativo guiado por prompt, com sua própria fábrica, `LibreVLM`, e
seu próprio extra; sem nenhum vocabulário definido, ele recorre às categorias
panópticas do COCO nas quais foi ajustado. Seus pesos são não comerciais. A
latência por imagem é bem mais alta que a de um segmentador dedicado, porque
cada predição é uma decodificação por difusão.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

`conf` filtra a seleção de queries. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Formato do dataset

O LibreYOLO adota o formato COCO-panoptic tal como está, de Kirillov et al.,
CVPR 2019. Não existe um layout panóptico específico do LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Cada imagem é pareada com um PNG RGB na mesma resolução, onde a cor de cada
pixel codifica o id do segmento ao qual ele pertence:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

O id de segmento `0`, o preto RGB, é void: pixels sem rótulo que não premiam nem
penalizam uma predição. Todos os demais pixels pertencem a exatamente um
segmento.

O JSON lista, por imagem, o PNG de ids de segmento e os segmentos dentro dele:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` nomeia o PNG dentro do diretório panóptico, e
`segments_info[].id` corresponde a um valor nesse PNG. `iscrowd` marca regiões
de grupo: elas nunca são contadas como falsos negativos, e uma predição que
cobre a maior parte de uma delas não é um falso positivo. `isthing` fica em
`categories` e nunca em um segmento individual.

O YAML aponta para os dois:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` e `panoptic_dir` aceitam, cada um, um único caminho ou um
mapeamento por split. Os ids de categoria brutos do COCO costumam ser não
contíguos, enquanto os modelos preveem um intervalo contíguo `0..nc-1`, então os
ids são remapeados através de `names` pelo nome da categoria. Uma categoria do
JSON ausente em `names` é um erro, em vez de um descarte silencioso, porque
descartá-la contaria como um falso negativo permanente.

O loader canônico é `libreyolo.data.PanopticDataset`.

## Treinamento

Nenhuma família treina segmentação panóptica no LibreYOLO hoje: o `train()` do
EoMT levanta `NotImplementedError`, então os checkpoints panópticos são usados
tal como publicados.

## Validação

`val()` retorna um dicionário simples de chaves `metrics/`, calculadas na
resolução do ground truth sobre o split indicado por `val` no YAML do dataset.
Um segmento predito e um verdadeiro da mesma categoria casam quando seu IoU
supera 0.5, e essa correspondência é única.

<code-tabs name="val" />

`metrics/PQ` é a Panoptic Quality, o número principal. Dentro de uma categoria,
é o produto de dois fatores. A qualidade de segmentação é o IoU médio sobre os
segmentos casados e diz o quão bem as formas casadas se encaixam. A qualidade de
reconhecimento é `TP / (TP + 0.5 FP + 0.5 FN)`, o F1 da própria correspondência,
e diz quantos segmentos foram encontrados afinal. Os três números passam então
por uma média sobre as categorias que apareceram, e são reportados como
`metrics/PQ`, `metrics/SQ` e `metrics/RQ`, de modo que o PQ reportado é a média
dos produtos por categoria, e não o produto das duas médias reportadas.

`metrics/PQ_things` e `metrics/PQ_stuff` fazem a média desse mesmo PQ por
categoria sobre as categorias thing e as categorias stuff separadamente, e
`metrics/categories` conta as categorias que apareceram e sobre as quais,
portanto, a média foi feita. O dicionário também carrega `fitness`, uma cópia do
valor de PQ.

## Exportação

Checkpoints panópticos não exportam. `export()` levanta `NotImplementedError`
para esta tarefa, porque a saída de máscaras por query ainda não tem um contrato
de exportação em runtime. A tarefa semântica do EoMT exporta; veja
[segmentação semântica](/docs/tasks/semantic-segmentation) e
[exportação e deploy](/docs/export).
