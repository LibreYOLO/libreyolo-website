---
title: Formatos de dataset
seo_title: Formatos de dataset da LibreYOLO para cada tarefa
description: >-
  O contrato de arquivos de dataset por tarefa canônica: chaves YAML, estruturas
  de pastas, linhas de rótulos, convenções de máscaras e mapas, e o loader que
  lê cada um.
lead: >-
  Esta página espelha o contrato de arquivos de dataset do próprio
  docs/dataset_schema.md da biblioteca. Ela cobre as chaves YAML e o layout em
  disco que cada tarefa canônica espera.
keywords:
  - formato dataset libreyolo
  - formato de rótulo yolo txt
  - data.yaml
  - dataset de máscaras de segmentação
  - formato coco panoptic
  - dataset de profundidade
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Espelha docs/dataset_schema.md do repositório libreyolo na v1.5.0, com os
  nomes dos loaders conferidos contra libreyolo/data/.
snippets:
  usage:
    - label: Fazer parse de uma linha de rótulo de detecção
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, normalizado para [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) em pixels

        print(row)
source_hash: a8282c079624044d
---

## YAML comum

Vale para `detect`, `segment`, `pose` e `obb`.

| Chave | Obrigatória | Significado |
|---|---|---|
| `path` | | Raiz do dataset |
| `train` | Para treinamento | Imagens de treinamento |
| `val` | Para validação | Imagens de validação |
| `test` | | Imagens de teste |
| `names` | Sim | Lista de classes, ou um mapeamento com chaves inteiras |
| `nc` | | Contagem de classes; deve bater com `names` quando presente |
| `download` | | Instruções de download; scripts Python exigem opt-in explícito |
| `annotations` | | Split para o arquivo COCO JSON nativo, para detect, segment e obb |

`train`, `val` e `test` podem ser diretórios de imagens, arquivos `.txt` com
listas de imagens, ou listas desses. Os caminhos dos rótulos seguem uma única
substituição:

```text
images/.../image.jpg -> labels/.../image.txt
```

Para um dataset COCO JSON nativo, `annotations` mapeia um split para o seu
arquivo JSON e o caminho do split dá a raiz das imagens:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Quando `names` está presente, os nomes de categoria do COCO JSON nativo devem
bater com os nomes de classe do YAML, e esses nomes definem os IDs de rótulo do
modelo. Sem `names`, os IDs de categoria do COCO são ordenados e mapeados de
forma densa para `0..N-1`.

Um YAML de dataset não carrega uma chave `task`. A seleção explícita de modelo e
tarefa vence.

Regras comuns a todo arquivo de rótulos em texto:

- um arquivo de rótulos `.txt` por imagem;
- um arquivo de rótulos ausente ou vazio significa nenhum objeto;
- `class_id` é um inteiro em `0..nc-1`;
- as coordenadas são floats normalizados e finitos em `[0, 1]`;
- as coordenadas são relativas à largura e à altura originais da imagem;
- as linhas não carregam confiança nem ID de tracking.

<code-tabs name="usage" />

## detect

Exatamente cinco campos por linha:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` é uma caixa alinhada aos eixos e normalizada, e `w` e `h` devem ser
positivos.

## segment

Uma linha de polígono:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` é pelo menos 3, a contagem de coordenadas depois de `class_id` deve ser par,
e o polígono não pode ser degenerado. Uma linha de detecção com cinco campos
também é aceita e representa um segmento retangular.

## pose

O YAML acrescenta `kpt_shape`, que é obrigatório e é `[K, 2]` ou `[K, 3]`, e o
opcional `flip_idx`, uma permutação de inteiros de `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

A contagem de campos é exatamente `5 + K * D`, onde `D` é o segundo valor de
`kpt_shape`. As coordenadas dos keypoints são normalizadas. A visibilidade `v`,
quando presente, é `0`, `1` ou `2`.

## obb

Exatamente nove campos:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Os quatro pontos são coordenadas de imagem normalizadas em `[0, 1]` e formam um
retângulo orientado não degenerado. Nenhum ângulo é armazenado no arquivo de
rótulos.

O parser canônico é estrito por padrão e rejeita coordenadas fora do intervalo.
A ingestão de dataset e de validação pode fazer clipping das coordenadas para
`[0, 1]` em rótulos de borda de recorte que de resto seriam válidos, e ainda
assim rejeita caixas degeneradas. O parsing leva a tarefa em conta: nove campos
significam `obb` apenas no modo `obb`, enquanto no modo `segment` eles podem ser
um polígono de quatro pontos.

Internamente, os cantos normalizados são convertidos para o `xywhr` canônico,
com o ângulo em radianos representando a rotação do lado da largura em torno do
centro da caixa. Os resultados públicos expõem as detecções OBB como linhas
`xywhr, conf, cls`.

O carregamento de OBB a partir de COCO JSON nativo aceita anotações nesta ordem
de prioridade: `obb` como oito cantos em espaço de pixels; `obb` como
`[cx, cy, w, h, angle]` com o ângulo em radianos; um polígono ou RLE de
`segmentation` do COCO, reajustado para um retângulo de área mínima; e um `bbox`
do COCO, lido como alinhado aos eixos e canonicalizado.

Mosaic e mixup ficam desativados no treinamento de OBB até que exista um data
augmentation de OBB ciente dos cantos.

O parser canônico de linhas é `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Cada imagem é pareada com uma máscara densa de canal único em um formato sem
perdas, tipicamente PNG, em vez de um arquivo `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

A máscara é de canal único, e PNGs em modo paleta são lidos como índices de
paleta. Cada valor de pixel é um ID de classe em `0..nc-1`, o valor de pixel
`255` significa ignorar e é excluído da loss e das métricas, e a resolução da
máscara deve ser igual à resolução da imagem.

Duas chaves YAML opcionais se somam ao contrato comum. `masks_dir` é o nome do
diretório de máscaras que substitui `images` em cada caminho de imagem, com
padrão `masks`. `label_mapping` é um remapeamento `{source_id: train_id}`
aplicado aos valores de pixel da máscara no momento do carregamento, onde
valores de origem não mapeados viram ignorar e os train IDs devem cair em
`0..nc-1`.

Quando `masks_dir` é omitido, as máscaras são rasterizadas no momento do
carregamento a partir dos rótulos de polígono de `segment`, resolvidos pela
convenção de `images` para `labels`, e uma classe `background` é acrescentada
depois das classes de objetos, então `nc` cresce em um.

Loader canônico: `libreyolo.data.SemanticDataset`.

## panoptic

A LibreYOLO adota o formato COCO-panoptic literalmente (Kirillov et al., CVPR
2019). Não existe um formato panóptico específico da LibreYOLO.

Um PNG RGB por imagem, na resolução da imagem, codifica na sua cor o ID de
segmento de cada pixel:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Todo pixel pertence a exatamente um segmento e os segmentos nunca se sobrepõem.
O ID de segmento `0`, preto RGB, é void: pixels não rotulados excluídos da
métrica.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` nomeia o PNG de IDs de segmento dentro de
`panoptic_dir`, e `segments_info[].id` corresponde a um valor nesse PNG.
`iscrowd` marca regiões de grupo: elas nunca são falsos negativos, e uma
predição que cobre a maior parte de uma delas não é um falso positivo.

Thing versus stuff é uma propriedade por categoria. `isthing` fica em
`categories`, nunca em `segments_info`.

Os valores de `category_id` do COCO-panoptic são os IDs brutos do dataset e
costumam ser não contíguos. Os modelos predizem `0..nc-1` contíguos, então os
IDs brutos são remapeados pelo `names` do YAML através do nome da categoria, a
mesma regra que o loader de detecção do COCO JSON nativo segue. Uma categoria do
JSON ausente de `names` é um erro, e não um descarte silencioso, porque de outro
modo ela pontuaria como um falso negativo permanente.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` e `panoptic_dir` aceitam ou um caminho único ou um mapeamento por
split.

A validação reporta o Panoptic Quality, calculado na resolução do ground truth e
mediado sobre as categorias que aparecem, e depois dividido em `PQ_things` e
`PQ_stuff`. O pareamento é único: um segmento predito e um segmento do ground
truth da mesma categoria casam quando o IoU está acima de 0.5.

Loader canônico: `libreyolo.data.PanopticDataset`.

## depth

Cada imagem é pareada com um mapa de profundidade denso de canal único:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

O mapa é um PNG ou TIF de canal único, ou um arquivo `.npy`, na resolução da
imagem. Os valores são profundidade pura em uma unidade consistente dentro do
dataset. Valores zero, negativos, NaN e infinitos marcam pixels inválidos e são
excluídos da loss e das métricas.

| Chave | Padrão | Significado |
|---|---|---|
| `depths_dir` | `depths` | Diretório de profundidade que substitui `images` |
| `depth_stem_suffix` | | Sufixo acrescentado ao stem da imagem; quando omitido, são testados tanto o mesmo stem quanto um sufixo `_depth` |
| `depth_mask_suffix` | `_mask` | Sufixo para uma máscara de validade; valores de máscara iguais ou abaixo de zero, NaN e infinitos invalidam o pixel de profundidade |
| `depth_scale` | `256.0` | Divisor para mapas de profundidade de tipo inteiro, a convenção comum de PNG de 16 bits |

Mapas `.npy` de ponto flutuante são usados como estão e não aplicam
`depth_scale`.

Loader canônico: `libreyolo.data.DepthDataset`.

## edge

Cada imagem RGB é pareada com um mapa sem perdas de canal único e mesmo stem, e
com uma máscara de validade opcional:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

O mapa é um PNG ou TIF de canal único, não uma visualização RGB, na resolução da
imagem. Mapas inteiros são divididos pelo máximo do seu dtype; mapas de ponto
flutuante já devem ser finitos e estar em `[0, 1]`. `0` significa não borda e `1`
significa borda. Os pixels da máscara opcional são válidos quando diferentes de
zero. O redimensionamento usa interpolação por vizinho mais próximo para alvos e
máscaras, e os pixels de padding são inválidos e não contribuem para a validação.

| Chave | Padrão | Significado |
|---|---|---|
| `edges_dir` | `edges` | Diretório de mapas de borda que substitui `images` |
| `edge_stem_suffix` | | Sufixo acrescentado aos stems das imagens |
| `edge_extension` | `.png` | Extensão sem perdas do alvo |
| `edge_invert` | | Defina como true quando os mapas de origem armazenam bordas pretas sobre branco |
| `masks_dir` | `masks` | Diretório opcional de máscaras de validade |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

A validação afina as predições contínuas com non-maximum suppression de
gradiente em quatro direções e reporta as F-measures ODS e OIS sobre uma
varredura de limiar configurável. Os pixels preditos e os do ground truth são
pareados um a um dentro de `edge_max_dist * image_diagonal`, com uma tolerância
normalizada padrão de `0.0075`.

Loader canônico: `libreyolo.data.EdgeDataset`. O loader é apenas de formato: ele
não baixa nem redistribui dados de benchmark.

## normal

Cada imagem é pareada com um PNG de 16 bits e três canais com o mesmo stem, mais
uma máscara de validade opcional de mesmo stem:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

O PNG é exatamente `uint16` de três canais, com os canais armazenados como RGB,
na resolução da imagem. Decodifique com `n = png / 65535 * 2 - 1` e depois
renormalize cada vetor. Os vetores decodificados usam o referencial de câmera do
OpenCV, `+x` à direita, `+y` para baixo, `+z` para dentro da cena, e apontam para
a câmera. A máscara opcional é um PNG de canal único onde diferente de zero
significa válido; sem máscara, todo vetor decodificado finito e diferente de zero
é válido. Pixels de alvo inválidos e de padding são representados internamente
por `(0, 0, 0)`. O redimensionamento interpola os três componentes bilinearmente
e depois renormaliza, as máscaras de validade usam interpolação por vizinho mais
próximo, e um flip horizontal também nega o componente x.

| Chave | Padrão | Significado |
|---|---|---|
| `normals_dir` | `normals` | Diretório de mapas de normais que substitui `images` |
| `masks_dir` | `masks` | Diretório opcional de máscaras de validade |

A validação reporta o erro angular médio e mediano em graus e a porcentagem de
pixels válidos dentro de 11.25, 22.5 e 30 graus.

Loader canônico: `libreyolo.data.NormalDataset`.

## restore

Cada imagem de entrada degradada é pareada com um alvo RGB limpo:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

A entrada e o alvo são arquivos de imagem compatíveis com RGB e suas resoluções
devem coincidir exatamente. A validação mantém a resolução nativa e aplica apenas
o padding necessário para empilhar um batch, e as métricas são calculadas sobre o
canvas original da imagem. O treinamento aplica um crop e um flip horizontal
acoplados ao par de entrada e alvo.

| Chave | Padrão | Significado |
|---|---|---|
| `input_dir` | `inputs` | Diretório de entradas degradadas usado nos caminhos dos splits |
| `target_dir` | `targets` | Diretório de alvos limpos que substitui `input_dir` |
| `target_stem_suffix` | | Sufixo acrescentado ao stem da entrada antes da busca pelo alvo |
| `target_stem_suffixes` | | Forma em lista de `target_stem_suffix` |
| `degradation` | | Rótulo de metadados como `deblur` ou `denoise` |
| `dataset` | | Rótulo de dataset ou de proveniência |

Os campos YAML parecidos com classes são placeholders de schema: use `nc: 1` e
`names: {0: image}`. Os modelos de restore expõem `Results.restored`, não
detecções.

Loader canônico: `libreyolo.data.RestoreDataset`.

## matte

Cada imagem RGB é pareada com um matte de ground truth de canal único que
compartilha o mesmo stem, onde 0 é fundo e 255 é primeiro plano:

```text
images/subject.jpg -> mattes/subject.png
```

Dois layouts são aceitos. Uma raiz de diretório contendo `images/` e um diretório
de mattes, detectado automaticamente entre `mattes/`, `matte/`, `gt/`, `masks/`,
`mask/` e `alpha/`, passada como `data=`. Ou um YAML com `path` mais
`val_images` e `val_mattes` por split, e opcionalmente `train_images` e
`train_mattes`, cada um relativo a `path` ou absoluto.

O matte é em escala de cinza e é lido como opacidade em `[0, 1]`, e é
redimensionado para o canvas da predição com interpolação bilinear quando os
shapes diferem. As métricas são MAE e S-measure (Fan et al., ICCV 2017) sobre o
canvas original da imagem, com o S-measure como fitness do melhor checkpoint.

Os campos YAML parecidos com classes são placeholders de schema: use `nc: 1` e
`names: {0: matte}`. Os modelos de matte expõem `Results.matte`.

A validação é somente de inferência nesta versão. Resolvedor canônico de pares:
`libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Os rótulos são um arquivo JSONL por split, um objeto JSON por imagem:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` é um quadrilátero de quatro pontos em coordenadas absolutas de pixel,
ordenado como superior esquerdo, superior direito, inferior direito, inferior
esquerdo. Regiões com texto ilegível usam `"text": "###"`, a convenção
do-not-care do ICDAR: elas são excluídas da pontuação de reconhecimento, e as
predições que se sobrepõem a elas são ignoradas em vez de penalizadas no
pareamento de detecção.

As métricas são o hmean de detecção com pareamento um a um de polígonos acima de
IoU 0.5, o F1 end-to-end, que exige tanto IoU acima de 0.5 quanto uma transcrição
exata após normalização NFKC e remoção de espaços em branco, sensível a
maiúsculas e minúsculas, e o 1-NED sobre os pares pareados. O fitness do melhor
checkpoint é o F1 end-to-end.

Dois layouts são aceitos: uma raiz de diretório contendo `images/<split>/` e
`labels/<split>.jsonl`, passada como `data=`, ou um YAML com `path` mais os nomes
opcionais de diretório `images` e `labels`.

Os campos YAML parecidos com classes são placeholders de schema: use `nc: 1` e
`names: {0: text}`. Os modelos de OCR expõem `Results.ocr`.

A validação é somente de inferência nesta versão. Resolvedor canônico de
amostras: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Uma árvore de diretórios no estilo ImageFolder, não arquivos de rótulos:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` é obrigatório para o treinamento e define o mapeamento de classe para
índice pelo nome de pasta ordenado. `val/` é obrigatório para a validação.
`test/` pode estar presente, mas os comandos padrão de train e val não o usam. Os
splits que não são de treinamento devem conter os mesmos nomes de pasta de classe
do conjunto de classes esperado do train ou do checkpoint. As extensões de imagem
suportadas são definidas em
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze e point

Nenhum contrato de arquivos de dataset para treinamento ou validação está
implementado para `gaze`.

`point` é uma tarefa de saída de modelo, e não um schema de rótulos de dataset.
As famílias de point podem adaptar rótulos existentes internamente, por exemplo
derivando centros de objetos a partir de linhas de caixa, mas não há um formato
de rótulo em texto exclusivo para point.
