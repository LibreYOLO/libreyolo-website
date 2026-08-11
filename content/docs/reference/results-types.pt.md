---
title: Tipos de Results
seo_title: Referência do objeto Results do LibreYOLO
description: >-
  Todos os payloads que um objeto Results do LibreYOLO pode carregar, um slot
  por formato de tarefa: boxes, masks, keypoints, probs, obb, depth, ocr,
  embeddings e mais dez.
lead: >-
  Results é o único tipo de retorno por imagem de todos os modelos do LibreYOLO.
  Ele carrega dezoito slots de payload opcionais, um por formato de tarefa, e
  preenche apenas os que o modelo produziu.
keywords:
  - objeto Results libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - pegar coordenadas de bounding box python
  - resultado de detecção para json python
last_verified: 1.5.0
verification: >-
  Nomes de slot, formatos, propriedades e valores padrão lidos de
  libreyolo/utils/results.py na v1.5.0. Semântica citada das docstrings das
  classes de payload.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Todo payload se move junto.
        result = result.cpu().numpy()

        # As linhas, como dicts simples, depois como JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## O objeto Results

Um `Results` descreve uma imagem. Uma fonte de imagem única retorna um deles,
uma fonte em lista ou um diretório retorna uma lista, e `stream=True` retorna
um gerador que os produz.

| Atributo | Tipo | Significado |
|---|---|---|
| `orig_shape` | `(int, int)` | Altura e largura da imagem original |
| `path` | `str` | Caminho da fonte quando a entrada veio do disco |
| `names` | `dict[int, str]` | Índice da classe para o nome da classe |
| `speed` | `dict[str, float]` | Milissegundos por etapa |
| `track_id` | tensor | IDs de rastreio quando o resultado veio de `track()` |
| `frame_idx` | `int` | Índice do quadro para fontes de vídeo e de stream |
| `restore_scale` | `int` | Fator de aumento de escala entre saída e entrada de um resultado de restore; `1` em todo o resto |

<code-tabs name="usage" />

## Slots de payload

Cada slot é `None` a menos que o modelo o tenha produzido. O slot que uma
família preenche é decidido pela tarefa dela.

| Slot | Classe | Tarefa |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, com uma galeria |
| `meshes` | `Meshes` | mesh |

`result.normals` é um alias de leitura e escrita para `result.normal_map`.

Mais de um slot pode estar preenchido ao mesmo tempo. Um modelo de segmentação
preenche tanto `boxes` quanto `masks`; um modelo de gaze preenche `boxes` com
os boxes dos rostos e `gaze` com os ângulos; um modelo de mesh preenche `boxes`
com os boxes das pessoas e `meshes` alinhado linha a linha com eles.

## Boxes

Boxes de detecção para uma imagem.

| Membro | Retorna |
|---|---|
| `xyxy` | Coordenadas dos cantos em pixels da imagem original |
| `xywh` | Centro e tamanho em pixels |
| `xyxyn` | Cantos normalizados para `[0, 1]` |
| `xywhn` | Centro e tamanho normalizados para `[0, 1]` |
| `conf` | Confiança por box |
| `cls` | Índice da classe por box |
| `id` | ID de rastreio por box, ou `None` |
| `is_track` | `True` quando há IDs de rastreio |
| `data` | O tensor empacotado |

`with_id(id)` e `with_orig_shape(orig_shape)` retornam um novo `Boxes` com esse
campo substituído.

## Masks

Máscaras de instância para uma imagem. `data` é o tensor de máscaras; `xy`
retorna os contornos por instância em pixels e `xyn` os retorna normalizados.

## Keypoints

Keypoints de pose, alinhados linha a linha com `boxes`. `xy` é o par de
coordenadas por keypoint e `xyn` o par normalizado. `conf` é o terceiro canal
quando os dados carregam um, caso contrário `None`. `has_visible` é um array
booleano, verdadeiro onde `conf > 0`, e todo verdadeiro quando não há canal de
confiança.

## Points

Localização de pontos para uma imagem. `data` tem formato `(N, 4)` com as
linhas `x, y, class, confidence`. As coordenadas são pixels absolutos; `xy`,
`cls` e `conf` separam as colunas, e `xyn` normaliza as coordenadas.

## Probs

Scores de classificação. `top1` é o índice vencedor, `top5` os cinco melhores
índices, e `top1conf` e `top5conf` os scores deles.

## OBB

Caixas orientadas. `data` guarda 7 ou 8 valores por linha: `xywhr`, um ID de
rastreio opcional, depois confiança e classe.

| Membro | Retorna |
|---|---|
| `xywhr` | Centro, tamanho e rotação em radianos |
| `xyxyxyxy` | Os quatro cantos em pixels |
| `xyxyxyxyn` | Os quatro cantos normalizados |
| `xyxy` | Envoltória alinhada aos eixos em pixels |
| `conf`, `cls`, `id`, `is_track` | Como em `Boxes` |

## Gaze

Ângulos de gaze por rosto em radianos, formato `(N, 2)`, alinhados linha a
linha com os boxes dos rostos em `boxes`. A coluna 0 é o pitch e a coluna 1 é o
yaw, na convenção L2CS: yaw positivo gira o olhar para a esquerda do sujeito e
pitch positivo o gira para baixo. `pitch_deg` e `yaw_deg` convertem para graus,
e `direction_3d` retorna o vetor de direção unitário.

## SemanticMask

Mapa semântico denso, formato `(H, W)` de IDs de classe inteiros no canvas da
imagem original. `255` é o valor de ignorar e nunca conta como classe
(`SemanticMask.IGNORE_INDEX`). `classes` lista os IDs de classe presentes, e
`class_mask(class_id)` retorna a máscara booleana de uma classe.

## PanopticSegmentation

Cada pixel recebe exatamente um segmento sem sobreposição, unificando regiões
de stuff e instâncias de thing. `data` é um mapa `(H, W)` de IDs de segmento
inteiros; o ID de segmento `0` é não rotulado
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` é uma lista de dicts, um
por segmento, cada um com pelo menos `{"id": int, "category_id": int}`, onde
`id` corresponde a um valor no mapa e `category_id` indexa `names`.
`segment_ids` lista os IDs presentes e `segment_mask(segment_id)` retorna a
máscara booleana de um segmento.

Thing versus stuff é uma propriedade da categoria, não do segmento. Um payload
pode desnormalizar isso em cada segmento como `"isthing": bool`, e quando o
faz, o valor precisa concordar com o mapa em nível de categoria.

## DepthMap

Mapa denso de profundidade inversa relativa, formato `(H, W)` de floats no
canvas da imagem original. Valores mais altos significam mais perto da câmera.
Os valores são relativos, não metros métricos. `min`, `max` e `mean` são
calculados sobre os valores finitos, e `normalized()` reescala o mapa para
`[0, 1]`.

## NormalMap

Campo denso de normais de superfície, float32 `(H, W, 3)` no canvas da imagem
original, no referencial de câmera do OpenCV: `+x` para a direita, `+y` para
baixo, `+z` para dentro da cena. As normais apontam para a câmera, então uma
superfície fronto-paralela é `(0, 0, -1)`. Cada pixel é um vetor unitário.
`assert_normalized(atol=1e-4)` verifica essa invariante.

## EdgeMap

Mapa denso de probabilidade de borda, float32 `(H, W)` no canvas da imagem
original, onde `0` é não borda e `1` é borda. O mapa contínuo é mantido para
que o limiar continue sendo escolha de quem chama: `binary(threshold=0.5)`
aplica um, e `array` retorna a view numpy.

## RestoredImage

A imagem RGB restaurada, `(H, W, 3)` uint8. Para super-resolução o canvas é
`Results.restore_scale` vezes o da entrada. `array` retorna a view numpy e
`save(path)` grava a imagem.

## Matte

Matte de opacidade suave, float32 `(H, W)` em `[0, 1]` no canvas da imagem
original. `1` é totalmente primeiro plano e `0` é totalmente fundo. Um matte
suave engloba uma máscara rígida de remoção de fundo, limiarizada em 0.5, e
mantém as bordas com anti-aliasing que uma máscara binária descarta. `array`
retorna a view numpy.

Em um resultado de matte, `Results.cutout(image=None)` retorna um array RGBA
`(H, W, 4)` uint8 cujo quarto canal é o matte, e `Results.save(path, image=None)`
grava esse recorte como um PNG de fundo transparente. Os dois pegam o RGB de
`image` quando ele é passado, senão recarregam a imagem de `Results.path`.

## OCRRegions

Texto localizado com transcrições. `data` são polígonos float `(N, 4, 2)` em
pixels da imagem original, ordenados como superior esquerdo, superior direito,
inferior direito, inferior esquerdo, e as regiões vêm em ordem de leitura, de
cima para baixo e depois da esquerda para a direita. `texts` é a lista das N
transcrições. `conf` é o score de reconhecimento por região e `det_conf` o
score de detecção, ambos `(N,)`.

Os quadriláteros de detecção são polígonos de verdade, então eles não preenchem
`Results.boxes`. `xyxy` dá as envoltórias alinhadas aos eixos.

## Embeddings

Vetores normalizados em L2 da tarefa `embed`, sempre com formato `(N, D)`. Um
resultado de imagem inteira carrega uma linha e nenhum box; embeddings de
região ficam alinhados linha a linha com `boxes`. Como cada linha é
normalizada, a similaridade de cosseno é um produto escalar.

| Membro | Retorna |
|---|---|
| `dim` | `D` |
| `normalized` | As linhas, renormalizadas |
| `similarity(other)` | Similaridade de cosseno par a par contra outro `Embeddings` ou tensor |
| `verify(i, j, threshold=0.4)` | `True` quando as linhas `i` e `j` combinam |

## Identities

Correspondências nomeadas de galeria, alinhadas linha a linha com
`embeddings`. Produzidas quando uma `Gallery` é passada para uma predição
`embed`. `name` é uma lista onde uma entrada é `None` abaixo do limiar de
correspondência, e o nome mais próximo abaixo do limiar nunca é adivinhado.
`score` é o array de scores de correspondência e `data` junta os dois.

## Meshes

Meshes paramétricos de corpo humano, alinhados linha a linha com os boxes das
pessoas em `boxes`. Tudo está no referencial de câmera da imagem original.
`transl` é métrico em metros com `+z` apontando para longe da câmera;
`vertices` e `joints3d` são métricos e já incluem `transl`; `joints2d` está em
pixels no canvas da imagem original, não no recorte que a rede viu. Nenhum
campo carrega um referencial de mundo ou de gravidade.

Os layouts de parâmetros diferem entre modelos de corpo, então nada sobre os
formatos é fixado no código. `body_model` nomeia a parametrização e as
contagens são lidas de volta dos tensores: `num_vertices`, `num_joints`,
`num_betas` e `has_vertices`. `params` retorna o dict de parâmetros, e
`save_obj(path, index=0)` grava um mesh. Os campos são `global_orient`,
`body_pose`, `betas`, `transl`, `vertices`, `faces`, `joints3d`, `joints2d`,
`conf`, `focal_length` e `extras`.

Para `body_model="mhr"` as rotações são ângulos de Euler em radianos em vez de
eixo-ângulo, `body_pose` é um vetor plano de parâmetros por junta em vez de um
tripleto por junta, e `betas` são coeficientes de blendshape de identidade. A
escala do esqueleto, a pose das mãos e a expressão facial ficam em `extras`.

## Conversão e seleção

Todo payload carrega `to(*args, **kwargs)`, `cpu()`, `cuda()` e `numpy()`, e
chamar um deles no `Results` aplica a operação a todos os slots preenchidos de
uma vez.

<code-tabs name="convert" />

`result[idx]` seleciona linhas em todos os payloads alinhados linha a linha.
`len(result)` é o número de detecções, ou de pontos quando não há boxes.
`result.update(...)` retorna uma cópia com os slots nomeados substituídos; ele
aceita todos os slots mais `track_id` e `restore_scale`.

## summary e to_json

`summary(normalize=False, decimals=5, embeddings=False)` retorna uma lista de
dicts simples, uma linha por detecção, segmento, ponto ou região dependendo de
quais slots estão preenchidos. `to_json(**kwargs)` repassa os argumentos dele
para `summary` e retorna a string JSON.

`plot()` renderiza um resultado denso de normal ou de edge na visualização
canônica dele; ele levanta erro para outros tipos de resultado. As imagens
anotadas das outras tarefas vêm de `predict(save=True)`.
