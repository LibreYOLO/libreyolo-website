---
title: Trabalhando com resultados
seo_title: O objeto Results do LibreYOLO
description: >-
  Um objeto Results por imagem, com um slot por tipo de payload: boxes,
  máscaras, keypoints, probs, profundidade, panóptico, OCR e mais. Desenho,
  salvamento e JSON.
lead: >-
  Toda predição retorna um objeto Results por imagem. Ele tem um slot nomeado
  por tipo de payload, todos vazios exceto os que o modelo produz, mais os
  mesmos slots em um artefato exportado.
keywords:
  - objeto results yolo python
  - results.boxes xyxy
  - converter results em json
  - salvar imagem anotada yolo
  - máscaras de segmentação python
  - keypoints results python
  - mapa de profundidade results
  - results summary yolo
  - onnx mesmos resultados yolo
last_verified: 1.5.0
verification: >-
  Classes de payload, slots, semântica de movimentação, summary(), to_json(),
  plot(), save() e cutout() lidos de libreyolo/utils/results.py. Comportamento
  de anotação e de escrita em disco vindo de
  InferenceRunner._save_annotated_image em libreyolo/models/base/inference.py e
  de resolve_save_path em libreyolo/utils/general.py. Despacho por sufixo vindo
  de LibreYOLO() em libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Boxes
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (altura, largura) da imagem de origem

        print(result.path)         # caminho de origem, None para entrada em
        memória


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Coordenadas normalizadas
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # pixels, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # pixels, centro x, centro y, w, h

        print(result.boxes.xyxyn[:1])   # o mesmo box dividido pela largura e
        pela altura

        print(result.boxes.xywhn[:1])
    - label: NumPy e dispositivos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Cada um deles retorna um novo Results; o original não muda.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary e to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # O mesmo conteúdo como string, com os mesmos argumentos nomeados.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Imagens anotadas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True desenha o payload e grava em runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Instalar o extra de exportação
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: O mesmo Results a partir de um artefato exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # retorna o caminho gravado

        # LibreYOLO() despacha pelo sufixo do arquivo.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Um objeto, um slot por payload

Uma predição sobre uma imagem retorna um `Results`. Ele carrega dezoito slots
de payload, e um modelo preenche apenas os que a sua tarefa produz. Todos os
outros slots são `None`, então ler `result.masks` em um detector dá `None` em
vez de um erro.

| Slot | Classe | Shape | Produzido por |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` mais scores e classes | Detecção, e qualquer tarefa que localiza primeiro |
| `masks` | `Masks` | `(N, H, W)` | Segmentação de instâncias |
| `keypoints` | `Keypoints` | `(N, K, 2)` ou `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Classificação |
| `obb` | `OBB` | `(N, 7)` ou `(N, 8)` | Caixas orientadas |
| `gaze` | `Gaze` | `(N, 2)` com pitch e yaw em radianos | Estimativa de olhar |
| `points` | `Points` | `(N, 4)` como x, y, classe, confiança | Localização de pontos |
| `semantic_mask` | `SemanticMask` | `(H, W)` com ids de classe | Segmentação semântica |
| `panoptic` | `PanopticSegmentation` | `(H, W)` com ids de segmento, mais `segments_info` | Segmentação panóptica |
| `depth_map` | `DepthMap` | `(H, W)` de floats | Estimativa de profundidade |
| `normal_map` | `NormalMap` | `(H, W, 3)` de vetores unitários | Normais de superfície |
| `edges` | `EdgeMap` | `(H, W)` de floats em `[0, 1]` | Detecção de bordas |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restauração e super-resolução |
| `matte` | `Matte` | `(H, W)` de floats em `[0, 1]` | Alpha matting e remoção de fundo |
| `ocr` | `OCRRegions` | `(N, 4, 2)` de polígonos mais as transcrições | Detecção e reconhecimento de texto |
| `embeddings` | `Embeddings` | `(N, D)` com linhas normalizadas em L2 | A tarefa `embed` |
| `identities` | `Identities` | N nomes e scores | A tarefa `embed` com uma galeria |
| `meshes` | `Meshes` | Parâmetros do corpo e vértices opcionais | Recuperação de malha corporal |

Ao lado deles ficam os campos que todo resultado tem: `orig_shape` como
`(altura, largura)`, `path` (o caminho de origem, ou `None` para entrada em
memória), `names` mapeando id de classe para nome de classe, `frame_idx` para
vídeo e frames ao vivo, `track_id` quando há tracking, e `restore_scale`, o
fator inteiro de aumento de escala de um resultado de restauração.

`result.normals` é um alias para `result.normal_map`.

`result.speed` existe em todo resultado, mas só é preenchido pelos
[ensembles](/docs/predict/ensembling), onde suas chaves são `member_0`,
`member_1` e `fusion` em milissegundos. Para um modelo único ele continua um
dict vazio.

## Boxes

<code-tabs name="basic" />

`Boxes` mantém coordenadas e scores como arrays separados, em vez de um único
tensor empacotado.

| Atributo | Conteúdo |
|---|---|
| `xyxy` | `(N, 4)` em pixels absolutos, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` em pixels absolutos, centro x, centro y, largura, altura |
| `xyxyn`, `xywhn` | Os mesmos divididos pela largura e pela altura da imagem |
| `conf` | `(N,)` de confiança |
| `cls` | `(N,)` de id de classe, como float |
| `id` | `(N,)` de id de track, ou `None` |
| `is_track` | Se `id` está definido |
| `data` | Tudo concatenado: boxes, id opcional, conf, cls |

`cls` é um array de floats, então use `result.names[int(cls)]`.

`xyxyn` e `xywhn` precisam de `orig_shape`, que o `Results` preenche para você.

## Payloads densos

Payloads que cobrem a imagem inteira se comportam de forma diferente dos
payloads por instância, e isso importa na hora de fatiar.

`SemanticMask` guarda ids de classe `(H, W)` no canvas original, com `255`
reservado como o valor de ignorar, que nunca conta como classe. `classes` lista
os ids presentes e o exclui; `class_mask(id)` retorna um `(H, W)` booleano.

`PanopticSegmentation` guarda ids de segmento `(H, W)`, com `0` como o id de
vazio, e uma lista `segments_info` de dicts que carregam ao menos `id` e
`category_id`. `segment_ids` lista os ids presentes, `segment_mask(id)`
seleciona um deles.

`DepthMap` guarda profundidade inversa relativa `(H, W)`: mais alto significa
mais perto, e os valores não são metros em escala métrica. Ele expõe `min`,
`max` e `mean` sobre os valores finitos, e `normalized()`, que reescala para
`[0, 1]`.

`NormalMap` guarda vetores unitários `(H, W, 3)` no sistema de câmera do
OpenCV, com `+x` para a direita, `+y` para baixo e `+z` para dentro da cena, de
modo que uma superfície voltada para a câmera é `(0, 0, -1)`.
`assert_normalized()` verifica se todo pixel é finito e tem comprimento
unitário.

`EdgeMap` guarda float32 `(H, W)` em `[0, 1]`. O mapa contínuo é mantido em vez
de limiarizado, então `binary(threshold=0.5)` é onde você escolhe o ponto de
corte.

`Matte` guarda float32 `(H, W)` em `[0, 1]`, onde `1` é totalmente primeiro
plano. `array` retorna esse mapa com clipping, como float32.

`RestoredImage` guarda `(H, W, 3)` uint8 RGB, com `array` para o ndarray cru e
`save(path)` para gravá-lo.

`Probs` guarda um vetor de probabilidades para a imagem. `top1` e `top5` são
índices de classe, `top1conf` e `top5conf` os scores correspondentes.

`Embeddings` guarda linhas `(N, D)` que já vêm normalizadas em L2, então a
similaridade de cosseno é um produto escalar. `similarity(other)` retorna
`(N, M)` contra uma galeria ou `(N,)` contra um único vetor, e
`verify(i, j, threshold=0.4)` compara duas linhas.

`OCRRegions` guarda polígonos `(N, 4, 2)` na ordem de leitura, com os cantos
ordenados como superior-esquerdo, superior-direito, inferior-direito,
inferior-esquerdo. As transcrições ficam em `texts`, os scores de
reconhecimento em `conf`, os scores de detecção em `det_conf`. Como são
polígonos rotacionados de verdade, eles não preenchem `boxes`; `ocr.xyxy` dá os
envoltórios alinhados aos eixos quando você precisa de retângulos.

## Fatiamento e movimentação

`result[i]` retorna um novo `Results` com uma única instância. Payloads por
instância são fatiados; payloads da imagem inteira são repassados sem
alteração, de modo que fatiar um resultado de classificação não tem como
truncar o seu vetor de probabilidades para uma única classe, e fatiar um
resultado de profundidade não tem como corromper o layout `(H, W)`.

`len(result)` conta instâncias: boxes, pontos, embeddings, regiões de OCR ou
malhas. Qualquer payload denso da imagem inteira conta como `1`. Um resultado
sem nada dentro é `0`.

`to()`, `cpu()`, `cuda()` e `numpy()` retornam, cada um, um novo `Results` com
todos os slots preenchidos convertidos. Eles não modificam o original.

`update()` é o único método que muta no lugar, substituindo slots nomeados e
retornando o mesmo objeto.

## JSON

<code-tabs name="json" />

`summary()` retorna uma lista de dicts simples, e `to_json()` é essa lista
passada por `json.dumps`. Os dois recebem os mesmos três argumentos:
`normalize=False` muda as coordenadas para `[0, 1]`, `decimals=5` define o
arredondamento, e `embeddings=False` controla se os vetores de embedding são
incluídos.

O formato da linha segue o payload. Linhas de detecção carregam `name`,
`class`, `confidence` e um dict `box`, e ganham `segments` quando há máscaras,
`obb` e `corners` para caixas orientadas, ângulos de `gaze` em radianos e em
graus, `track_id` quando há tracking, e parâmetros de `mesh` quando há malhas.

Quando não há boxes, um único payload decide as linhas: o OCR emite uma linha
por região com o seu `text`, points uma linha por ponto, panóptico uma linha
por segmento com `pixel_count` e `pixel_fraction`, semântico uma linha por
classe presente, classificação as cinco classes do topo. Profundidade, normais,
bordas, restauração e matting emitem, cada um, uma única linha de resumo
descrevendo o mapa em vez dos seus pixels.

Dois payloads são abreviados de propósito. Um vetor de embedding é reportado
apenas como `embedding_dim`, porque uma linha de 512 floats dá cerca de 2 KB
por rosto; passe `embeddings=True` para incluir os valores. Os vértices de
malha nunca são incluídos, já que isso são dezenas de milhares de coordenadas
por pessoa. Leia `result.meshes.vertices` ou chame
`result.meshes.save_obj(path)` para obter a geometria.

## Desenho e salvamento

<code-tabs name="saving" />

`predict(save=True)` é o caminho que anota e grava. Ele escolhe a rotina de
desenho a partir do slot que estiver preenchido, então um resultado semântico é
gravado como uma máscara colorida, um resultado de profundidade como uma
visualização de profundidade, um resultado panóptico com os seus segmentos, um
matte como um PNG RGBA de fundo transparente, e um detector como boxes com as
máscaras por baixo. O caminho gravado fica anexado ao resultado como
`result.saved_path`.

`Results.plot()` é mais estreito do que o nome sugere. Ele está definido apenas
para mapas de normais e mapas de bordas, e levanta `NotImplementedError` para
qualquer outra coisa. Use `save=True` nas demais tarefas.

`Results.save(path)` é igualmente estreito: grava um resultado de matte como um
recorte PNG RGBA de fundo transparente e levanta `NotImplementedError` no resto
dos casos. `Results.cutout()` retorna esse mesmo array RGBA sem gravá-lo. Os
dois precisam da imagem de origem, obtida de `result.path` ou passada como
`image=`.

Dois payloads trazem os seus próprios gravadores: `result.restored.save(path)`
para uma imagem restaurada, e `result.meshes.save_obj(path, index=0)` para uma
malha.

Para saber onde os arquivos vão parar e como `output_path` e
`output_file_format` se comportam, veja
[Fontes de predição](/docs/predict/sources).

## Artefatos exportados retornam o mesmo objeto

<code-tabs name="exported" />

`LibreYOLO()` despacha pelo sufixo do arquivo, então um artefato exportado
carrega pela mesma chamada que um checkpoint `.pt` e retorna o mesmo `Results`.
Arquivos `.onnx`, `.engine`, `.pte` e `.mnn` são reconhecidos pelo sufixo,
assim como diretórios OpenVINO, Paddle e ncnn e uma URL de modelo do Triton.
Código que lê `result.boxes.xyxy` não muda quando um modelo é trocado pelo seu
build exportado. Veja [Exportação](/docs/export) para o conjunto completo de
formatos.

Recorrer à API do próprio runtime, em vez disso, significa assumir você mesmo o
pré-processamento, o pós-processamento e os nomes de classe.
