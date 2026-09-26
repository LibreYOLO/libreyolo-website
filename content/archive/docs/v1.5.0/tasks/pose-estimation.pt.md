---
title: Estimativa de pose
seo_title: Estimativa de pose no LibreYOLO
description: >-
  Preveja keypoints por instância no LibreYOLO: as famílias que atendem à
  tarefa, o formato de rótulos e as chamadas de predição, treinamento, validação
  e exportação.
lead: >-
  A estimativa de pose localiza cada instância e retorna um conjunto ordenado de
  keypoints nomeados para ela, de modo que a saída carrega a estrutura interna
  do objeto, e não apenas sua extensão. A chave da tarefa é pose.
keywords:
  - estimativa de pose python
  - detecção de keypoints
  - modelo de pose humana
  - keypoints COCO
  - OKS mAP
  - treinar modelo de pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -pose no nome do arquivo seleciona a cabeça de keypoints,
        # então nenhum argumento de tarefa é necessário.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) coordenadas em pixels
        print(result.boxes.xyxy.shape)     # (N, 4), as mesmas N instâncias
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Apenas keypoints visíveis
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible é derivado da terceira coluna de keypoints e é todo
        # verdadeiro quando o checkpoint prevê apenas (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Top-down como alternativa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O HRNet é top-down: ele recorta cada pessoa primeiro. Sem uma fonte de

        # pessoas, ele se combina com um detector LibreYOLO9t e registra a
        escolha.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml carrega um script de download embutido, então precisa
        # de permissão explícita a menos que os dados já estejam locais.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Seu próprio dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml precisa declarar kpt_shape, e as linhas de rótulo precisam
        # carregar exatamente 5 + K * D campos.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() retorna um dict simples, não um objeto.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como um checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definição

A estimativa de pose retorna estrutura, não apenas extensão. Cada instância
continua recebendo uma caixa, uma classe e uma pontuação, e também recebe `K`
keypoints em uma ordem fixa, de modo que o índice 5 significa a mesma parte do
corpo em cada instância e em cada imagem. O conjunto de rótulos define essa
ordem; nada na saída identifica um keypoint pelo nome.

`pose` é a chave canônica da tarefa, e o sufixo `-pose` no nome do arquivo de um
checkpoint a seleciona, então `task=` não é necessário ao carregar pesos
publicados.

`predict()` preenche `result.keypoints` junto de `result.boxes`. `.data` é
`(N, K, 2)` ou `(N, K, 3)`, alinhado por linha com as caixas, então a instância
`i` em um é a instância `i` no outro. `.xy` fatia as coordenadas em pixels e
`.xyn` as normaliza pelo tamanho da imagem original. `.conf` é a terceira coluna
quando o checkpoint prevê uma e `None` quando não prevê, e `.has_visible` é a
máscara booleana derivada dela, toda verdadeira quando não há terceira coluna.

Duas arquiteturas chegam a essa saída. Um modelo de um estágio prevê caixas e
keypoints em uma única passada. Um modelo top-down roda um detector primeiro,
recorta cada instância e faz a regressão dos keypoints dentro do recorte, então
sua acurácia depende do detector que está na frente dele.

## Modelos

Três famílias treinam e fazem predição:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) e
[YOLO-NAS](/docs/models/yolo-nas), todas de um estágio. O RF-DETR precisa do seu
próprio extra, `pip install "libreyolo[rfdetr]"`. RF-DETR e EdgeCrafter trazem
checkpoints de pose publicados e ambos fazem fine-tuning em datasets de classe
única, apenas com pessoas; a cabeça de keypoints do EdgeCrafter é fixada na
construção e rejeita um dataset que declare uma contagem diferente, enquanto o
RF-DETR reinicializa sua cabeça para ela. O YOLO-NAS baixa seus pesos do CDN da
própria Deci.AI sob uma licença não comercial, e o LibreYOLO não publica nenhum
deles; sua cabeça de pose também é reconstruída para uma nova contagem de
keypoints, e ela é a única das três cuja contagem de classes não é fixada em um,
então é a família para um esqueleto multiclasse ou não humano, como pose de
animais.

O [HRNet](/docs/models/hrnet) é a opção top-down. Ele prevê, valida e exporta, e
seu `train()` levanta `NotImplementedError`. Sem uma fonte de pessoas, ele se
combina automaticamente com um detector LibreYOLO9t; `cropped=True` trata a
imagem inteira como uma instância, `person_boxes=` recebe caixas que você já
tem, e `person_detector=` nomeia um detector diferente.

O [SenseNova-Vision](/docs/models/sensenova-vision) também emite keypoints. É um
modelo generativo guiado por prompt, com sua própria factory, `LibreVLM`, e seu
próprio extra; sem nenhum vocabulário definido, `set_task("pose")` recorre à
categoria pessoa.
Seus pesos são não comerciais, e a latência por imagem é bem mais alta do que a
de uma cabeça de pose feita para isso, porque cada predição é uma decodificação
por difusão.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

As contagens e as ordens de keypoints são propriedades do checkpoint, não da
biblioteca, então um modelo treinado em outro esqueleto retorna um `K` diferente
e um significado diferente por índice. O que a terceira coluna de keypoints
guarda também é uma propriedade do checkpoint: o EdgeCrafter escreve uma
constante ali em vez de uma pontuação por ponto, e ele não tem cabeça de caixas
nenhuma, então cada uma de suas caixas de pose é a extensão delimitadora dos
próprios keypoints daquela instância. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

O layout é o layout de detecção: um arquivo de rótulos `.txt` por imagem,
encontrado trocando `images` por `labels` no caminho da imagem e mudando a
extensão.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Uma linha é uma linha de detecção com os keypoints anexados:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

A contagem de campos é exatamente `5 + K * D`, onde `D` é o segundo valor de
`kpt_shape`. As coordenadas de caixa e de keypoints são floats normalizados em
relação à largura e à altura da imagem original. A visibilidade `v`, presente
apenas quando `D` é 3, é `0`, `1` ou `2`.

O YAML adiciona duas chaves ao contrato compartilhado:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` é obrigatório e é `[K, 2]` ou `[K, 3]`. `flip_idx` é opcional e é
uma permutação de `0..K-1` que dá, para cada keypoint, o índice que ele assume
depois de um espelhamento horizontal, que é como um pulso esquerdo continua
sendo um pulso esquerdo. Omita-o e o data augmentation de espelhamento
horizontal é desligado para os keypoints, em vez de aplicado com a ordem de
índices errada.

## Treinamento

<code-tabs name="train" />

O treinamento continua a partir de um checkpoint `-pose` publicado, que já
carrega a cabeça de keypoints; a tarefa é lida do checkpoint que você carrega, e
não de uma flag passada no momento do treinamento, então um checkpoint de
detecção não vira uma execução de pose só por pedir isso. O `kpt_shape` do seu
YAML precisa bater exatamente com a cabeça no caso do EdgeCrafter, já que a
cabeça dele é fixada na construção, enquanto RF-DETR e YOLO-NAS redimensionam a
cabeça para uma contagem diferente. Veja
[treinamento](/docs/train) para datasets, augmentation, multi-GPU e loggers.

## Validação

`val()` retorna um dicionário simples de chaves `metrics/`. A pontuação usa a
avaliação de keypoints do COCO baseada na Object Keypoint Similarity, que pondera
o erro de distância de cada keypoint pela escala da instância e por uma
tolerância por keypoint, então ela faz o papel que o IoU faz para as caixas. Ela
precisa do `pycocotools`, que está na instalação base.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` é o número principal, a precisão média (mean
average precision) calculada sobre os limiares de OKS de 0.50 a 0.95, e é o que
o treinamento usa para escolher a melhor época. `metrics/keypoints_mAP50` e
`metrics/keypoints_mAP75` são as versões de limiar único, e
`metrics/keypoints_mAP_M` e `metrics/keypoints_mAP_L` separam a média por área
da instância, média e grande; a avaliação de keypoints do COCO não define um
grupo de instâncias pequenas. Os números de
recall médio correspondentes são `metrics/keypoints_AR50-95`,
`metrics/keypoints_AR50`, `metrics/keypoints_AR75`, `metrics/keypoints_AR_M` e
`metrics/keypoints_AR_L`. Toda chave desta tarefa tem o prefixo `keypoints_`,
então as chaves de `mAP` de caixa que um detector retorna não aparecem.

## Exportação

<code-tabs name="export" />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do seu
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
retorna o mesmo `Results`. A cobertura de formatos difere por família; a matriz
em cada página de modelo é gerada a partir do conjunto validado, e não digitada
à mão. Veja
[exportação e deploy](/docs/export) para os formatos, seus extras e suas
restrições.
