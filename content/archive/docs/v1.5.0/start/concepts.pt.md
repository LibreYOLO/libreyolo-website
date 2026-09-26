---
title: Conceitos básicos
seo_title: Conceitos básicos do LibreYOLO
description: >-
  Como as tarefas, as famílias de modelos, os tamanhos e os nomes de arquivo dos
  checkpoints se encaixam no LibreYOLO, e o que cada nível de suporte promete.
lead: >-
  Quatro ideias descrevem todos os modelos do LibreYOLO: a tarefa que ele
  executa, a família a que pertence, o tamanho dentro dessa família e o nível de
  suporte em que a família está. O nome de arquivo do checkpoint codifica os
  três primeiros.
keywords:
  - conceitos libreyolo
  - tarefas libreyolo
  - famílias de modelos libreyolo
  - nome de arquivo checkpoint libreyolo
  - níveis de suporte libreyolo
  - tipos de tarefa visão computacional
last_verified: 1.5.0
meta:
  - label: Esquema do nome de arquivo
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Tarefas canônicas
    value: 17
  - label: Níveis de suporte
    value: 'Flagship, Core, Supported, Inference only, Museum, Sibling tier'
snippets:
  inspect:
    - label: Listar famílias
      language: bash
      code: |
        # Tarefas, tamanhos e resoluções de entrada de cada família registrada.
        libreyolo models
    - label: Um modelo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Escolher uma tarefa
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Os aliases são normalizados na fronteira da API: "keypoints" vira
        # "pose", "det" vira "detect", "semantic-segmentation" vira "semantic".
        model = LibreYOLO("LibreYOLO9t.pt", task="det")
        print(model.task)
source_hash: 23d045463a6a8411
---

## Tarefas

Uma tarefa é o que um modelo retorna. O LibreYOLO tem dezessete nomes canônicos
de tarefa, e cada um deles dá nome ao campo do objeto `Results` que carrega sua
saída.

| Tarefa | Retorna |
|---|---|
| `detect` | Caixas alinhadas aos eixos com uma classe e uma confiança |
| `segment` | Máscaras por instância, uma máscara por objeto detectado |
| `semantic` | Um rótulo de classe por pixel, sem separação de instâncias |
| `panoptic` | Um rótulo não sobreposto por pixel, unindo coisas contáveis com regiões amorfas |
| `pose` | Keypoints por instância, com as linhas alinhadas às caixas |
| `classify` | Uma probabilidade sobre um conjunto de rótulos para a imagem inteira |
| `obb` | Caixas orientadas, com um ângulo de rotação |
| `point` | Uma coordenada de imagem por detecção, em vez de uma caixa |
| `depth` | Um mapa denso de profundidade inversa relativa |
| `normal` | Um campo denso de normais de superfície em vetores unitários |
| `edge` | Um mapa denso de probabilidade de bordas |
| `restore` | Uma imagem RGB restaurada, para deblurring, remoção de ruído ou super-resolução |
| `matte` | Um mapa suave de primeiro plano de 0 a 1, para remoção de fundo |
| `ocr` | Quadriláteros de texto com transcrições, na ordem de leitura |
| `embed` | Um vetor normalizado em L2 cujo produto escalar mede a concordância |
| `gaze` | Uma direção do olhar por rosto detectado |
| `mesh` | Um corpo 3D com pose por pessoa detectada |

Esses são os nomes que aparecem nos metadados do checkpoint e nos nomes de
arquivo. Aliases conhecidos são aceitos em qualquer lugar onde uma tarefa é
passada e normalizados antes de qualquer outra coisa acontecer: `detection` e
`det` viram `detect`, `keypoints` vira `pose`, `cls` vira `classify`, `deblur`,
`denoise` e `super-resolution` viram todos `restore`, `face-recognition` e
`reid` viram `embed`. Um nome não reconhecido levanta erro em vez de cair
silenciosamente em um padrão.

`segment`, `semantic` e `panoptic` são três tarefas diferentes, não três
palavras para a mesma coisa. Máscaras de instância, rótulos por pixel e o mapa
combinado de coisas contáveis e regiões amorfas têm ground truth diferente,
métricas diferentes e campos de resultado diferentes.

## Famílias de modelos

Uma família é uma linhagem de arquitetura com seu próprio código de carregamento,
pré-processamento e pós-processamento. Toda família declara um identificador
`FAMILY` como `yolo9`, `rfdetr` ou `dfine`, as tarefas que suporta e a resolução
de entrada para cada tamanho que publica.

`LibreYOLO()` é uma fábrica, não uma classe. Dado um caminho, ela carrega o
arquivo, identifica a família pelos metadados do checkpoint ou, na falta deles,
pelas próprias chaves dos tensores, e retorna uma instância do modelo daquela
família. É por isso que trocar de detector é uma mudança de uma linha: o objeto
que volta expõe a mesma superfície `predict`, `train`, `val` e `export` e retorna
o mesmo tipo `Results`.

<code-tabs name="inspect" />

Uma família que atende mais de uma tarefa normalmente publica um checkpoint
separado por tarefa, muitas vezes com um conjunto de tamanhos diferente para cada
uma; algumas poucas compartilham um único artefato entre duas tarefas em tempo de
execução. De um jeito ou de outro, as tarefas suportadas são uma lista fixa, e
pedir uma que esteja fora dela levanta erro com a lista suportada na mensagem, em
vez de carregar algo aproximado.

A lista completa, com benchmarks por família e os pesos publicados, está em
[todos os modelos](/docs/models).

## Tamanhos

Um tamanho é uma variante dentro de uma família, escrito como um código em letras
minúsculas colado direto no prefixo da família. As letras comuns são `n` para
nano, `t` para tiny, `s` para small, `m` para medium, `l` para large e `x` para
xlarge, mas os códigos são específicos de cada família e várias delas usam algo
completamente diferente: códigos que nomeiam o backbone, como `r50` ou `r101`,
em que o tamanho é uma profundidade de ResNet; códigos de escalonamento composto,
como `b0` até `b3`; ou um nome que identifica o único checkpoint lançado. O
YOLOv9 usa `c` para compact onde outras famílias usam `l`.

O tamanho também fixa a resolução de entrada, e para famílias com várias tarefas
a resolução pode diferir por tarefa. Ambos são lidos da família, nunca
presumidos; `libreyolo models` imprime os dois.

## Nomes de arquivo dos checkpoints

Todo arquivo de pesos publicado segue um único esquema:

```text
Libre<FAMILY><size>[-<task>].pt
```

O prefixo da família é uma string fixa por família, o tamanho é minúsculo e vem
colado sem separador, e o sufixo de tarefa é precedido por hífen. Detecção não
carrega sufixo, seguindo a convenção que os checkpoints YOLO sempre usaram, então
`LibreYOLO9t.pt` é um detector e `LibreRFDETRn-seg.pt` é um modelo de segmentação
da mesma família.

| Tarefa | Sufixo |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Uma família sem nenhuma tarefa sem sufixo pode exigir o sufixo, de modo que um
nome sem ele não é aceito como checkpoint válido para ela. Uma família que
publica pesos treinados em um dataset diferente do seu padrão acrescenta o nome
do dataset como um sufixo a mais, e essa variante continua fazendo parte do nome
do repositório de onde o arquivo é baixado.

Três níveis ficam fora desse esquema. As famílias de segmentação promptable, as
famílias de visão e linguagem e os detectores de vocabulário aberto não são
registrados na fábrica de checkpoints e não emitem nenhum arquivo
`Libre<FAMILY><size>.pt`. O prefixo delas nomeia um snapshot baixado do Hugging
Face ou um checkpoint promptable, e ali as maiúsculas da marca original são
preservadas de propósito.

## Como a tarefa é decidida

Quando vários sinais poderiam nomear uma tarefa, eles são consultados em uma
ordem fixa e o primeiro que estiver presente vence: o argumento `task` que você
passou, depois a tarefa registrada nos metadados do checkpoint, depois o sufixo
de tarefa no nome do arquivo, depois a tarefa padrão da família. O resultado é
conferido contra as tarefas suportadas pela família antes de o modelo ser
construído, então uma incompatibilidade falha no momento da carga em vez de
produzir uma saída errada mais adiante.

## Níveis de suporte

As famílias são inscritas em exatamente um nível. Um nível é uma afirmação sobre
atenção de engenharia, não sobre acurácia: ele diz onde um recurso novo chega
primeiro e o que é mantido no verde.

| Nível | O que significa |
|---|---|
| Flagship | Os recursos são projetados e totalmente validados em GPU aqui primeiro |
| Core | Detectores treináveis principais. Os recursos seguem os flagships na mesma leva de releases |
| Supported | Famílias treináveis de apoio. Mantidas no verde no CI, os recursos chegam de forma oportunista |
| Inference only | Predizer, validar e exportar. Recursos de treinamento não se aplicam |
| Museum | Uma peça de museu congelada. Somente correções de bugs |
| Sibling tier | Uma superfície de produto separada, com sua própria fábrica e seu próprio contrato |

Cada página de modelo traz o nível da sua família no cabeçalho. As duas famílias
flagship são [YOLOv9](/docs/models/yolov9) para os detectores CNN e
[RF-DETR](/docs/models/rf-detr) para os detectores transformer; comece por elas a
menos que você tenha um motivo para não fazer isso.

Inference only diz o que está faltando, que é um laço de treinamento no
LibreYOLO. Predizer, validar e, onde a família suportar, exportar funcionam
todos. Chamar `train()` numa família dessas levanta `NotImplementedError`
nomeando o motivo.
