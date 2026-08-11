---
title: Ensembles de detectores
seo_title: Ensembles de detectores na LibreYOLO
description: >-
  Rode vários detectores em uma mesma imagem e funda os boxes deles com weighted
  boxes fusion ou NMS, inclusive modelos com listas de classes diferentes.
lead: >-
  O LibreEnsemble roda dois ou mais detectores sobre a mesma imagem decodificada
  e funde os boxes deles em um único objeto Results. Cada membro mantém seus
  próprios pesos, limiares, dispositivos e lista de classes.
keywords:
  - ensemble de modelos detecção de objetos
  - weighted boxes fusion
  - wbf python
  - combinar dois detectores
  - fundir bounding boxes
  - LibreEnsemble
  - ensemble de detecção python
  - min_votes
last_verified: 1.5.0
verification: >-
  Assinaturas do construtor e da chamada, valores padrão, erros de validação,
  unificação do espaço de classes, contagem de votos e o Results retornado lidos
  de libreyolo/ensemble/model.py. Algoritmos de fusão e seus argumentos de
  libreyolo/ops/fusion.py. Intenção de design de
  docs/adr/0004-model-ensembling.md. Padrões de uso conferidos contra
  tests/unit/test_ensemble.py e tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: 'Dois detectores, fundidos'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Os membros podem ser caminhos de checkpoint ou modelos já carregados.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Pesos e uma exigência de votos
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # por convenção, proporcionais ao mAP de validação
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # mantém só os boxes que os dois membros acharam
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Limiares por membro
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Um escalar vale para todos os membros; uma lista é lida por membro.
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: Trazendo um detector que a LibreYOLO não carregou
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Retorna (boxes, scores, labels): xyxy em pixels da imagem original.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: As mesmas fontes que um modelo único aceita
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Troque clip.mp4 por um arquivo de vídeo em disco.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 4f4c54c52b295795
---

## O que é um ensemble

`LibreEnsemble` recebe dois ou mais detectores, roda cada um na mesma imagem e
funde os boxes deles em um único `Results`. É um construto de tempo de predição:
não há nada para treinar, e os membros continuam sendo modelos independentes que
podem ser validados e exportados por conta própria.

Detecção é a única tarefa suportada. Um membro cuja tarefa seja outra levanta
`ValueError` na construção, nomeando o índice do membro e sua tarefa.

Os dois nomes são importados de forma preguiçosa, então não custam nada até
serem usados:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Construindo um

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` é uma sequência de dois ou mais. Uma entrada `str` ou `Path` é
carregada através de `LibreYOLO()`; qualquer outra coisa precisa ser chamável e
expor um dict `names`. Menos de dois levanta `ValueError`, e passar uma string
solta levanta `TypeError` em vez de iterar seus caracteres.

`weights` tem `None` como padrão, o que significa peso uniforme. Os pesos
fornecidos precisam ser um por membro e estritamente positivos, então um peso
zero levanta erro em vez de descartar um membro silenciosamente. A convenção
documentada é defini-los proporcionais ao mAP de validação de cada membro.

`fusion_iou` tem `0.55` como padrão e é o IoU no qual boxes de membros diferentes
são agrupados em um mesmo cluster. É um limiar diferente do `iou` de cada
chamada, que é a configuração de NMS de cada membro.

`min_votes` tem `1` como padrão, ou seja, um único membro basta para sustentar um
box. Aumentar o valor mantém apenas os clusters confirmados por essa quantidade
de membros distintos. Precisa ser um inteiro positivo não maior que o número de
membros, e é limitado por classe ao número de membros que de fato conhecem
aquela classe, de modo que uma classe que só um membro treinou não é apagada
silenciosamente.

## Métodos de fusão

Três são aceitos por nome, e um callable também é aceito.

| `fusion` | Comportamento |
|---|---|
| `"wbf"` | Weighted boxes fusion, sequencial e fiel ao artigo original. O padrão |
| `"wbf_seeded"` | Weighted boxes fusion em uma passada; um NMS por classe escolhe as sementes dos clusters |
| `"nms"` | Concatena os boxes de todos os membros e aplica NMS por classe |

A weighted boxes fusion faz a média das coordenadas de um cluster ponderada pela
confiança, produzindo um box que nenhum membro sozinho propôs. As duas variantes
ponderadas concordam sempre que os clusters são inequívocos e podem divergir
levemente em cadeias de clusters sobrepostos. `"nms"` escolhe um sobrevivente em
vez de fazer média, então os sobreviventes mantêm seus scores originais, e os
pesos só influenciam qual box vence. Como ele seleciona em vez de agrupar, não
consegue contar votos: combinar `fusion="nms"` com `min_votes` maior que `1`
levanta `ValueError`.

A weighted boxes fusion reescala o score de um cluster pela fração do peso dos
membros que o sustentou. Com dois membros de peso igual, um box que só um deles
encontrou fica com metade do score: `0.9` vira `0.45`. Uma confiança fundida
pode, portanto, cair abaixo do `conf` com que cada membro rodou, então filtre
pelo score fundido em vez de supor que o limiar do membro continua valendo.

## Membros com listas de classes diferentes

Os membros não precisam compartilhar uma lista de classes. Os espaços de labels
deles são unidos por nome, e cada membro ganha uma tabela de lookup que remapeia
seus próprios ids de classe para a união. `ensemble.names` é essa união, e é ela
que o `Results` retornado carrega.

Boxes só se fundem dentro do mesmo nome de classe. Uma classe que só um membro
conhece passa sem fusão, e não é penalizada por isso: o reescalonamento do score
usa um denominador por classe, então uma classe conhecida por um só membro
mantém seu score.

Sobreposição parcial registra um aviso nomeando as classes que não são
compartilhadas por todos os membros. É esse aviso que vale ler com atenção,
porque um checkpoint cujos nomes de classe são placeholders como `class_0`
constrói uma união disjunta da de todos os outros membros, e nenhuma fusão entre
membros acontece.

Um membro que retorna um id de classe fora do seu próprio `names` levanta
`RuntimeError`.

## Detectores externos

<code-tabs name="external" />

`ExternalDetector(fn, names)` envolve qualquer callable que receba uma imagem PIL
e retorne `(boxes, scores, labels)`, com os boxes em xyxy nos pixels da imagem
original. Ele valida a aridade, o formato dos boxes, a concordância de
comprimento e que todo id de classe aparece em `names`, e aplica o limiar `conf`
por conta própria.

É assim que um detector que a LibreYOLO não carregou participa de uma fusão.

## Chamando o ensemble

<code-tabs name="sources" />

A assinatura da chamada espelha a de um modelo único, e ela aceita as mesmas
fontes: imagens, pastas, listas, vídeo, captura de tela, webcams e streams de
rede. Fontes ao vivo exigem `stream=True` pelo mesmo motivo de sempre.

| Argumento | Padrão | Observações |
|---|---|---|
| `conf` | `0.25` | Por membro; um escalar faz broadcast, ou um valor por membro |
| `iou` | `0.45` | O limiar de NMS de cada membro, não o limiar de fusão |
| `imgsz` | `None` | Uma `list` é lida por membro; um `int` ou tupla faz broadcast |
| `device` | `None` | Escalar ou um por membro, então os membros podem ficar em dispositivos diferentes |
| `classes` | `None` | Filtra o resultado fundido, sobre os ids de classe da união |
| `max_det` | `300` | Se aplica ao resultado fundido |

Como uma `list` significa por membro no caso de `imgsz`, `imgsz=[480, 640]` é 480
para o primeiro membro e 640 para o segundo, enquanto `imgsz=(480, 640)` é um
único tamanho retangular para todo mundo. É uma distinção fácil de tropeçar.

Os membros são chamados com um `max_det` de pelo menos 300 independentemente do
que você pedir, então cada um roda com folga e o ensemble corta uma única vez no
fim.

A imagem é decodificada uma vez e o mesmo objeto é entregue a todos os membros.
`batch` é aceito por paridade e ignorado; as imagens são processadas
sequencialmente.

## O que volta

Um `Results` comum, o mesmo tipo que um modelo único retorna, com `names`
apontando para o espaço de classes da união. Tudo o que está em
[Trabalhando com resultados](/docs/predict/results) vale sem mudanças.

A única diferença é `result.speed`, que um ensemble de fato preenche. Suas chaves
são `member_0`, `member_1` e assim por diante, mais `fusion`, em milissegundos.
Este é o único lugar da biblioteca onde `speed` é preenchido.

Linhas com boxes ou scores não finitos são descartadas antes da fusão. Quando os
membros estão em dispositivos diferentes, a fusão roda no dispositivo do primeiro
membro que retornou alguma coisa.

## O que um ensemble não faz

`val()` e `export()` levantam `NotImplementedError` e apontam você para os
membros: valide e exporte cada um individualmente. Não existe método `train`
nenhum, então chamá-lo levanta `AttributeError`.

A meia precisão não é tratada no nível do ensemble. `half=True` cai no mesmo
caminho de no-op com aviso de sempre; configure a precisão em cada membro.

Não há interface de linha de comando para ensembling. É uma API Python.
