---
title: API de ensemble
seo_title: API do LibreEnsemble e operações de fusão
description: >-
  LibreEnsemble, ExternalDetector e as três operações de fusão em libreyolo.ops:
  weighted boxes fusion, sua variante com sementes e a fusão por NMS por classe.
lead: >-
  LibreEnsemble roda vários detectores sobre a mesma imagem e funde as detecções
  deles em um único Results. A fusão acontece depois do pós-processamento
  próprio de cada membro, então cada um mantém o seu tamanho de entrada, a sua
  normalização e a sua supressão.
keywords:
  - LibreEnsemble
  - ensemble de detecção de objetos
  - weighted boxes fusion python
  - ExternalDetector
  - libreyolo.ops.fusion
  - consenso min_votes
last_verified: 1.5.0
verification: >-
  Assinaturas e valores padrão lidos de libreyolo/ensemble/model.py e
  libreyolo/ops/fusion.py na v1.5.0. Intenção de projeto vinda de
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Dois membros, fusão padrão'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Uma fonte de imagem única retorna um Results, não uma lista.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Consenso e limiares por membro
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: 'Operação de fusão, sem modelo envolvido'
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

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

| Argumento | Padrão | Significado |
|---|---|---|
| `members` | | Dois ou mais detectores |
| `weights` | `None` | Fatores de confiabilidade por membro; todos `1.0` quando omitido |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` ou um callable |
| `fusion_iou` | `0.55` | Limiar de IoU para o clustering da fusão |
| `min_votes` | `1` | Mantém apenas os boxes confirmados por pelo menos esta quantidade de membros |

Um membro é um caminho de pesos resolvido pela factory `LibreYOLO()`, um modelo
já construído, um backend exportado ou um `ExternalDetector`. Todo membro
precisa ser um modelo da tarefa de detecção.

<code-tabs name="usage" />

A construção rejeita menos de dois membros, uma lista `weights` de tamanho
errado, um peso não positivo, um `min_votes` que não seja um inteiro positivo e
um `min_votes` maior que a quantidade de membros. `fusion="nms"` com
`min_votes > 1` também gera erro, porque o NMS descarta a informação de qual
cluster cada box pertence e não consegue contar votos.

`weights` escala a confiabilidade atribuída a cada membro. Um peso maior puxa as
coordenadas e as pontuações fundidas na direção daquele membro. A convenção é
deixá-los proporcionais ao mAP de validação.

## Espaços de classes

Membros com `names` idênticos passam direto. Caso contrário, os espaços de
classes são unidos por nome, os IDs de classe de cada membro são remapeados por
tabelas de correspondência e o `Results.names` fundido é a união. A fusão junta
boxes apenas dentro de uma mesma classe unificada, então uma classe que só um
membro conhece passa sem ser fundida. Uma divergência registra um aviso na
construção.

`min_votes` é limitado por classe conforme a quantidade de membros cujos espaços
de rótulos contêm aquela classe, de modo que o consenso continue fazendo sentido
em vocabulários parcialmente compartilhados.

## Chamar o ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` é um alias de `__call__`. O retorno é o `Results` de sempre, cujo
`speed` detalha o custo por membro e acrescenta uma entrada `fusion`. Uma fonte
de imagem única retorna um deles, uma lista ou um diretório retorna uma lista, e
`stream=True` retorna um gerador.

`conf`, `iou` e `device` se propagam para todos os membros e também aceitam um
valor por membro, então `conf=[0.25, 0.4]` dá ao membro 0 um limiar de 0.25 e ao
membro 1 um limiar de 0.4. `imgsz` se propaga quando é um int ou uma tupla, e é
por membro apenas quando é uma lista, então `imgsz=(480, 640)` é um único tamanho
retangular para todos, enquanto `imgsz=[480, 640]` é 480 para o membro 0 e 640
para o membro 1. Cada entrada precisa ser válida para a família daquele membro.

`augment` se propaga para os membros que suportam aumento em tempo de teste
(test-time augmentation), e os backends exportados o ignoram. `classes` recebe
IDs de classe da união e `max_det` se aplica ao resultado fundido, então os
membros rodam com folga e o ensemble corta uma vez só. `batch` é aceito por
paridade de API; as imagens são processadas sequencialmente.

`val()` e `export()` levantam `NotImplementedError`. Valide e exporte os membros
individualmente.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Adapta qualquer callable de detecção em um membro. `fn` recebe uma imagem PIL e
retorna `(boxes, scores, labels)`, em que os boxes são xyxy em pixels da imagem
original e os rótulos são IDs de classe válidos em `names`. Tensores, arrays e
listas aninhadas funcionam todos. A LibreYOLO não importa nada do código externo.

O adaptador valida o retorno: precisa ser uma tupla de três elementos, os boxes
precisam ter forma `(N, 4)`, os três arrays precisam ter o mesmo tamanho e todo
ID de classe precisa aparecer em `names`. Detecções iguais ou inferiores a `conf`
são descartadas antes da fusão.

## Operações de fusão

As primitivas de fusão são ops de torch independentes em `libreyolo.ops`. Elas
não dependem de modelo e podem ser importadas por conta própria, e é por isso que
são exportadas separadamente do ensemble.

<code-tabs name="ops" />

As três recebem os mesmos argumentos posicionais, `boxes, scores, labels,
model_ids`, e retornam `(boxes, scores, labels)`.

| Operação | Chave no registro | Comportamento |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Weighted boxes fusion sequencial, fiel ao paper |
| `wbf_seeded` | `wbf_seeded` | Variante paralela em uma passada só da mesma redução |
| `nms_fusion` | `nms` | Concatena tudo e aplica NMS por classe |

`FUSIONS` associa as três chaves do registro aos callables, e `LibreEnsemble`
resolve o valor de `fusion=` ali.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` tem a assinatura idêntica. `nms_fusion` recebe os mesmos argumentos
exceto `conf_type`, e levanta `ValueError` quando `min_votes > 1`.

Em `weighted_boxes_fusion`, as detecções são percorridas em ordem decrescente de
confiança escalada pelo peso. Cada uma ou entra no cluster existente com cujo box
fundido parcial ela tem a melhor sobreposição, com IoU acima de `iou_thr` e o
mesmo rótulo, ou abre um cluster novo. O box fundido de um cluster é a média das
coordenadas dos seus membros ponderada pela confiança, e a pontuação dele é a
média ponderada ou o máximo das confianças deles, reescalada para que boxes
confirmados por menos modelos pontuem mais baixo.

`wbf_seeded` escolhe as sementes dos clusters com NMS por classe em `iou_thr`,
atribui cada detecção à semente de mesmo rótulo com a qual tem o melhor IoU e
depois reduz cada cluster do mesmo jeito. As formas dos clusters nunca mudam no
meio da passada, então a op inteira é aritmética de tensores de forma fixa. As
duas variantes concordam sempre que os clusters são inequívocos e podem divergir
um pouco em cadeias de clusters sobrepostos.

`nms_fusion` mantém o box de maior confiança de cada grupo sobreposto, sem
alterá-lo. Os `weights` por modelo escalam as confianças apenas para o ranking da
supressão, e os boxes que sobrevivem mantêm as pontuações originais.

## Fusão personalizada

`fusion=` também aceita um callable com a mesma assinatura das ops acima. O nome
dele fica registrado em `ens.fusion`, ou `"custom"` quando ele não tem nome. O
retorno é validado: precisa ser uma tripla `(boxes, scores, labels)` com formas
coerentes.
