---
title: Matriz completa de exportação
seo_title: Matriz de suporte a exportação do LibreYOLO e suas regras
description: >-
  Como o LibreYOLO decide se uma combinação de família, tarefa e formato
  exporta: os doze formatos, os três níveis, as regras de fallback e os limiares
  de paridade.
lead: >-
  O suporte a exportação é uma consulta na tripla (família, tarefa, formato).
  Esta página descreve o formato dessa matriz, as regras que preenchem as
  células que nenhuma entrada explícita cobre, e como consultá-la para uma
  combinação que te interessa.
keywords:
  - suporte a exportação libreyolo
  - matriz de exportação
  - onnx tensorrt openvino tflite
  - comando libreyolo formats
  - limiar de paridade de exportação
  - NotImplementedError export
last_verified: 1.5.0
verification: >-
  Formatos, níveis, ordem de fallback, bloqueios por tarefa e por família e
  bloqueios de NCNN lidos de libreyolo/export/support.py; aliases e argumentos
  compartilhados de libreyolo/export/exporter.py; definições dos níveis de
  docs/adr/0011-export-support-tiers.md; limiares de paridade de
  docs/export_support.md, tudo na v1.5.0. As células por combinação não são
  transcritas aqui; você as consulta com o snippet abaixo.
snippets:
  usage:
    - label: 'Consultar a matriz, sem precisar de modelo'
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Exportar e ler uma rejeição
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # Verifique antes de chamar: uma combinação bloqueada levanta erro no
        # preflight e a mensagem carrega este motivo.
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Formato da matriz

A matriz é indexada por `(family, task, format)`. As chaves de família são os
nomes canônicos do registro de modelos, as chaves de tarefa vêm de
`libreyolo.tasks.TASKS`, e há doze formatos:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` aceita ainda dois aliases: `engine` para
`tensorrt`, e `litert` para `tflite`, que é o nome atual do TensorFlow Lite. O
formato e o sufixo `.tflite` continuam os mesmos.

<code-tabs name="usage" />

Como uma célula é função de três chaves, a grade completa é grande e muda a
cada versão. Ela é gerada em vez de escrita à mão, e fica em
`docs/export_support.md` no repositório da biblioteca. Consulte a matriz pelo
Python ou pela CLI em vez de ler uma cópia.

## Os três níveis

| Nível | Significado |
|---|---|
| `validated` | A paridade numérica é coberta no CI ou em uma execução noturna documentada |
| `available` | A conversão está implementada, mas não há evidência registrada de paridade numérica em runtime |
| `blocked` | O preflight levanta `NotImplementedError` com um motivo antes do tracing |

Combinações validated e available seguem em frente sem confirmação nem aviso
genérico. A evidência registrada e as restrições delas continuam visíveis na
documentação gerada. Uma combinação blocked falha antes das verificações de
dependências, do carregamento da calibração, do tracing ou da criação do
artefato.

Adicionar uma entrada validated exige um teste de paridade e um campo `since`.

Um `SupportEntry` carrega quatro campos: `tier`, uma string `reason`, a versão
`since` e uma string `constraint`. A restrição é a parte que importa na hora de
integrar: uma marca de confirmação só vale sob as condições que ela nomeia, que
costumam ser um canvas de entrada fixo, batch 1, FP32 e uma versão específica do
runtime.

## Como uma célula é decidida

`get_support(family, task, fmt)` resolve nesta ordem. A primeira regra que casar
vence.

1. Uma tarefa desconhecida, ou um formato fora dos doze, retorna `blocked`.
2. Uma entrada explícita `(family, task, format)` retorna como foi registrada.
3. Um bloqueio para a família inteira retorna `blocked` com o motivo daquela família.
4. Um bloqueio para a tarefa inteira retorna `blocked` com o motivo daquela tarefa.
5. Para `ncnn`, uma família na lista de bloqueio do NCNN retorna `blocked`.
6. `mnn` retorna `blocked`: não há contrato de runtime para essa família e tarefa.
7. `rknn` retorna `blocked`. Nesta versão, o RKNN se limita exatamente às variantes de detecção testadas no simulador: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s e PicoDet-s no RK3588.
8. `tensorrt` e `openvino` retornam `available`: o caminho de conversão existe, mas a paridade em runtime não foi registrada para essa família e tarefa.
9. `tflite`, `paddle`, `coreai` e `coreml` retornam `blocked`, cada um com seu próprio motivo.
10. Todo o resto retorna `available`: a conversão está implementada, a paridade numérica em runtime não está registrada.

A assimetria dos passos 8 a 10 é proposital. TensorRT e OpenVINO convertem
genericamente a partir do ONNX, então vale a pena tentar uma combinação não
listada. TFLite, Paddle, Core AI e CoreML precisam cada um de um caminho por
família, então uma combinação não listada é uma rejeição, e não um convite.

## Tarefas bloqueadas

Estas tarefas são bloqueadas para qualquer família sem entrada explícita.

| Tarefa | Motivo |
|---|---|
| `ocr` | Duas redes com recorte dinâmico por região não cabem no contrato de exportação de grafo único |
| `point` | A família não está conectada ao contrato compartilhado de mapa de calor de pontos e de decodificação de picos no backend |
| `semantic` | A família não está conectada ao contrato compartilhado de logits densos e de argmax no backend |
| `mesh` | As saídas do grafo de malha corporal, os metadados e o contrato de runtime não estão definidos |
| `normal` | A família não está conectada ao contrato de normais unitárias densas em canvas fixo e de renormalização no backend |
| `panoptic` | A exportação panóptica não tem contrato de runtime no backend |
| `gaze` | A família não está conectada ao contrato compartilhado de logits de duas cabeças e de decodificação por valor esperado no backend |

Uma entrada explícita sobrepõe isso, que é como, por exemplo, uma família
semantic já conectada ainda exporta.

## Famílias bloqueadas

| Família | Bloqueada para |
|---|---|
| `depth_anything3` | Todos os formatos; seu grafo de profundidade não está no contrato de runtime exportado |
| `domedetr` | Todos os formatos. O PAQI define a quantidade de queries por imagem, então um grafo traçado só é válido para a imagem em que foi traçado. Use o D-FINE para um DETR exportável |
| `eomt` | Exportação de instâncias e panóptica, que não têm parsing em runtime |
| `l2cs` | Qualquer coisa fora de ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO |
| `hrnet` | Qualquer coisa fora de ONNX, TorchScript, OpenVINO e TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Todos os formatos; a exportação de modelos promptable está fora do escopo do contrato de runtime da v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Todos os formatos; a exportação de runtime de vocabulário aberto está fora do escopo da v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Todos os formatos; a exportação de VLM generativa está fora do escopo da v1 |

O PicoSAM3 é a exceção no nível promptable: ele exporta para ONNX sua rede de
ROI bruta de 96 pixels.

## Bloqueadas para o NCNN

Decodificadores no estilo DETR precisam de operações de amostragem que o NCNN
não implementa, então estas famílias são bloqueadas para `ncnn`, a não ser que
uma entrada explícita diga o contrário: Deformable DETR, DETR, DINO-DETR,
D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR e EC. A
rejeição aponta ONNX, OpenVINO, TorchScript e TensorRT como alternativas.

## Limiares de paridade

Uma célula validated significa que o artefato exportado reproduziu o modelo
nativo dentro destes limites:

| Grupo de tarefas | Limiar |
|---|---|
| Detecção e OBB | IoU das caixas correspondentes acima de 0.95, MAE do score abaixo de 0.01 |
| Segmentação e panóptica | IoU de máscara acima de 0.95 |
| Pose | L2 dos keypoints abaixo de 2 pixels na resolução nativa |
| Classificação | Cosseno dos logits acima de 0.999 e mesma classe top-1 |
| Profundidade e restauração | PSNR acima de 40 dB em relação à saída nativa |
| Normais de superfície | Erro angular médio abaixo de 0.1 grau |
| Ponto | Localizações dos picos iguais dentro de uma célula de saída |

As linhas de query do DETR são um conjunto sem ordem, então a paridade das
famílias DETR alinha as linhas de query como conjunto, e não por posição.

## Exportando

<code-tabs name="export" />

Uma combinação blocked levanta `NotImplementedError` no preflight e a mensagem
carrega o motivo registrado. `validated_alternatives(family, task)` retorna os
formatos que estão validated para esse par, que é a informação útil de imprimir
ao lado de uma rejeição.

Os argumentos que todos os exportadores compartilham estão listados na
[página da API do modelo](/docs/reference/model-api). Os argumentos específicos
de cada formato ficam nas páginas de cada formato.

## Lendo uma restrição

Uma célula validated é uma afirmação sobre uma configuração medida, não sobre o
formato em geral. Uma string de restrição como
`FP32, batch 1, fixed 520x520 input` significa que a paridade foi registrada
naquela forma e naquela precisão. Exportar em outra resolução ou com outro
tamanho de batch ainda produz um artefato; só não é a configuração de onde o
número veio.
