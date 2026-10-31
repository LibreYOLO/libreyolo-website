---
title: Core AI
seo_title: Exportar para o Apple Core AI a partir do LibreYOLO
description: >-
  Exporte um modelo LibreYOLO para um asset .aimodel do Apple Core AI: somente
  macOS, canvas fixo, FP32 e o contrato de ordenação das saídas nomeadas que os
  consumidores precisam respeitar.
lead: >-
  Core AI é a stack de inferência no dispositivo da Apple. O LibreYOLO captura o
  modelo com torch.export, faz o lowering pelo conversor do Core AI e escreve um
  asset .aimodel que carrega os metadados do modelo e os nomes das saídas
  exportadas.
keywords:
  - exportar libreyolo core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - inferência no dispositivo apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Escreve
    value: Um asset .aimodel com os metadados anexados
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Recarrega
    value: Não pelo LibreYOLO. Os consumidores usam o runtime do Core AI diretamente.
  - label: Formas
    value: Canvas fixo. dynamic=True levanta NotImplementedError.
  - label: Precisão
    value: Somente FP32. half=True e int8=True são rejeitados.
  - label: Requer
    value: >-
      macOS. A toolchain não converte nem roda em outro lugar, e coreai-torch
      fixa o torch em 2.11.x.
verification: >-
  Lido de libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py e pyproject.toml no
  branch dev.
snippets:
  install:
    - label: 'Instalação, no macOS'
      language: bash
      code: |
        # Fica fora de todos os extras agregados de propósito: coreai-torch fixa
        # o torch em 2.11.x e arrastaria o ambiente inteiro para essa versão.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, ou (altura, largura); este é o canvas de execução
            batch=1,
            output_path=None, # None escreve weights/<stem>.aimodel
        )

        # dynamic=True levanta NotImplementedError.
        # half=True e int8=True são rejeitados durante a validação.
  outputs:
    - label: Leia a ordem das saídas antes de ligar um consumidor
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")

        model.export(format="coreai", imgsz=640)


        # Os metadados do asset registram os nomes das saídas exportadas, na

        # ordem do grafo, sob "coreai_output_names". Mapeie por nome o
        dicionário

        # devolvido pelo Core AI usando essa lista; nunca o pareie por posição

        # com a tupla do modo eager.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Instalação

Este formato é somente macOS. O requisito `coreai-torch` carrega um marcador
`sys_platform == 'darwin'`, e a toolchain não converte nem roda em nenhum outro
lugar.

<code-tabs name="install" />

O extra fica fora de todos os extras agregados, incluindo `libreyolo[all]`,
porque `coreai-torch` fixa o torch na série 2.11. Instale em um ambiente que você
esteja disposto a restringir a esse par.

## Exportação

<code-tabs name="export" />

A captura é `torch.export`, uma captura de grafo de verdade, com guards, e não um
único trace gravado. Isso é mais rígido que o caminho do Core ML: leituras de
escalares no host e controle de fluxo dependente de dados são rejeitados em vez de
ficarem embutidos em silêncio, e é por isso que algumas famílias aparecem
bloqueadas aqui com uma falha de captura registrada.

Três etapas de preparação rodam dentro de um escopo que restaura o modelo vivo de
quem chamou, tendo a exportação sucesso ou não. As famílias derivadas do Darknet
têm a batch normalization de inferência fundida exatamente nas convoluções
anteriores, porque o Core AI 0.4.1 não preserva a fórmula do Darknet com o epsilon
depois da raiz quadrada. As famílias de grid e de âncoras têm suas âncoras
congeladas para o canvas fixo. O RF-DETR tem seu position embedding refeito
(rebake) para o canvas solicitado, rodando de novo o caminho de bake do próprio
modelo, porque o conversor não tem lowering para
`aten._upsample_bicubic2d_aa`.

O lowering incorpora à tabela de decomposições a decomposição de referência do
PyTorch para `aten.grid_sampler_2d`, já que o conversor do Core AI não tem
lowering para o sampler de deformable attention que as famílias DETR usam.

Os assets declaram um SO mínimo de v27, que é o único valor que a toolchain
oferece. Isso limita o deploy, não a conversão: a conversão e a execução pelo lado
do Python funcionam em versões anteriores do macOS graças ao runtime que vem
dentro do wheel, mas os números diferem entre versões do SO, então a paridade
registrada é medida no macOS 27.

## Executar o artefato

Não existe entrada de Core AI em `libreyolo/backends`, então `LibreYOLO()` não
carrega um `.aimodel`. Os consumidores usam o runtime do Core AI diretamente, e o
pré-processamento, a decodificação, o NMS e o reescalonamento de coordenadas ficam
por conta deles. Uma linha validada na matriz de suporte afirma que o grafo
exportado calcula os mesmos números que a referência, não que o `predict` vai
rodá-lo.

A única coisa que um consumidor não consegue deduzir sozinho é a ordem das saídas:

<code-tabs name="outputs" />

O Core AI devolve um dicionário nomeado cuja ordem de chaves não bate nem com a
ordem da tupla do forward em modo eager nem com nada adivinhável. Os nomes
exportados são gravados nos metadados do asset como `coreai_output_names`
exatamente por esse motivo. Mapeie por nome.

## Restrições

Canvas fixo, FP32, batch como foi exportado. `dynamic=True` levanta
`NotImplementedError`, e `half=True` e `int8=True` são rejeitados durante a
validação.

A cobertura é ampla do lado da conversão. As combinações validadas incluem as
famílias YOLO9, YOLOX, YOLO7, os quatro detectores da era Darknet, YOLO-NAS,
PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC e a
detecção RF-DETR; as quatro famílias de classificação CNN mais CLIP e SigLIP2 com
classes congeladas; Depth Anything V2 e ZipDepth; a restauração com NAFNet e
Real-ESRGAN; a segmentação semântica com PIDNet e LingBotVision; e a detecção de
pontos FOMO. Cada uma carrega seu próprio contexto registrado, que o
`libreyolo formats` imprime.

Bloqueadas, com o motivo registrado por combinação:

| Combinação | Por quê |
|---|---|
| Segmentação semântica EoMT | A captura estrita falha com `GuardOnDataDependentSymNode`: algo no caminho das máscaras lê um valor de um tensor e ramifica a partir dele |
| Segmentação semântica SegFormer | O caminho de captura não foi avaliado, e os pesos publicados são não comerciais em qualquer formato |
| Olhar (gaze) L2CS | O próprio modelo suporta apenas ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO, o que é uma decisão do lado do modelo |
| Profundidade Depth Anything 3 | A família rejeita a exportação em todos os formatos |

O RF-DETR traz uma ressalva que vale a pena ler antes de comparar artefatos. A
paridade dele é registrada contra o grafo que o próprio exportador do Core AI
prepara, e não contra o ONNX, e em um canvas de 640 o artefato ONNX do RF-DETR
discorda desse grafo preparado. O rebake do Core AI preserva o redimensionamento
com antialiasing que o modelo faz em modo eager, enquanto o caminho do ONNX
desliga o antialiasing. Por isso o ONNX não é uma referência válida para essa
família em um canvas que não seja o nativo.

Para o formato anterior da Apple, veja [Core ML](/docs/export/coreml). Para a
grade completa de famílias e tarefas, veja [a matriz de
exportação](/docs/reference/export-matrix). Para uma combinação só:

<code-tabs name="support" />
