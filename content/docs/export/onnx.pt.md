---
title: ONNX
seo_title: "Exportar para ONNX a partir do LibreYOLO"
description: "Exporte um modelo LibreYOLO para ONNX: o opset que o LibreYOLO escolhe por família, os eixos dinâmicos, o NMS embutido, INT8 e como o grafo é recarregado."
lead: "ONNX é um formato de grafo portátil. O LibreYOLO traça o modelo com torch.onnx.export, opcionalmente simplifica o grafo e escreve a família, a tarefa, os nomes das classes e o tamanho de entrada nos metadados do próprio arquivo, para que qualquer backend do LibreYOLO consiga reconstruir o pós-processamento."
keywords:
  - exportar yolo onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - eixos dinâmicos onnx
  - nms embutido onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="onnx")'
    mono: true
  - label: Escreve
    value: "Um arquivo .onnx, com os metadados embutidos no grafo"
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.onnx")'
    mono: true
  - label: Formas
    value: "Batch dinâmico por padrão no Python; exceções por tarefa abaixo"
  - label: Precisão
    value: "FP32, FP16 (half=True), INT8 (int8=True, detecção YOLO9)"
verification: "Lido de libreyolo/export/onnx.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/onnx.py e libreyolo/cli/commands/export.py no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, ou (altura, largura)
            batch=1,
            dynamic=True,     # padrão no Python; o CLI usa False
            simplify=True,    # roda o onnxsim sobre o grafo
            opset=None,       # None escolhe 13, ou 17 para famílias estilo DETR
            half=False,       # pesos e ativações em FP16
            int8=False,       # QDQ INT8, somente detecção YOLO9
            data=None,        # data.yaml de calibração, só para INT8
            device=None,      # dispositivo do trace; None usa o do modelo
            output_path=None, # None escreve weights/<stem>.onnx
        )
  nms:
    - label: Embutir o NMS no grafo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Somente detecção YOLO9, batch 1. dynamic é forçado para False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 com dados de calibração
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # algumas centenas de imagens representativas
            fraction=1.0,
        )
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime puro
      language: python
      code: |
        import numpy as np
        import onnx
        import onnxruntime as ort

        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )

        # O pré-processamento e o pós-processamento ficam por sua conta neste caminho.
        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        print([out.shape for out in outputs])

        # O grafo carrega a família, a tarefa, os nomes das classes e o tamanho de entrada.
        meta = {p.key: p.value for p in onnx.load("weights/LibreYOLO9t.onnx").metadata_props}
        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Conferir uma família e tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

O extra puxa `onnx`, `onnxsim` e `onnxruntime`. Só o `onnx` já basta para
escrever o arquivo; o `onnxsim` roda o passo de simplificação e o `onnxruntime`
executa o artefato e faz a calibração INT8.

## Exportação

<code-tabs name="export" />

Sem `output_path`, o arquivo vai parar em `weights/` com o nome-base do
checkpoint, e com `_fp16` ou `_int8` acrescentado quando essa precisão foi
pedida.

`dynamic` tem `True` como padrão no Python e `False` no CLI. Quando está ligado,
o eixo de batch vira simbólico e algumas tarefas abrem ainda mais: a segmentação
semântica também abre a altura e a largura da máscara, a restauração com
Real-ESRGAN abre os eixos espaciais, e os detectores de dois estágios mantêm a
altura e a largura de origem dinâmicas porque o redimensionamento deles acontece
dentro do grafo.

O `opset` é escolhido por família quando você omite. As famílias estilo DETR
(`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) mais `deit`, `midas` e
`moge2` recebem o opset 17, que é onde o `aten::scaled_dot_product` é rebaixado.
Todo o resto recebe 13. O matting sobe para 19 de qualquer jeito, porque o
decodificador do BiRefNet precisa do operador `DeformConv`, que o ONNX define a
partir do opset 19.

`simplify=True` roda o `onnxsim` e mantém o grafo original se o passo falhar,
então um erro de simplificação é um aviso, e não uma falha de exportação. No
macOS arm64 com `onnx` 1.22 ou mais novo e `onnxsim` 0.6.5 ou mais antigo o passo
é pulado por completo, porque essa combinação pode abortar o processo Python.

### NMS embutido

<code-tabs name="nms" />

`nms=True` vale só para detecção YOLO9 e exige batch 1; pedi-lo junto com
`dynamic=True` registra um aviso e desliga o modo dinâmico. O grafo então tem
duas saídas: `output`, de forma `(batch, max_det, 6)`, e `raw`, o tensor não
decodificado do detector que o próprio backend do LibreYOLO usa para que o
pós-processamento continue idêntico ao do caminho PyTorch.

### DeepStream

`deepstream=True` é uma opção exclusiva do ONNX. Ela exporta o grafo no layout
que o parser do NVIDIA DeepStream espera e escreve dois arquivos auxiliares ao
lado dele, `config_infer_primary_<stem>.txt` e `<stem>_labels.txt`, de modo que o
artefato entra em um pipeline sem configuração escrita à mão.

Ela é mutuamente exclusiva com `nms=True`, e pedir as duas levanta um
`ValueError`: o DeepStream faz a supressão no próprio estágio de clustering.
Passá-la para qualquer formato que não seja ONNX também levanta erro. Veja
[DeepStream](/docs/export/deepstream) para a tabela de famílias e tarefas
suportadas e para a compilação do parser.

### INT8

<code-tabs name="int8" />

`int8=True` roda a quantização estática do ONNX Runtime e escreve um grafo QDQ
com entradas e saídas em float32. Só os nós `Conv` e `Gemm` são quantizados.
Deixar a decodificação da cabeça de detecção em float32 é proposital: essa
concatenação mistura coordenadas de caixa na escala de pixels com scores de
classe na faixa de 0 a 1, e uma única escala de ativação por tensor dominada pela
magnitude das caixas levaria todos os scores a zero.

Essa flag hoje se aplica somente à detecção YOLO9, e qualquer outra coisa levanta
`NotImplementedError` na verificação prévia. Omitir `data` cai de volta em
`coco8.yaml` com um aviso; oito imagens não são um conjunto de calibração
representativo. Um modelo que já foi quantizado no PyTorch segue outro caminho,
descrito em [Quantização](/docs/export/quantization).

## Executar o artefato

<code-tabs name="run" />

`LibreYOLO()` despacha pelo sufixo `.onnx` e devolve o mesmo objeto `Results` que
um checkpoint `.pt`, porque os nomes das classes, a tarefa, o tamanho de entrada
e o esquema de pose foram escritos no `metadata_props` do grafo na hora da
exportação. Com `device="auto"` a sessão pega o `CUDAExecutionProvider` quando o
ONNX Runtime informa que ele existe, e cai para a CPU caso contrário.

O segundo snippet é para quem não tem o LibreYOLO instalado. Pré-processamento,
decodificação, NMS e reescalonamento de coordenadas ficam todos por sua conta
nesse caminho; o bloco de metadados continua lá para ser lido.

## Restrições

Os nomes dos tensores de saída são fixos por tarefa, e é com eles que o consumidor
sem metadados tem que casar:

| Tarefa | Nomes das saídas |
|---|---|
| Detecção, cabeças de grade e de âncoras | `output` |
| Detecção, estilo DETR | `pred_logits`, `pred_boxes` |
| Detecção, RF-DETR | `dets`, `labels` |
| Classificação | `output` |
| Segmentação semântica | `semantic_logits` |
| Profundidade | `depth` |
| Normal de superfície | `normal` |
| Bordas | `edges` |
| Restauração | `restored` |
| Matting | `matte` |
| Olhar | `yaw_logits`, `pitch_logits` |

O RF-DETR é também a única família cujo tensor de entrada se chama `input` em vez
de `images`.

Várias tarefas carregam nesta versão um contrato de resolução fixa em runtime.
Profundidade, normal de superfície e bordas rejeitam `batch != 1` e forçam
`dynamic=False`. O matting força o quadrado nativo de 1024, porque as tabelas de
posição relativa do Swin do BiRefNet estão presas à resolução delas. A
restauração força uma tela fixa para todas as famílias, exceto a Real-ESRGAN,
cujo gerador é totalmente convolucional.

Um `imgsz` retangular funciona para as famílias YOLO9, HRNet, NAFNet e
Real-ESRGAN. As famílias com contrato de quadrado fixo (`clip`,
`deformable_detr`, `detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`)
o rejeitam de saída.

Duas combinações são recusadas antes do trace: segmentação YOLO9, porque o YOLO9
é só detecção no LibreYOLO, e segmentação RTMDet-Ins, cuja decodificação de
máscaras com kernels dinâmicos não tem contrato de runtime exportado.

Para a tabela completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação
específica, pergunte direto à biblioteca:

<code-tabs name="support" />
