---
title: TFLite
seo_title: "Exportar para TFLite (LiteRT) a partir do LibreYOLO"
description: "Exporte um modelo LibreYOLO para um FlatBuffer .tflite através do onnx2tf: formas estáticas, somente FP32, entradas NHWC e as famílias que convertem sem problema."
lead: "TFLite é o formato FlatBuffer que o LiteRT executa em alvos móveis e embarcados. O LibreYOLO exporta um grafo ONNX estático, converte esse grafo com o onnx2tf em modo flatbuffer-direct e escreve os metadados do modelo ao lado do artefato como um sidecar JSON."
keywords:
  - exportar yolo tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - flatbuffer tflite
  - entrada nhwc tflite
  - inferência em edge
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="tflite")'
    mono: true
  - label: Escreve
    value: "Um arquivo .tflite mais um sidecar de metadados .tflite.json"
  - label: Extra
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Recarrega com
    value: 'LibreYOLO("weights/LibreYOLO9t.tflite")'
    mono: true
  - label: Formas
    value: "Somente estáticas. dynamic=True é rejeitado."
  - label: Precisão
    value: "Somente FP32. half=True e int8=True são rejeitados."
  - label: Requer
    value: "Python 3.12 ou superior, porque o onnx2tf 2.4.x não publica wheels mais antigos"
verification: "Lido de libreyolo/export/tflite.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/tflite.py e pyproject.toml no branch dev."
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        # LiteRT é o nome atual do Google para o TensorFlow Lite. Os dois extras
        # instalam o mesmo toolchain e produzem a mesma saída .tflite.
        pip install "libreyolo[tflite]"
    - label: Confirmar a versão do Python antes
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.tflite e weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" é aceito como alias e resolve para o mesmo exportador.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, ou (altura, largura)
            batch=1,
            simplify=True,    # onnxsim sobre o intermediário ONNX
            output_path=None, # None escreve weights/<stem>.tflite
            verbose=False,    # True transmite o log do onnx2tf
        )

        # dynamic=True levanta ValueError: o conversor precisa de formas estáticas.
        # half=True e int8=True são rejeitados antes do tracing.
  run:
    - label: Pelo LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT puro
      language: python
      code: |
        import json

        import numpy as np
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")
        interpreter.allocate_tensors()
        detail = interpreter.get_input_details()[0]
        print(detail["shape"], detail["dtype"])   # NHWC, não NCHW

        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], np.float32))
        interpreter.invoke()
        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Os nomes das classes, a tarefa e o tamanho de entrada ficam no sidecar.
        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))
        print(meta["model_family"], meta["task"], meta["names"])

        # O pré-processamento, a transposição de NCHW para NHWC e o pós-processamento são seus.
  support:
    - label: Conferir uma família e uma tarefa antes de exportar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

<code-tabs name="install" />

O extra traz o `onnx2tf` para a conversão e o `ai-edge-litert` para rodar o
resultado, ambos atrás de um marcador de Python 3.12. Em um interpretador mais
antigo a exportação levanta um `ImportError` que nomeia o requisito de versão em
vez de falhar dentro do conversor.

`libreyolo[litert]` instala exatamente a mesma coisa. A string de formato `litert`
é um alias para `tflite`, e o arquivo de saída é um `.tflite` de qualquer jeito.

## Exportação

<code-tabs name="export" />

A família e a tarefa são conferidas antes de qualquer outra coisa acontecer, então
uma combinação sem suporte falha imediatamente com o erro específico de conversor
ou de runtime que a deixou de fora, não com uma mensagem genérica. A conversão em
si é uma chamada de subprocesso ao `onnx2tf` em modo `flatbuffer_direct` sobre um
intermediário ONNX estático.

Os metadados ficam em um sidecar. `weights/LibreYOLO9t.tflite.json` carrega a
família, a tarefa, os nomes das classes, o tamanho de entrada e o schema de pose;
o próprio FlatBuffer não tem campo de metadados do LibreYOLO, então os dois
arquivos andam juntos.

## Rodar o artefato

<code-tabs name="run" />

`LibreYOLO()` despacha pelo sufixo `.tflite` e devolve o mesmo objeto `Results`
que o checkpoint. O backend lê o sidecar, transpõe o blob NCHW para NHWC quando o
interpretador pede uma entrada com os canais no final, aplica a escala de
quantização e o zero point do interpretador onde eles existem, e transpõe as
saídas de volta para o layout que o pós-processamento do LibreYOLO espera.

O segundo snippet é o caminho do runtime puro. O pré-processamento, a transposição
de layout, a decodificação, o NMS e o reescalonamento de coordenadas passam a ser
seus ali, e o detalhe de layout é o mais fácil de deixar passar: o onnx2tf emite
entradas com os canais no final, então um blob com forma `(1, 3, 640, 640)` não
vai encaixar.

## Restrições

Somente formas estáticas. `dynamic=True` levanta `ValueError` antes do tracing, e
o canvas de exportação fica fixo no valor para o qual `imgsz` foi resolvido.

Somente FP32. `half=True` e `int8=True` são ambos rejeitados durante a validação,
então o deploy quantizado não é alcançável por este exportador hoje.

A cobertura aqui é mais estreita do que a dos formatos de grafo, e é decidida por
medição em vez de por família. As combinações validadas incluem detecção YOLO9,
YOLOX e YOLO-NAS, segmentação semântica PIDNet, as quatro famílias de classificação
CNN, embedding DINOv2 e SigLIP2, classificação SigLIP2, bordas TEED e DexiNed, e
restauração Real-ESRGAN e SwinIR. O SwinIR carrega uma ressalva extra: a paridade
se mantém quando as dimensões da fonte batem exatamente com o canvas de exportação,
e fontes menores recebem padding até o canvas antes de o transformer rodar, o que
pode divergir da inferência nativa de tamanho variável.

As entradas bloqueadas nomeiam a falha exata, o que vale a pena ler antes de
tentar uma solução alternativa. Alguns exemplos: a detecção RF-DETR converte no
canvas nativo de 384, mas o LiteRT não consegue alocá-la porque o `STRIDED_SLICE`
recebe uma entrada acima do rank 5-D que ele suporta; o PicoDet é recusado porque
um `RESHAPE` mapeia 19.200 elementos de entrada para 9.600 elementos de saída; o
D-FINE derruba o conversor no tratamento de formas do `GatherElements`; o RTMDet
exporta e recarrega com a paridade em cru intacta, mas os boxes públicos caem para
0.911 IoU com 29.9 px de deriva de coordenadas.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação só,
incluindo a string de motivo por trás de um bloqueio:

<code-tabs name="support" />
