---
title: Hailo
seo_title: Rodar modelos LibreYOLO em aceleradores Hailo
description: >-
  Faça o deploy de um modelo LibreYOLO em um Hailo-8 ou Hailo-8L: a exportação
  ONNX estática, a etapa do Dataflow Compiler que você mesmo roda, e quais
  arquiteturas compilam.
lead: >-
  Os aceleradores Hailo são compilados com o Hailo Dataflow Compiler, um SDK
  proprietário distribuído pela Developer Zone da Hailo. A parte do fluxo que
  cabe ao LibreYOLO é uma exportação ONNX estática e simples; o parsing, a
  quantização e a compilação para um HEF acontecem depois, dentro do DFC.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - compilar hef hailo
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Passo do LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Não é um formato
    value: Não existe format="hef". O DFC não pode ser uma dependência do pip.
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Host de compilação
    value: >-
      Linux x86_64, incluindo WSL2 com Ubuntu 22.04. A compilação não pode rodar
      em ARM.
  - label: Compila
    value: >-
      Grafos de CNN pura e forma fixa. Atenção, formas dinâmicas e designs
      dominados por LayerNorm, não.
  - label: Status
    value: >-
      Nenhuma família do LibreYOLO foi levada de ponta a ponta pelo DFC até um
      HEF em funcionamento.
verification: >-
  Lido de skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py e
  libreyolo/cli/commands/export.py no branch dev. As restrições do DFC são as
  registradas nessa skill; nenhum HEF do LibreYOLO foi compilado nem medido.
snippets:
  install:
    - label: Lado LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Lado Hailo, instalado por você'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # O Hailo precisa de batch 1, uma resolução fixa e nenhum eixo dinâmico.

        # A API Python usa dynamic=True por padrão, então desligue
        explicitamente.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # A CLI já usa formas estáticas por padrão.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Confirme que o grafo é estático antes de compilar
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Fazer parse, quantizar e compilar'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Para o YOLOX, faça o translate uma vez sem end_node_names: o log do
        DFC

        # imprime os nós finais que ele sugere. Rode de novo com esses nós.

        runner.translate_onnx_model(ONNX)


        # A normalização precisa bater com o pré-processamento do LibreYOLO.
        YOLOX

        # e YOLO9 não precisam de média nem de desvio padrão, só da escala de

        # 0-255 para 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Opcional: deixe o Hailo cuidar do NMS. A configuração é específica
        tanto

        # da contagem de classes quanto do tamanho de entrada, então uma config
        de

        # COCO-80 está errada para um modelo de três classes com fine-tuning.
        Sem

        # esta linha o HEF emite os tensores brutos da cabeça e a aplicação os

        # decodifica.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # As imagens de calibração precisam ser representativas dos dados do
        deploy.

        # Imagens aleatórias compilam e destroem a acurácia silenciosamente.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Nós finais do YOLO9
      language: python
      code: >
        # Os grafos do LibreYOLO usam o prefixo "/head/...", não o prefixo

        # "model.N" visto em configurações escritas para outras exportações. Uma

        # config copiada não vai bater. Confirme os nomes no seu próprio grafo
        se

        # o parsing falhar.

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 com o AI Kit ou o AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # checa o dispositivo, e diz qual é
        a arch

        hailortcli run libreyoloxs.hef       # smoke test e throughput
source_hash: 33b077f1c23d5535
---

## Instalação

No LibreYOLO não existe `format="hef"` e nem vai existir. O Hailo Dataflow
Compiler é um SDK proprietário distribuído como um wheel privado atrás do cadastro
na Developer Zone, então ele não pode ser uma dependência nem um extra. O deploy
tem duas etapas: o LibreYOLO escreve um arquivo ONNX estático, e você roda o DFC
em cima dele.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Exportação

<code-tabs name="export" />

Não passe `half=True`. O DFC ingere ONNX em FP32 e faz sua própria quantização
INT8. Também não passe `nms=True`: ou o Hailo cuida do NMS através de
`nms_postprocess`, ou a aplicação cuida, e um subgrafo de NMS é peso morto depois
dos nós finais. O opset padrão funciona; se o parser do DFC reclamar, reexporte
com `opset=11`.

O DFC corta o grafo nos nós finais que você informa, que são as convoluções da
cabeça de detecção, e descarta tudo o que vem depois. O ONNX decodificado comum do
LibreYOLO é, portanto, uma entrada aceitável: a cauda de decodificação é
simplesmente ignorada pelo parser.

## Compilação

<code-tabs name="compile" />

Escolha o `hw_arch` do alvo: `hailo8` para o Hailo-8, o AI HAT+ de 26 TOPS e os
módulos M.2 e PCIe; `hailo8l` para o Hailo-8L, o Raspberry Pi AI Kit e o AI HAT+
de 13 TOPS; `hailo10h` para o Hailo-10H, que precisa de um DFC e de um Model Zoo
mais novos e compatíveis. `hailortcli fw-control identify` no dispositivo responde
a essa pergunta quando você está na dúvida.

Duas famílias mapeiam para uma meta-arquitetura de NMS do HailoRT, de modo que o
Hailo pode cuidar da supressão dentro do pipeline compilado: o YOLOX através de
`meta_arch=yolox`, e o YOLO9 através da meta-arquitetura de cabeça desacoplada da
Hailo, cujo layout de cabeça é idêntico. Pegue a configuração `nms_postprocess`
correspondente no Hailo Model Zoo e ajuste para a sua contagem de classes e o seu
tamanho de entrada. Todo outro detector convolucional compila como um grafo sem
meta-arquitetura correspondente: o HEF emite os tensores brutos da cabeça e a
aplicação roda a decodificação e o NMS na CPU.

Guarde o log de compilação quando algo falhar. Toda correção depende do nome exato
da camada ou do operador que falhou.

## Rodar o artefato

<code-tabs name="device" />

A inferência da aplicação usa a API Python `hailo_platform`. Com o
`nms_postprocess` compilado junto, a saída é `(batch, num_classes, max_dets, 5)`
carregando `[y1, x1, y2, x2, score]` em coordenadas do modelo, que você mesmo
escala de volta para a imagem de origem. O pipeline `Results` do LibreYOLO não
entra em jogo em tempo de execução; o HEF é um artefato independente, e o
pré-processamento e o pós-processamento são da aplicação.

## Restrições

Se um modelo pode ou não ter o Hailo-8 ou o Hailo-8L como alvo é uma propriedade
da sua arquitetura, não do seu nome, então a regra abaixo vale para as famílias
adicionadas depois que esta página foi escrita.

Um modelo não vai compilar se contiver qualquer um destes:

- Atenção de qualquer tipo, self, cross, deformável ou por janelas. Isso descarta
  todo detector no estilo DETR, todo detector de vocabulário aberto ou
  condicionado a texto, todo backbone ViT, e toda torre de linguagem ou de
  visão-linguagem. O próprio zoo da Hailo traz alguns HEFs de transformer
  ajustados à mão; isso é trabalho sob medida do fornecedor e não é evidência de
  que um grafo de atenção qualquer compile.
- Formas dinâmicas ou fluxo de controle dependente dos dados. O DFC compila uma
  forma de entrada fixa e um grafo estático, então contagens variáveis de queries,
  prompts de texto, top-k dinâmico, `NonZero`, `Gather` ou `TopK` com índices
  dinâmicos, e `grid_sample` estão todos fora.
- Um design dominado por LayerNorm ou por GELU. O BatchNorm se funde às
  convoluções sem problemas; o suporte a LayerNorm é ruim e a GELU não é uma
  ativação nativa, então uma stack no estilo ConvNeXt é uma escolha ruim mesmo
  sendo nominalmente convolucional.
- Trabalho de imagem para imagem em resolução nativa. Os modelos de restauração
  rodam na resolução plena de entrada e estouram os orçamentos práticos de SRAM do
  Hailo.

Uma família é candidata quando é só convolução, usa BatchNorm com ReLU ou SiLU, e
tem tamanho de entrada fixo. Nesta biblioteca isso significa os detectores CNN de
estágio único, com YOLOX e YOLO9 como alvos principais; outros detectores
convolucionais como PicoDet, YOLO-NAS e RTMDet, com decodificação do lado da
aplicação; os classificadores CNN ResNet, MobileNetV4-conv e EfficientNetV2, dos
quais o ResNet é o mais bem suportado porque o Model Zoo da Hailo traz receitas
para ele; e cabeças de tarefa convolucionais pequenas, como a detecção por pontos
do FOMO e o gaze L2CS sobre um backbone ResNet, que são compiláveis em princípio
mas não têm receita da Hailo.

Uma ressalva de status, que é o motivo de nada nesta página ser apresentado como
suportado: nenhuma família do LibreYOLO foi levada de ponta a ponta pelo DFC até
um HEF em funcionamento. As regras acima preveem a compilabilidade a partir da
arquitetura. O comportamento do parser, a quantização e a acurácia continuam sem
comprovação até que um HEF seja compilado e medido, então trate cada candidato
como algo que exige sua própria evidência registrada: um HEF compilado a partir do
checkpoint exato, com as versões de DFC, Model Zoo e HailoRT registradas,
calibração documentada, e uma comparação de acurácia no dispositivo contra a
baseline FP32, em vez de um número de throughput.

Se o modelo for desqualificado, as alternativas são os runtimes com paridade
registrada: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) e
[OpenVINO](/docs/export/openvino).
