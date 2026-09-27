---
title: Instalação leve
seo_title: Rode inferência ONNX do LibreYOLO sem PyTorch
description: >-
  Instale o LibreYOLO com --no-deps e rode detecção ONNX só com numpy, sem torch
  no disco. A técnica, seus limites e a lista exata de pacotes.
lead: >-
  O caminho de inferência ONNX do LibreYOLO é numpy de ponta a ponta, incluindo
  o decode e o NMS. Nada nele precisa de PyTorch em tempo de execução, então uma
  instalação que pula a resolução de dependências consegue rodar detecção com o
  torch ausente da máquina.
keywords:
  - libreyolo sem pytorch
  - inferência onnx sem torch
  - instalação leve libreyolo
  - pip install no-deps
  - onnxruntime inferência python
  - libreyolo espaço em disco
  - detecção de objetos sem torch
  - torch cpu only
last_verified: 1.5.0
meta:
  - label: Aplica-se a
    value: 'Detecção ONNX, sete famílias de modelos'
  - label: Ponto de entrada
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Nível de suporte
    value: 'Best effort, não é uma distribuição separada'
snippets:
  install:
    - label: Leve
      language: bash
      code: |
        # Instala o pacote sem sua lista de dependências e depois fornece os
        # quatro pacotes que o caminho de detecção ONNX realmente importa.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Torch só de CPU
      language: bash
      code: |
        # Tente isso primeiro. Mantém todos os recursos e evita o wheel de
        # CUDA, que é onde a maior parte do disco vai embora.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # xyxy aqui é um ndarray do numpy, não um tensor do torch.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Por que isso funciona

`pip install --no-deps libreyolo` instala o pacote e pula inteiramente a lista
de dependências. Nada é resolvido em seu nome, e você fica responsável por
instalar o que de fato usa.

Isso só é útil se o caminho de código que você quer realmente não precisar das
dependências que você pulou, e no caso da detecção ONNX ele não precisa. O
decode, incluindo a non-maximum suppression, é numpy. As receitas de
pré-processamento são numpy. O PyTorch é uma dependência de treinamento e de
inferência eager, e nesse caminho ele nunca é chamado.

Antes desta versão o import falhava de qualquer jeito: importar qualquer coisa
sob `libreyolo.models` construía todas as classes de modelo para popular o
registro de detecção automática de checkpoint, e essas classes são subclasses
de `torch.nn.Module`. As receitas de pré-processamento agora vivem no próprio
pacote delas, `libreyolo.preprocess`, e o import do torch é adiado até que algo
toque em um atributo do torch, então o caminho ONNX importa com o torch ausente
da máquina. Esse pacote tem um pré-processador nativo em numpy por família:
`yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`, `dfine`, `deim` e
`deimv2`, duas a mais que as sete famílias verificadas de ponta a ponta abaixo.
Cada `libreyolo/models/<family>/utils.py` reexporta a partir dele, então os
caminhos de import existentes continuam funcionando.

## Tente primeiro o wheel só de CPU

A maioria das pessoas que pede isso quer evitar uma instalação de vários
gigabytes, e o tamanho está concentrado em um lugar: o wheel padrão do `torch`
embute o CUDA. Uma build só de CPU é uma fração disso e não precisa de nenhum
caminho de instalação especial.

<code-tabs name="install" />

A opção só de CPU mantém todos os recursos do LibreYOLO: treinamento,
validação, todas as tarefas, todas as famílias, a CLI. Escolha o caminho leve
quando quiser zero torch na máquina, não apenas menos torch.

## O que a instalação leve cobre

| | |
|---|---|
| Tarefa | Detecção |
| Formato | ONNX |
| Ponto de entrada | `OnnxBackend` |
| Interface | Biblioteca Python |

Sete famílias foram verificadas nesse caminho: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) e [DEIM](/docs/models/deim), contando junto as
variantes de cada família.

Esse é o escopo verificado, não um limite que a biblioteca imponha. Outras
tarefas e outras famílias estão simplesmente fora do que foi checado: algumas
vão puxar o torch quando você chamá-las, e umas poucas podem por acaso
funcionar. Trate qualquer coisa além desta lista como não testada, e não como
suportada ou quebrada.

Dentro dela, os resultados são idênticos aos da instalação normal, não apenas
próximos. Cada família foi exportada para ONNX e rodada duas vezes, uma vez
normalmente e outra com o torch bloqueado; os boxes, os scores e as classes
bateram exatamente. Um teste de paridade na suíte impede que esse contrato
saia do lugar.

## As cinco coisas que pegam as pessoas

**Use `OnnxBackend`, não as classes de modelo.** `LibreYOLO9("model.onnx")`
ainda exige o torch, porque `LibreYOLO9` é ela mesma uma subclasse de
`nn.Module`. Esse é o erro mais provável, já que todas as outras páginas destes
docs carregam um modelo pela classe dele ou por `LibreYOLO()`.

**Exporte em outro lugar.** Produzir o arquivo `.onnx` exige o torch, então a
máquina leve não consegue criar um. Exporte em uma máquina de desenvolvimento
ou de CI e mande o artefato para o alvo enxuto.

**Os resultados carregam arrays do numpy.** `result.boxes.xyxy` aqui é um
`ndarray`. Os containers aceitam qualquer um dos dois tipos, então os nomes dos
atributos não mudam, mas código que chama `.cpu()` ou `.numpy()` em um
resultado vai falhar.

**Uma única imagem retorna um único `Results`.** `predict()` retorna um
`Results` para uma imagem e uma lista para várias. Indexar um resultado único
com `[0]` seleciona a primeira detecção, não a primeira imagem, o que
silenciosamente te dá um resultado de uma caixa em vez de levantar um erro.

**A CLI não vai funcionar.** `typer` e `click` não estão nos quatro pacotes,
então o comando `libreyolo` fica indisponível. Esta é uma instalação de
biblioteca.

## Predict

<code-tabs name="predict" />

Troque `onnxruntime` por `onnxruntime-gpu` para rodar em CUDA. Os quatro
pacotes são os que um `predict()` completo sem torch realmente importa,
registrados durante a chamada em vez de deduzidos. `opencv-python-headless`
entra no lugar do `opencv-python` declarado: mesmo módulo, sem bibliotecas de
GUI, menor em disco.

Das dependências declaradas restantes, `requests` só é necessário para carregar
uma imagem a partir de uma URL, `pycocotools` e `scipy` são validação e
avaliação, e `typer` e `click` são a CLI.

## Esta lista vai sair do lugar, por design

A lista de pacotes acima está correta para a versão indicada no topo desta
página. `--no-deps` te tira da resolução de dependências, então nada checa isso
por você, e uma versão posterior pode importar algo que não está listado aqui.

Se você bater em um `ModuleNotFoundError`, você já entende a técnica: instale o
pacote que falta. Esse é o modelo de manutenção pretendido, e não um relatório
de bug. Este caminho é best effort e não é uma distribuição suportada em
separado, o que também é o motivo de não haver um segundo pacote leve no PyPI
nem plano para um.

Para confirmar que seu ambiente está realmente sem torch em vez de estar
caindo de volta em uma cópia instalada sem alarde, garanta isso com um assert:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Vale manter essa checagem no CI da imagem enxuta. Sem ela, um ambiente que por
acaso tenha o torch vai passar em todos os testes e não vai te dizer nada.
