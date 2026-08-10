---
title: Quantização
seo_title: "Quantizar um modelo LibreYOLO no PyTorch"
description: "A API de quantização do LibreYOLO em PyTorch: nove receitas, calibração separada dos dados de treinamento, QAT e QAD, e dois artefatos de deploy."
lead: "A quantização no LibreYOLO roda inteiramente em PyTorch: model.quantize() troca os módulos Conv2d e Linear do modelo por equivalentes quantizados e os calibra. O resultado mantém o contrato comum de predict, val, train e save, então um modelo quantizado é avaliado pelos mesmos validadores que um modelo float."
keywords:
  - quantização libreyolo
  - int8 ptq
  - quantization aware training
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dataset de calibração
  - exportar onnx qdq
last_verified: "1.5.0"
meta:
  - label: Chamada
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Comando
    value: "libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml"
    mono: true
  - label: Extra
    value: "Nenhum. A quantização roda em PyTorch."
  - label: Famílias
    value: "yolo9, rfdetr, birefnet, feynobg"
  - label: Receitas
    value: "fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2"
    mono: true
  - label: Artefatos de deploy
    value: 'export(format="pt") para um checkpoint empacotado, export(format="onnx") para um grafo QDQ INT8'
    mono: true
verification: "Lido de libreyolo/quant/api.py, libreyolo/models/base/model.py, libreyolo/cli/commands/quantize.py e docs/quantization.md no branch dev. Os números de tamanho de checkpoint são os valores medidos registrados em docs/quantization.md."
snippets:
  quantize:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Troca de estrutura mais calibração. calib é um pequeno conjunto de imagens
        # SEM RÓTULOS, lido só no forward para derivar faixas de ativação e escalas.
        qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

        print(qmodel.quant_info())
        qmodel.val(data="coco8.yaml")          # mesmos validadores de um modelo float
        qmodel.save("LibreYOLO9s-int8.pt")     # o checkpoint carrega um manifesto quant
    - label: CLI
      language: bash
      code: |
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco128.yaml
    - label: Argumentos
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # caminho do data.yaml ou nome embutido; None pula a calibração
            samples=128,               # máximo de imagens de calibração
            batch=8,                   # tamanho de batch da calibração
            algorithm="auto",          # auto e minmax são iguais; percentile é a alternativa
            keep_high_precision=None,  # None usa a política da família
            verbose=True,
        )
  reload:
    - label: Um checkpoint quantizado recarrega como tal
      language: python
      code: |
        from libreyolo import LibreYOLO

        # O manifesto quant reconstrói a estrutura quantizada e as escalas
        # antes de os pesos serem carregados.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT é um train() comum sobre um modelo quantizado
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Um fine-tuning, não um treinamento do zero: use learning rates de fine-tuning.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD acrescenta os argumentos de destilação já existentes
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5 --lr0 1e-4
  export:
    - label: Checkpoint PyTorch empacotado
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Escreve LibreYOLO9s-int8-final.pt: pesos e escalas de baixa precisão empacotados,
        # masters fp32 removidos, o restante não quantizado convertido para fp16.
        qmodel.export(format="pt")

        # remainder="fp32" mantém exatos os tensores não quantizados.
        qmodel.export(format="pt", remainder="fp32")
    - label: ONNX QDQ INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Pares QuantizeLinear/DequantizeLinear dentro do grafo, carregando as
        # próprias escalas calibradas ou treinadas por QAT do modelo.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: De volta ao float, preservando os pesos treinados por QAT
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # Qualquer exportador float agora se aplica, em qualquer precisão que suporte.
        qmodel.export(format="tensorrt", half=True)
---

## Instalação

A quantização não precisa de extra. A troca de módulos, a passada de calibração e a
aritmética simulada rodam todas em PyTorch, então `pip install libreyolo` é todo o
requisito. Os artefatos de deploy precisam do que o próprio formato deles exigir, o
que no caminho ONNX é `libreyolo[onnx]`.

## Quantizar

<code-tabs name="quantize" />

`quantize()` transforma o modelo carregado in place e o retorna. Não há gradientes
envolvidos: a troca instala módulos quantizados e a passada de calibração roda
somente no forward.

O checkpoint resultante é um checkpoint LibreYOLO comum com um manifesto `quant`
anexado, então ele recarrega com a estrutura e as escalas intactas:

<code-tabs name="reload" />

Os checkpoints escritos pelo trainer durante um treinamento QAT também carregam o
manifesto, o que significa que o `best.pt` desse treinamento é ele mesmo um
checkpoint quantizado.

## Receitas

Quatro famílias são suportadas: `yolo9`, `rfdetr`, `birefnet` e `feynobg`.

| Receita | O que faz | Famílias | Calibração |
|---|---|---|---|
| `fp16` | Converte para meia precisão com um contrato de entrada e saída em float32. Somente inferência. | as quatro | nenhuma |
| `bf16` | Converte para bfloat16, que mantém a faixa de expoente do float32. A solução quando fp16 estoura em um modelo estilo DETR. Somente inferência. | as quatro | nenhuma |
| `fp8` | Pesos e ativações E4M3 em `Conv2d` e `Linear`: escalas de peso por canal, escalas de ativação por tensor calibradas. | as quatro | obrigatória |
| `int8` | W8A8 em `Conv2d` e `Linear`: pesos simétricos por canal, ativações afins por tensor. | as quatro | obrigatória, ou `calib=None` para somente pesos |
| `w4a16` | Pesos INT4 simétricos agrupados, grupo 128 ao longo de `in_features`, ativações float, em `Linear`. | rfdetr, birefnet, feynobg | não é necessária |
| `w4a8` | Pesos INT4 agrupados mais ativações INT8 calibradas, em `Linear`. | rfdetr, birefnet, feynobg | obrigatória |
| `nvfp4` | W4A4 NVFP4 em `Linear`: elementos E2M1, blocos de 16 elementos, escalas de bloco FP8 E4M3, escala de tensor FP32. Escalonamento dinâmico de ativações. | rfdetr, birefnet, feynobg | não é necessária |
| `mxfp4` | OCP MXFP4 em `Linear`: elementos E2M1, blocos de 32 elementos, escalas de bloco E8M0 em potências de dois. Escalonamento dinâmico de ativações. | rfdetr, birefnet, feynobg | não é necessária |
| `int2` | Somente pesquisa: pesos agrupados de 2 bits, grupo 64, mais ativações INT8, em `Linear`. Sozinho, o pós-treinamento é inutilizável, então QAT ou QAD é obrigatório. | rfdetr | obrigatória |

As receitas de menos de 8 bits miram `nn.Linear` e são rejeitadas para `yolo9` de
propósito: essa aceleração é só de GEMM no hardware atual, então as convoluções
ficam em precisão mais alta. O YOLO9 usa `int8` ou `fp8`. `int2` é rejeitada para
`birefnet` e `feynobg` porque essas famílias são somente inferência, então o QAT de
que a receita depende para se recuperar não está disponível ali.

Os padrões por família mantêm a primeira camada e as cabeças em float, e a
convolução DFL do YOLO9 nunca é quantizada: ela é um operador fixo de esperança
integral. Sobrescreva com `keep_high_precision=("head.",)` quando você tiver um
motivo.

## Dados de calibração não são dados de treinamento

`calib=` recebe algumas centenas de imagens, não lê rótulo nenhum e roda somente no
forward para estimar faixas de ativação. `data=` em `train()` e `val()` é o dataset
rotulado usado para gradientes e métricas. São argumentos diferentes com propósitos
diferentes, e o padrão de `calib` é `coco128.yaml`.

`algorithm="minmax"` guarda os extremos absolutos vistos ao longo dos batches de
calibração e é o que `"auto"` seleciona. `"percentile"` usa a média dos percentis
0.1 e 99.9 de cada batch; mediu-se que ele derruba a acurácia da família DETR,
porque os outliers de ativação dos transformers são estruturais. O que de fato
resolve a sensibilidade a INT8 dos modelos pequenos é calibrar com batches
suficientes: com o padrão `coco128`, o YOLO9-t fica a cerca de um ponto de mAP do
seu score float. O algoritmo escolhido fica registrado no manifesto do checkpoint.

## Recuperar acurácia

<code-tabs name="train" />

Os módulos quantizados mantêm pesos master em fp32 e aplicam quantização falsa com
um straight-through estimator, então os gradientes chegam aos masters e os trainers
existentes funcionam sem mudanças: EMA, AMP, retomada de checkpoint e os argumentos
de destilação, tudo se combina.

QAT é um fine-tuning de um modelo já treinado. Use learning rates de fine-tuning em
vez dos padrões de treinamento do zero, ou um treinamento curto vai destruir os
pesos pré-treinados independentemente da quantização. A disponibilidade de QAD
acompanha o suporte a destilação de cada família, o que hoje significa `yolo9` e
`rfdetr`.

Modelos quantizados com `fp16` e `bf16` são somente inferência, e o trainer os
rejeita apontando para `amp=True`.

## Exportação

<code-tabs name="export" />

`format="pt"` cristaliza o modelo. Pesos e escalas de baixa precisão empacotados
substituem os masters, e o restante não quantizado é convertido para fp16 a menos
que `remainder="fp32"` seja passado. A invariante do empacotamento é que o
desempacotamento reproduz a simulação bit a bit no dispositivo em que você
finalizou, então o arquivo finalizado pontua exatamente o que você validou. Medido:
o YOLO9-s int8 vai de 29.5 MB para 9.6 MB, o RF-DETR-n nvfp4 de 122 MB para 26 MB.
Carregar um deles dá um modelo pronto para inferência, e chamar `train()` sobre ele
reconstrói os masters a partir dos pesos empacotados automaticamente.

`format="onnx"` se aplica a modelos `int8` e emite um grafo QDQ carregando as
próprias escalas calibradas ou treinadas por QAT do modelo, que o ONNX Runtime e o
TensorRT rodam com kernels INT8 reais. Este é um caminho diferente de
[`export(format="onnx", int8=True)`](/docs/export/onnx) em um modelo float, onde o
ONNX Runtime deriva as escalas por conta própria.

As receitas de conversão não precisam de exportador quantizado nenhum:

<code-tabs name="dequantize" />

## Restrições

A aritmética quantizada executa em simulação, que é quantização falsa calculada em
ilhas de float32 mesmo sob AMP. A simulação é fiel na numérica, então um score de
`val()` em qualquer dispositivo é uma afirmação real sobre a aritmética quantizada.
Não é uma afirmação sobre velocidade.

Duas exceções executam nativamente. `fp16` e `bf16` são conversões comuns. Módulos
`fp8` finalizados rodam seu GEMM diretamente sobre pesos E4M3 empacotados através de
`torch._scaled_mm` em hardware das classes Ada, Hopper e Blackwell, usando as mesmas
escalas de ativação calibradas da simulação; definir `LIBREYOLO_KERNELS=off`
restaura o caminho simulado exato em todo lugar.

A cobertura de deploy é mais estreita que a lista de receitas. Só `int8` tem uma
forma ONNX deployável aqui; `fp8` e as receitas lineares de menos de 8 bits executam
em PyTorch e cristalizam através de `format="pt"`. Pedir uma exportação ONNX a
partir delas gera um erro com essa instrução, assim como pedir qualquer formato que
não seja ONNX a partir de um modelo `int8`: construa os engines downstream a partir
do grafo QDQ.

Exportar um modelo `int8` cujas ativações nunca foram calibradas registra um aviso e
produz um grafo carregando somente a quantização dos pesos.
