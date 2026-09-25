---
title: Alternativa ao SuperGradients para visão computacional em tempo real
description: "O SuperGradients parou de receber versões após a NVIDIA adquirir a Deci em 2024. O LibreYOLO é uma alternativa mantida que executa os mesmos pesos YOLO-NAS: faça predições, treinamento e exportação em uma única API com licença MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "O SuperGradients ainda é mantido?"
    a: "Não. A última versão foi a 3.7.1, em abril de 2024, pouco antes de a NVIDIA adquirir a Deci. Desde então, o repositório não teve novas versões, as issues ficam sem resposta e o site da documentação saiu do ar. Ele não foi arquivado formalmente, mas não há ninguém cuidando dele."
  - q: "Qual é a melhor alternativa ao SuperGradients?"
    a: "Para executar YOLO-NAS, o LibreYOLO carrega os mesmos pesos da Deci por meio de uma API mantida e com licença MIT, além de oferecer treinamento, validação e exportação. Se você depende das receitas de treinamento do SuperGradients para outras arquiteturas, o repositório antigo ainda funciona quando você fixa as dependências."
  - q: "Posso usar YOLO-NAS comercialmente?"
    a: "Os pesos pré-treinados da Deci não podem ser usados comercialmente, independentemente da biblioteca que os carrega. A arquitetura tem uma licença permissiva, então treinar um YOLO-NAS do zero com seus próprios dados gera um modelo sem nenhum checkpoint da Deci no histórico."
  - q: "O LibreYOLO substitui tudo o que o SuperGradients fazia?"
    a: "Não. O LibreYOLO não exporta YOLO-NAS para CoreML, e o caminho de TensorRT para essa família não foi testado. As receitas de treinamento com reconhecimento de quantização do SuperGradients também não têm um equivalente direto. Nesses casos, mantenha o repositório antigo."
---

O SuperGradients era a biblioteca de treinamento de código aberto da Deci e a casa do YOLO-NAS, um dos detectores em tempo real mais precisos já lançados. Em maio de 2024, a NVIDIA adquiriu a Deci, e o SuperGradients ficou inativo. A última versão, 3.7.1, saiu em 8 de abril de 2024. A documentação em supergradients.com saiu do ar. Há cerca de 120 issues abertas, e a issue intitulada ["Este projeto morreu?"](https://github.com/Deci-AI/super-gradients/issues/2062) não tem resposta de um mantenedor, o que por si só já responde à pergunta.

O repositório não foi arquivado, então à primeira vista parece ativo. Na prática, o abandono fica evidente assim que você tenta instalá-lo: `super-gradients` fixa `torchmetrics==0.8`, que entra em conflito com as pilhas atuais do PyTorch, e ainda traz hydra, omegaconf, boto3 e tensorboard. A cada mês, essas versões fixadas entram em mais conflito com o restante do seu ambiente, e nenhuma correção está a caminho.

Isso não torna o YOLO-NAS um modelo pior. A variante grande ainda alcança 52.2 mAP no COCO em velocidade de tempo real. O modelo está bem; o lugar onde ele morava pegou fogo.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="Acurácia do YOLO-NAS vs. número de parâmetros - visionanalysis.org">
</iframe>

## Execute os mesmos pesos no LibreYOLO

O LibreYOLO carrega os checkpoints YOLO-NAS diretamente da CDN da Deci, usando a mesma API que usa para todas as outras famílias de modelos. Você não precisa escrever configurações de hydra nem desembrulhar um wrapper de predição:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # download automático na primeira execução
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

As variantes de detecção S, M e L funcionam, assim como a de pose: troque por `LibreYOLONASs-pose.pt` para obter os keypoints do COCO. Como todas as famílias retornam o mesmo objeto `Results`, comparar YOLO-NAS com RF-DETR ou D-FINE nos seus próprios dados exige uma alteração de uma linha, em vez de uma segunda base de código.

## Treinamento e exportação, não apenas inferência

O SuperGradients era, antes de tudo, uma biblioteca de treinamento, então uma alternativa de verdade precisa permitir treinar. O LibreYOLO faz fine-tuning do YOLO-NAS no seu dataset com a mesma chamada usada para os outros modelos:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Você também pode treinar a partir de um modelo com inicialização aleatória, o que importa para o licenciamento (falaremos mais disso abaixo). A validação retorna métricas mAP para qualquer dataset no seu formato, e a exportação inclui ONNX, TorchScript, OpenVINO, NCNN e TFLite. Essa matriz de exportação é mais ampla do que a biblioteca original chegou a oferecer para YOLO-NAS. Os detalhes estão na [página de documentação do YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), e há uma nota mais curta sobre como [YOLO-NAS continua sendo mantido](/articles/yolo-nas-with-libreyolo).

## Quando o repositório antigo ainda é a escolha certa

Uma seção franca. Três casos em que você deve manter o SuperGradients instalado:

**CoreML.** O LibreYOLO não exporta YOLO-NAS para CoreML. Se você distribui em dispositivos Apple e precisa de um `.mlpackage`, o SuperGradients ainda tem um caminho funcional.

**TensorRT.** O SuperGradients documentou e testou o fluxo com TensorRT para YOLO-NAS, incluindo todas as particularidades do tamanho de batch. O suporte do LibreYOLO a TensorRT para essa família não foi testado.

**Receitas de treinamento com reconhecimento de quantização.** O YOLO-NAS foi projetado para INT8, e o SuperGradients oferecia as receitas de QAT que faziam o modelo se destacar nesse cenário. O LibreYOLO exporta com quantização INT8 e FP16, mas não tem um equivalente a essas receitas de treinamento.

O repositório ainda funciona se você fixar um ambiente da época de 2024. Só não é uma boa ideia construir algo novo sobre uma base que ninguém mantém.

## Uma observação sobre os pesos

Os pesos pré-treinados do YOLO-NAS são da Deci e foram lançados sob uma licença para uso não comercial. Essa licença acompanha os pesos em qualquer biblioteca que os carregue. O LibreYOLO não hospeda nem espelha esses pesos: o download vem da CDN pública da Deci e exibe os termos da Deci antes de começar. Para pesquisa e trabalho não comercial, qualquer uma das opções serve.

Se você precisa de um YOLO-NAS para uso comercial, agora há um caminho claro: a própria arquitetura tem uma licença permissiva, então treinar do zero com seus dados gera um modelo que não deriva de nenhum checkpoint da Deci. O LibreYOLO oferece suporte exatamente a isso com `LibreYOLONAS(None, size="s")`.

## Experimente

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

O LibreYOLO usa a licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon e CPU comum sem nenhuma alteração no código. Uma API reúne YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet e muitos outros, para detecção, segmentação, pose, classificação, profundidade e rastreamento.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs)
