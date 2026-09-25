---
title: "RT-DETR sem Ultralytics: v1, v2 e v4 com Apache-2.0"
description: "Execute e faça fine-tuning de RT-DETR, RT-DETRv2 e RT-DETRv4 com pesos Apache-2.0 em uma biblioteca MIT: uma API para predict, train, val e export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR é gratuito para uso comercial?"
    a: "Sim, se você usar uma implementação permissiva. O código original do RT-DETR e do RT-DETRv2 (lyuwenyu/RT-DETR) e o do RT-DETRv4 (RT-DETRs/RT-DETRv4) são Apache-2.0, e o LibreYOLO executa as três versões com pesos Apache-2.0 em uma biblioteca MIT. A implementação de RT-DETR dentro do pacote ultralytics é AGPL-3.0, com uma Enterprise License paga como alternativa para uso de código fechado. Estas são informações gerais, não aconselhamento jurídico."
  - q: "Qual é a licença do RT-DETRv4?"
    a: "Apache-2.0. O arquivo LICENSE em github.com/RT-DETRs/RT-DETRv4 é Apache-2.0. O RT-DETRv4 usa um professor DINOv3 somente durante o treinamento; os pesos do aluno publicados não contêm parâmetros do DINOv3, portanto a licença do DINOv3 da Meta não se aplica a eles."
  - q: "Posso fazer fine-tuning de RT-DETR sem Ultralytics?"
    a: "Sim. O LibreYOLO faz fine-tuning de checkpoints de detecção de RT-DETR, RT-DETRv2 e RT-DETRv4 com model.train(), ou com libreyolo train na linha de comando, começando pelos pesos COCO publicados. Os modelos de caixa orientada do RT-DETRv2 são somente para inferência."
  - q: "Quais pesos de RT-DETR vêm com o LibreYOLO?"
    a: "RT-DETR em r18, r34, r50, r50m, r101, l e x; RT-DETRv2 em r18, r34, r50, r50m e r101; RT-DETRv4 em s, m, l e x, todos para detecção COCO a 640 px. O RT-DETRv2 também tem modelos de caixa orientada n, s, m, l e x, treinados no DOTA v1.0 a 1024 px. Todos os arquivos são Apache-2.0."
---

O RT-DETR é um transformer de detecção em tempo real da Baidu. Ele decodifica um conjunto fixo de queries em vez de uma grade densa, então não há uma etapa de NMS para ajustar. Pesquise por ele e uma das primeiras implementações que você encontra é a que vem dentro do pacote Python `ultralytics`. Isso levanta uma questão de licença que o próprio modelo nunca teve.

## A licença do RT-DETR depende do repositório

A licença pertence a um repositório, não a um modelo. O RT-DETR tem vários repositórios, e eles não usam a mesma licença:

| Repositório | Abrange | Licença |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR e RT-DETRv2, PyTorch e Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR dentro do pacote `ultralytics` | AGPL-3.0 ou Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 e RT-DETRv4 | Código MIT, pesos Apache-2.0 |

A implementação da Ultralytics é sólida. A [página do RT-DETR](https://docs.ultralytics.com/models/rtdetr/) lista os modelos pré-treinados `rtdetr-l.pt` e `rtdetr-x.pt`, com treinamento, validação, inferência e exportação. Ela vem em um pacote licenciado sob AGPL-3.0, e a Ultralytics vende uma Enterprise License para equipes que não querem abrir o código-fonte de todo o projeto. Se nenhuma das opções funcionar para você, a arquitetura está disponível sob Apache-2.0, nos repositórios dos próprios autores. O [guia de licenças YOLO](/articles/yolo-licenses-explained) aborda o mesmo padrão em toda a família YOLO, e [o guia de licença comercial](/articles/yolo-commercial-license) explica o que AGPL-3.0 significa para um produto de código fechado.

## Execute RT-DETR com o LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

A mesma chamada funciona pela linha de comando:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` e `max_det` filtram uma decodificação top-k entre queries e classes. `iou` é aceito, mas não é usado, porque não há NMS. O objeto `Results` retornado é o mesmo que todos os modelos LibreYOLO retornam, então trocar de detector exige uma alteração de uma linha.

## RT-DETR, RT-DETRv2 e RT-DETRv4 em um único loader

A versão faz parte do nome do arquivo, e `LibreYOLO()` encaminha com base no checkpoint, então as três versões são carregadas da mesma forma:

* **RT-DETR:** `LibreRTDETR` em r18, r34, r50, r50m, r101 (backbones ResNet) e l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` em r18, r34, r50, r50m e r101. Ele mantém a arquitetura da versão 1 e muda a forma como a atenção deformável faz a amostragem.
* **RT-DETRv4:** `LibreRTDETRv4` em s, m, l e x. Ele reutiliza a arquitetura do D-FINE, e seus pesos vêm da destilação de um professor DINOv3 em um aluno HGNetv2.

Todos os pesos de detecção são treinados no COCO e rodam a 640 px. Para referência, os READMEs upstream informam 46.5 AP para RT-DETR-R18 e 53.0 AP para RT-DETR-L no COCO, e 49.8 AP para RT-DETRv4-S até 57.0 AP para RT-DETRv4-X. A [página de documentação do RT-DETR](/docs/models/rt-detr) tem a tabela de benchmark do próprio LibreYOLO para cada checkpoint.

O RT-DETRv2 também detecta caixas orientadas. `LibreRTDETRv2n-obb.pt` até `LibreRTDETRv2x-obb.pt` são os checkpoints oficiais do DOTA v1.0, com 15 classes aéreas a 1024 px, convertidos para o formato do LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[A detecção orientada](/docs/tasks/oriented-detection) explica o formato dos rótulos e as métricas.

## Faça fine-tuning de RT-DETR nos seus próprios dados

O treinamento começa com um checkpoint publicado:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Para uma execução real, aponte `data` para o YAML do seu próprio dataset. Cada versão tem seu próprio learning rate padrão, e as versões 1 e 2 definem a taxa do backbone como um vigésimo de `lr0`, seguindo a receita original. Para usar várias GPUs, passe `device=0,1` na CLI; para fazer fine-tuning com adaptadores [LoRA](/docs/train/lora), use `lora=True` com o extra `lora` instalado. Os modelos de caixa orientada do RT-DETRv2 são somente para inferência. Consulte [treinamento](/docs/train) para ver informações sobre datasets, data augmentation e loggers.

## Valide e exporte

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` retorna um dicionário simples. Os formatos de exportação incluem ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO; a matriz de exportação na [página do modelo](/docs/models/rt-detr) mostra quais são validados para cada tarefa. Um arquivo `.onnx` ou `.engine` exportado pode ser carregado novamente com `LibreYOLO()` e retorna o mesmo `Results`. Consulte [exportação](/docs/export) para ver os guias de cada formato.

## Quando ainda vale a pena usar os repositórios originais

Se você precisa reproduzir o resultado de um artigo com as configurações exatas dos autores, quer a versão Paddle ou quer executar por conta própria a destilação do professor DINOv3 do RT-DETRv4, use os repositórios upstream. O LibreYOLO faz fine-tuning do aluno v4 publicado; ele não executa o professor. E, se seu projeto já usa AGPL-3.0 ou está coberto por uma Enterprise License da Ultralytics, não há motivo de licença para migrar.

## Uma observação sobre a licença

O código do LibreYOLO é MIT. Todo arquivo de pesos RT-DETR é Apache-2.0, e cada repositório de pesos no Hugging Face inclui o LICENSE e o NOTICE upstream, que a Apache-2.0 pede que você mantenha ao redistribuir os pesos. Os pesos que você treina com seus próprios dados são seus. O RT-DETRv4 usa o DINOv3 somente como professor durante o treinamento, e os alunos publicados não contêm parâmetros do DINOv3. Estas são informações gerais, não aconselhamento jurídico. Confira os arquivos LICENSE atuais antes de distribuir.

## Experimente

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

O LibreYOLO usa a licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon e CPU comum sem nenhuma alteração no código.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação do RT-DETR](/docs/models/rt-detr)
