---
title: Como executar o Depth Anything V2 com LibreYOLO
description: O Depth Anything V2 oferece estimativa monocular de profundidade de última geração. Veja como executá-lo em duas linhas com LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Qual é a maneira mais simples de executar o Depth Anything V2?"
    a: "Duas linhas com LibreYOLO: carregue LibreDepthAnythingV2l-depth.pt pelo nome e chame predict com save=True. Os pesos são baixados automaticamente no primeiro uso, e a normalização, o mapa de cores e a seleção do dispositivo são feitos para você em CUDA, Apple Silicon ou CPU comum."
  - q: "Posso usar o Depth Anything V2 comercialmente?"
    a: "Somente o encoder Small usa Apache 2.0. Base, Large e Giant usam CC-BY-NC-4.0, então os checkpoints mais robustos são para uso não comercial. A licença upstream se aplica independentemente da biblioteca que carrega os pesos."
  - q: "O LibreYOLO oferece suporte ao treinamento do Depth Anything V2?"
    a: "Não. O treinamento continua no repositório original; o LibreYOLO cobre inferência, vídeo e validação para a tarefa de profundidade."
---

![Guggenheim Bilbao ao lado do mapa de profundidade](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

O Depth Anything V2 produz alguns dos melhores mapas de profundidade monocular disponíveis atualmente.

Se você vem do mundo YOLO, o repositório oficial não é o mais fácil de começar a usar. Para obter um único mapa de profundidade, são necessárias várias etapas manuais:

* baixar o checkpoint certo manualmente e colocá-lo na pasta correta,

* combinar o encoder com a configuração correspondente (`encoder="vitl"`, `features=256`, `out_channels=...`),

* escrever por conta própria as etapas de normalização e aplicação de cores,

* configurar o dispositivo para que rode em um Mac ou na CPU, e não apenas em CUDA,

* e aprender uma API de inferência que não se parece em nada com o resto da sua stack.

Nada disso é difícil. Só dá trabalho, e você pode pular tudo isso.

O LibreYOLO carrega os mesmos pesos do Depth Anything V2 e disponibiliza a tarefa de profundidade em uma única chamada a `predict()`. Informe o nome do modelo e ele será baixado automaticamente no primeiro uso:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

É só isso. Se você já usou a API padrão do YOLO, a estrutura é familiar: carregue um modelo pelo nome, chame `predict` e deixe `save=True` gravar a visualização. É exatamente a mesma chamada que você usaria para detecção de objetos, só que o resultado contém um mapa de profundidade em vez de caixas. O mapa de cores, a normalização e a seleção do dispositivo são feitos para você. Ele roda da mesma forma no Linux com CUDA, em um Mac com Apple Silicon ou em uma CPU comum. Sem mudar o código.

A partir daí, a mesma chamada faz mais: você pode obter os valores brutos de profundidade e trabalhar diretamente com eles. O treinamento do próprio Depth Anything V2 continua no repositório original; o LibreYOLO cobre inferência, vídeo e validação.

A mesma chamada de uma linha, em cenas bem diferentes:

![A imagem de exemplo de parkour ao lado do mapa de profundidade: as pessoas saltando em primeiro plano se destacam diante das paredes de concreto ao fundo.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Vista aérea da baía de La Concha, em Donostia, ao lado do mapa de profundidade: os barcos e a linha costeira parecem próximos, enquanto o mar aberto fica ao fundo.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![O pátio com colunas da Casa de Juntas de Gernika ao lado do mapa de profundidade, que mostra a arquitetura recuando ao fundo.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Multidão em um festival noturno na Plaza de la Constitución, em Donostia, ao lado do mapa de profundidade: as pessoas nas fileiras próximas se destacam diante da fachada iluminada ao fundo.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Uma observação sobre os pesos: o LibreYOLO hospeda checkpoints convertidos do Depth Anything V2 e os baixa no primeiro uso, então não há nada para baixar manualmente. A licença upstream continua valendo: o encoder Small usa Apache-2.0, enquanto Base, Large e Giant usam CC-BY-NC-4.0 (não comercial), então os checkpoints mais robustos são para uso não comercial. Quer trabalhar offline ou converter os seus próprios pesos? O script de conversão, para uso pontual, continua disponível:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Experimente

```bash
pip install libreyolo
```

O LibreYOLO é a biblioteca de visão computacional mais completa que você pode instalar com pip. Uma API familiar abrange um conjunto crescente de modelos de última geração: detecção de objetos (RF-DETR, D-FINE, DEIM), segmentação, pose, caixas orientadas e agora estimativa monocular de profundidade com o Depth Anything V2, além de treinamento e validação para as tarefas compatíveis. Ele roda em todos os principais sistemas operacionais, em GPU ou CPU, e usa inteiramente a licença MIT, então você pode distribuí-lo comercialmente sem restrições.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
