---
title: Malha corporal
seo_title: Recuperação de malha corporal no LibreYOLO
description: >-
  Recupere uma malha corporal 3D paramétrica por pessoa no LibreYOLO. Faça
  predições a partir de boxes de pessoas ou de um detector, e leia vértices,
  articulações e a translação de câmera.
lead: >-
  A recuperação de malha corporal transforma uma única imagem e um conjunto de
  boxes de pessoas em um corpo 3D paramétrico por pessoa: parâmetros de forma e
  de pose, vértices posados, articulações 3D e a translação de câmera que os
  coloca na frente da lente.
keywords:
  - malha corporal 3d python
  - human mesh recovery python
  - pose 3d do corpo humano
  - reconstrução 3d de pessoas
  - SAM 3D Body
  - MHR
  - modelo corporal paramétrico
  - tarefa mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Esta família não está registrada na fábrica LibreYOLO(), então ela
        # é construída diretamente. model_path=None dispara o download
        # restrito do Hugging Face; uma string é tratada como um checkpoint
        # local existente e nunca é baixada. A inferência exige CUDA.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # a parametrização que esses tensores usam
        print(meshes.vertices.shape)  # (N, V, 3), referencial da câmera, metros
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2), pixels na imagem de origem
    - label: Com um detector de pessoas
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # person_detector aceita um detector LibreYOLO já construído, um

        # callable comum ou uma instância de PersonDetector. Não há atalho por
        nome.

        detector = LibreYOLO("LibreYOLO9s.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definição

A recuperação de malha corporal retorna um payload `Meshes` por imagem, alinhado
linha a linha com `result.boxes`: a linha `i` descreve a pessoa no box `i`, o
mesmo contrato que a tarefa de pose usa para os keypoints.

Tudo é expresso no referencial da câmera da imagem original.
`transl` é métrico, em metros, com +z apontando para longe da câmera.
`vertices` e `joints3d` são métricos e já incluem `transl`, então não exigem
nenhuma composição adicional. `joints2d` vem em pixels sobre o canvas da
imagem original, e não sobre o recorte que a rede viu. `faces` guarda a
topologia da malha uma vez para a imagem inteira, em vez de por linha, porque
todas as pessoas compartilham a mesma. Não existe referencial de mundo nem de
gravidade nesta versão, e nenhum campo faz esse papel em silêncio.

Os layouts de parâmetros diferem entre modelos corporais, então nada sobre os
formatos é fixo: `body_model` dá o nome da parametrização e as contagens são
lidas dos próprios tensores. Para `"mhr"`, o Momentum Human Rig, as
rotações são ângulos de Euler em radianos em vez de axis-angle, `body_pose` é um
vetor plano de parâmetros por articulação em vez de um trio por articulação, e
`betas` são coeficientes de blendshape de identidade. A escala do esqueleto, a
pose das mãos e a expressão facial ficam em `extras`.

A chave canônica da tarefa é `mesh`. `body-mesh`, `hmr` e `human-mesh-recovery`
são normalizados para ela.

## Modelos

O [SAM 3D Body](/docs/models/sam-3d-body) é a única família que atende esta
tarefa, e ele é um wrapper em vez de um port: o pacote `sam-3d-body` da Meta é
publicado sob a SAM License, da qual o código do próprio LibreYOLO não pode
derivar, então nada dele vem embutido. Dois backbones compartilham o mesmo
modelo corporal MHR, `d3` sobre um encoder DINOv3 ViT-H/16+ e `h` sobre o ViT-H
original.

Três requisitos se aplicam antes da primeira predição, e nenhum deles é
opcional.

O pacote upstream é instalado por você, não pelo LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Aponte a biblioteca para o clone com `sam_3d_body_path=` ou com a variável de
ambiente `SAM_3D_BODY_PATH`. Quem nunca constrói essa família nunca dispara o
import.

O espelho do checkpoint é restrito. Aceite a licença na página do modelo no
Hugging Face e autentique com `hf auth login`, ou o primeiro download falha. O
modelo corporal MHR em si é uma versão Apache-2.0 separada, baixada de um local
público próprio e mantida em cache localmente.

A inferência precisa de um dispositivo CUDA. O estimador upstream move o batch
para a GPU sem verificar, então não há caminho por CPU para o qual recorrer e
`device="cpu"` lança uma exceção.

## Predição

<code-tabs name="predict" />

As pessoas chegam ao modelo por um de dois caminhos. `person_boxes` passa boxes
que você já tem, e só para uma única imagem: um conjunto fixo de boxes não
consegue acompanhar as pessoas ao longo dos frames de um vídeo, então passá-lo
com uma fonte de vídeo lança uma exceção em vez de reutilizar em silêncio os
boxes do primeiro frame. `person_detector` aceita um detector LibreYOLO já
construído, um callable ou um `PersonDetector`, e é o caminho para vídeo.
`focal_length` fornece um parâmetro intrínseco de câmera conhecido; se você não
definir, o modelo usa a própria estimativa, que é o que `meshes.focal_length`
reporta.

Essa família não está ligada à fábrica `LibreYOLO()` nem ao comando de CLI
`libreyolo predict`. `LibreSAM3DBody` é o único ponto de entrada. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Treinamento

Nenhuma família desta tarefa treina dentro do LibreYOLO. `LibreSAM3DBody.train()`
lança uma exceção: treine no projeto upstream e carregue aqui o checkpoint
resultante.

## Validação

Não existe validador de malha, e `val()` lança uma exceção. Os benchmarks
habituais têm licença apenas para pesquisa, então nenhum vem junto e nenhum
pode ser baixado para você.

As métricas em si estão disponíveis como `libreyolo.validation.mesh_metrics`,
para avaliar contra um dataset que você já tem. A função recebe as articulações
preditas e as de referência, opcionalmente os vértices preditos e os de
referência, e retorna um dicionário com exatamente as mesmas chaves de um
validador:

`metrics/mpjpe` é o erro médio de posição por articulação depois de alinhar a
articulação raiz, então ele pontua a pose ignorando onde a pessoa está na cena.
`metrics/pa_mpjpe` é a mesma quantidade depois de um alinhamento de Procrustes
completo, rotação, escala uniforme e translação, o que remove a orientação
global e o erro de tamanho do corpo e deixa a pose articulada. `metrics/pve` é o
erro médio por vértice sobre a superfície da malha depois de alinhar pelo
centroide dos vértices; ao contrário das métricas de articulação, ele é sensível
à forma do corpo, e só aparece quando os dois arrays de vértices são fornecidos.
Nos três, quanto menor, melhor. As entradas são consideradas métricas, em
metros, e `scale_to_mm` converte os resultados para os milímetros que a
literatura reporta.

## Exportação

A exportação de malha não está implementada. O LibreYOLO ainda não definiu um
contrato de metadados de grafo exportado para esta tarefa, incluindo como levar
o layout de parâmetros do MHR para fora do PyTorch, então `export()` lança uma
exceção em vez de emitir um grafo cuja saída não poderia ser interpretada.
