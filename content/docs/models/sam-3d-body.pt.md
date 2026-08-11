---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: recuperação de malha de corpo inteiro no LibreYOLO'
description: >-
  Use o SAM 3D Body no LibreYOLO para recuperação de malha 3D de corpo humano
  inteiro. Instale e faça predições; a SAM License da Meta restringe os
  checkpoints e CUDA é obrigatório.
lead: >-
  O SAM 3D Body é o modelo da Meta guiado por prompts para recuperar uma malha
  3D de corpo inteiro, incluindo mãos e pés, a partir de uma única imagem e de
  caixas de pessoas. O LibreYOLO envolve o pacote upstream em vez de portá-lo.
keywords:
  - SAM 3D Body
  - MHR
  - Momentum Human Rig
  - malha 3d do corpo humano
  - reconstrução 3d de pessoas
  - human mesh recovery
  - pose 3d python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Esta família não está registrada na fábrica LibreYOLO(), então ela
        # é construída diretamente. model_path=None é o que dispara o
        # download restrito do Hugging Face; já uma string é tratada como o
        # caminho de um checkpoint local existente e nunca é baixada sozinha.
        # A inferência exige um dispositivo CUDA; não há caminho por CPU.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3), sistema da câmera, metros
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Com um detector de pessoas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Aqui não há atalho por string nomeada: passe um detector LibreYOLO
        # já construído, um callable comum ou uma instância de PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Instalação

```bash
pip install libreyolo
```

Isso te dá apenas o adaptador do LibreYOLO. O SAM 3D Body em si não vem junto,
porque a licença dele não é uma da qual o código do próprio LibreYOLO possa
derivar: clone o repositório upstream e instale as dependências dele por conta
própria, depois aponte o LibreYOLO para o clone.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

ou defina a variável de ambiente `SAM_3D_BODY_PATH` em vez de passar
`sam_3d_body_path` em toda chamada. Quem nunca constrói essa família nunca
dispara o import, e nunca esbarra na SAM License. Essa família não está ligada
à fábrica `LibreYOLO()` nem ao comando de CLI `libreyolo predict`;
`LibreSAM3DBody` é o único ponto de entrada.

## Predição

<code-tabs name="predict" />

O download do checkpoint é restrito: exige aceitar a licença da Meta na página
do modelo no Hugging Face e autenticar com `hf auth login` antes que o primeiro
download funcione. A inferência em si precisa de um dispositivo CUDA sem
exceção: o estimador upstream move o batch dele para a GPU sem checar nada,
então uma máquina só com CPU levanta um erro em vez de cair para o fallback.
`result.meshes` é um payload `Meshes`, alinhado linha a linha com
`result.boxes` (uma linha por pessoa detectada): `vertices` e `joints3d` são
métricos e já incluem a translação de câmera estimada, `joints2d` vem em pixels
sobre a imagem original, e as rotações seguem a convenção do MHR, ângulos de
Euler em vez de axis-angle. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Dois backbones por trás do mesmo modelo corporal MHR: `d3` usa um encoder
DINOv3 ViT-H/16+ e `h` usa o encoder ViT-H original.

## Exportação

<export-matrix />

A exportação de malha corporal não está implementada: o LibreYOLO ainda não
definiu um contrato de grafo exportado para a tarefa de malha, incluindo como
representar o layout de parâmetros do MHR fora do PyTorch.

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O modelo corporal que os checkpoints acionam, o MHR (Momentum Human Rig), é uma
versão separada da Meta sob Apache-2.0. O LibreYOLO baixa o asset TorchScript
dele a partir da própria versão pública do MHR em tempo de execução e o guarda
em cache localmente; esse arquivo não é espelhado pelo LibreYOLO e segue os
próprios termos Apache-2.0, não a SAM License.

</provenance-box>

## Citação

<citation-block />
