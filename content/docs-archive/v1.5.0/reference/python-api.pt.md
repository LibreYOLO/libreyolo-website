---
title: API Python
seo_title: Referência da API Python do LibreYOLO
description: >-
  Os nomes que o LibreYOLO exporta no nível do pacote: as cinco factories, as
  classes de família, os payloads de Results, backends, validadores, trackers e
  helpers de dados.
lead: >-
  A superfície pública em Python do LibreYOLO é a lista __all__ em
  libreyolo/__init__.py. Tudo nesta página é importável como from libreyolo
  import <name>; qualquer coisa fora dessa lista é interna.
keywords:
  - api python libreyolo
  - importar libreyolo python
  - factory LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Nomes e assinaturas lidos de libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py e
  libreyolo/ensemble/model.py na v1.5.0.
snippets:
  usage:
    - label: Carregar qualquer coisa por uma única factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Uma fonte de imagem única retorna um Results; uma lista ou um
        # diretório retorna uma lista deles.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Importar uma classe de família diretamente
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Os cinco pontos de entrada
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Factory que fareja os pesos, sobre as famílias sem prompt.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Dois ou mais detectores atrás de uma única superfície de predição.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # As outras três factories precisam de um extra instalado:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Pontos de entrada

Cinco chamáveis carregam um modelo. Eles são separados pelo contrato de
chamada, não pela arquitetura.

| Factory | Carrega | Prompt no momento da chamada | Extra necessário |
|---|---|---|---|
| `LibreYOLO` | Famílias sem prompt, farejando o checkpoint ou o sufixo do arquivo | | |
| `LibreSAM` | Segmentadores com prompt, por alias de tamanho | Pontos, boxes ou texto de conceito | `sam` |
| `LibreVLM` | Detectores generativos de visão e linguagem, por alias | Vocabulário de classes ou um prompt livre | `vlm` |
| `LibreOpenVocab` | Detectores condicionados por texto, por alias | Vocabulário de classes | `openvocab` |
| `LibreEnsemble` | Dois ou mais detectores, fundidos em uma única superfície | | |

<code-tabs name="factories" />

`LibreYOLO` é a única que lê um arquivo. As outras três recebem um alias em
string e o resolvem para um repositório do Hugging Face, então o argumento é um
nome de modelo e não um caminho.

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path` aceita um checkpoint `.pt`, um arquivo ONNX `.onnx`, um
ExecuTorch `.pte`, um MNN `.mnn`, uma engine TensorRT `.engine`, um diretório
OpenVINO, Paddle ou ncnn, ou uma URL HTTP ou HTTPS de modelo no Triton. `size`
e `nb_classes` são lidos do checkpoint quando omitidos. `compute_units` só é
lido em cargas de CoreML `.mlpackage` e é um de `all`, `cpu_only`,
`cpu_and_gpu`, `cpu_and_ne`. `task` aceita qualquer nome canônico de tarefa de
`libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Classes de família

Toda família que a factory pode retornar também é exportada por nome, então uma
classe pode ser construída diretamente quando o checkpoint é conhecido de
antemão. Os construtores seguem `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` não tem valor padrão em uma classe de família, que é a diferença em
relação à factory. YOLO9 e suas variantes inserem `reg_max: int = 16` depois de
`size`.

Famílias de detecção e multitarefa: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Famílias de predição densa: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Famílias de classificação e embedding: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Outras tarefas: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Os tiers irmãos também exportam suas classes de família: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (também escrito
`LibreModus`).

## Superfície de predição

Chamar um modelo roda a inferência. `predict` é um alias de `__call__`, então
os dois são intercambiáveis.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

Uma fonte de imagem única retorna um `Results`. Uma lista, uma tupla ou um
diretório retorna uma lista deles, e `stream=True` retorna um gerador. Os
outros métodos do objeto de modelo estão documentados na
[página da API do modelo](/docs/reference/model-api).

## Payloads de Results

`Results` e suas dezoito classes de payload são exportados no nível do pacote:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Cada uma é descrita em [Tipos de Results](/docs/reference/results-types).

## Backends

Artefatos exportados carregam por `LibreYOLO()` a partir do sufixo do arquivo,
então as classes de backend raramente são construídas na mão. Elas são
exportadas para os casos em que um backend precisa ser escolhido
explicitamente: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`,
`TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, mais
`create_triton_config`. `BaseExporter` é o registro de exportadores por trás de
`model.export()`.

## Validadores

`model.val()` despacha para o validador certo conforme a tarefa, então estes
são exportados para uso direto e para herança: `DetectionValidator`,
`SegmentationValidator`, `PoseValidator`, `SemanticValidator`,
`PanopticValidator`, `DepthValidator`, `NormalValidator`, `EdgeValidator` e o
`ValidationConfig` compartilhado.

## Rastreamento

`model.track()` escolhe um tracker pelo nome. As classes de tracker e suas
dataclasses de configuração também são exportadas: `ByteTracker` com
`TrackConfig`, `BoTSortTracker` com `BoTSortConfig` e `OCSortTracker` com
`OCSortConfig`.

## Helpers de dados

`DATASETS_DIR` é a raiz resolvida dos datasets, `load_data_config` lê um YAML
de dataset e `check_dataset` valida um. Os loaders específicos de tarefa
citados em [formatos de dataset](/docs/reference/dataset-formats) ficam em
`libreyolo.data` e não no nível do pacote.

## Galerias e destilação

`Gallery` e `FaceGallery` guardam os vetores de identidade cadastrados para a
tarefa `embed` e produzem o payload `Identities`. `Distiller` e
`get_distill_config` conduzem o treinamento professor-aluno.

## Assets

`SAMPLE_IMAGE` é um caminho absoluto para uma imagem embutida no pacote, então
todo snippet destes docs roda sem precisar baixar uma figura antes.

## Imports lazy e classes renomeadas

A maioria dos nomes de tier irmão, os backends, os validadores e os helpers de
dados resolvem pelo `__getattr__` de nível de módulo, então importar
`libreyolo` não importa as dependências deles. O import ainda falha com uma
mensagem clara quando o extra necessário está faltando.

Dois nomes de classe foram renomeados e a grafia antiga ainda resolve, com um
`DeprecationWarning`: `LibreYOLORTDETR` agora é `LibreRTDETR`, e
`LibreYOLORFDETR` agora é `LibreRFDETR`.
