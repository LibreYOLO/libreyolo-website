---
title: NVIDIA DeepStream
seo_title: Rodar modelos YOLO no NVIDIA DeepStream
description: >-
  Exporte um modelo LibreYOLO para o NVIDIA DeepStream: um grafo ONNX mais um
  config do nvinfer gerado. Comandos exatos para compilar o parser e para o
  pipeline.
lead: >-
  O NVIDIA DeepStream roda a inferência pelo seu elemento nvinfer, que precisa
  de um grafo ONNX, de um arquivo de configuração correspondente e de um parser
  de bounding boxes. Colocar deepstream=True na exportação para ONNX escreve os
  dois primeiros e os liga ao terceiro.
keywords:
  - deepstream yolo
  - exportar yolo para deepstream
  - nvinfer config yolo
  - parser de bounding box deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - engine tensorrt deepstream
  - jetson deepstream
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Escreve
    value: 'Um grafo ONNX, config_infer_primary_<stem>.txt e <stem>_labels.txt'
  - label: Cobertura
    value: 43 combinações de família e tarefa distribuídas em nove tarefas
  - label: Parser
    value: >-
      NvDsInferParseYolo, do projeto DeepStream-Yolo de Marcos Luciano,
      licenciado sob MIT. Compilado uma vez por dispositivo.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Disponibilidade
    value: Chega na v1.5.0. Mesclado no dev em 2026-08-08 na pull request 728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Validado em execução
    value: 'DeepStream 8.0.0 em uma RTX 5070 Ti, só detecção, 2026-08-08'
verification: >-
  Escrito a partir da validação em execução de 2026-08-08. As listas de
  famílias, as chaves de configuração e os valores padrão foram lidos de
  libreyolo/export/deepstream.py e libreyolo/export/exporter.py no commit
  5f81e11e, que foi mesclado no dev no mesmo dia na pull request 728.
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Escreve libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # e libreyolo9s_labels.txt no diretório de trabalho.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Mantenha cada modelo de detecção em seu próprio diretório: todo config

        # de detecção nomeia o mesmo arquivo de cache do engine. Veja
        "Armadilhas conhecidas".

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argumentos
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True é rejeitado em todos os outros formatos
            deepstream=True,
            conf=0.25,         # alimenta pre-cluster-threshold (e classifier-threshold,
                               # segmentation-threshold nessas tarefas)
            iou=0.45,          # alimenta nms-iou-threshold, omitido em cluster-mode=4
            batch=1,           # alimenta batch-size e o nome do arquivo de cache do engine
            half=False,        # True marca o config como network-mode=2 (build fp16)
            int8=False,        # True marca o config como network-mode=1
            dynamic=True,      # eixo de batch dinâmico no grafo ONNX
            imgsz=640,         # alimenta infer-dims=3;H;W
        )


        # deepstream=True e nms=True são mutuamente exclusivos: o DeepStream
        roda a

        # supressão na sua etapa de clustering, então nada é embutido no grafo.
    - label: Baixe os pesos do D-FINE primeiro
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Confirme o passthrough da GPU antes de qualquer outra coisa
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, rode dentro do container do DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 nesta imagem é um stub e o build morre nele com

        # "fatal error: crt/host_defines.h: No such file or directory". Resolva
        um

        # toolkit que realmente traga o header; na imagem 8.0 esse é o
        cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # A imagem traz libcublas.so.12 e libcublas.so.12.8.4.1, mas não o

        # libcublas.so sem versão de que -lcublas precisa, então a etapa de link
        falha

        # com "/usr/bin/ld: cannot find -lcublas". Dê ao linker os nomes que ele
        quer.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: A segmentação de instâncias usa um parser diferente
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Rode
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Os dois passos em um só container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Disponibilidade

A exportação para DeepStream chega na v1.5.0. Foi mesclada no `dev` em 2026-08-08
na pull request 728, então uma instalação atual já a tem e não é preciso fixar
nenhum branch.

<code-tabs name="install" />

Se você clonou o branch `deepstream-export` antes de 2026-08-08, substitua. Esse
branch foi rebaseado e enviado com force-push, e o histórico antigo não tem o
conserto que faz essas exportações rodarem em uma máquina com CUDA.

## O que a exportação escreve

`model.export(format="onnx", deepstream=True)` escreve três arquivos lado a lado.
Para `libreyolo9s.pt`:

- `libreyolo9s.onnx`, o grafo de detecção, um tensor de saída de forma
  `(batch, num_detections, 6)`, cada linha `[x1, y1, x2, y2, score, class_id]` em
  coordenadas de pixel da entrada da rede.
- `config_infer_primary_libreyolo9s.txt`, uma configuração do `nvinfer` que carrega
  as constantes de pré-processamento da família, a contagem de classes, os limiares
  e a ligação com o parser.
- `libreyolo9s_labels.txt`, um nome de classe por linha.

Um arquivo de rótulos aparece sempre que o checkpoint carrega nomes de classe. Os
modelos de profundidade não têm nenhum, então não recebem nem o arquivo nem uma
chave `labelfile-path`.

A LibreYOLO não emite nenhum `.so`. O `.so` que o DeepStream carrega é o parser de
bounding boxes de `marcoslucianops/DeepStream-Yolo`, compilado uma vez por
dispositivo, e é o mesmo binário para qualquer detector da LibreYOLO que você
apontar para ele. O modelo é o ONNX. Classificação e segmentação semântica não
precisam de parser nenhum, porque o `nvinfer` faz esse pós-processamento sozinho.

## Exportar o modelo

<code-tabs name="export" />

`LibreDFINE._load_weights` levanta `FileNotFoundError` quando o arquivo ainda não
está em disco, sem tentar baixá-lo, então baixe `LibreDFINEs.pt` você mesmo antes.
Essa lacuna está registrada no
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Os pesos do YOLO9
são baixados no primeiro uso.

A flag é só de Python. `libreyolo export` neste branch não tem opção `deepstream`,
e a CLI monta seus argumentos de exportação a partir de uma lista fixa em vez de
repassar chaves desconhecidas.

## Compilar o parser de bounding boxes

A detecção precisa da biblioteca do parser, a segmentação de instâncias precisa de
outra, e as demais tarefas não precisam de nenhuma. Duas coisas na imagem do
DeepStream 8.0 quebram o comando de build documentado, e ambas são problemas de
ambiente, não da LibreYOLO.

A imagem traz `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` e `cuda-12.9` em
`/usr/local`. Só `cuda-12.5` tem um toolkit completo. Ela também traz
`libcublas.so.12` e `libcublas.so.12.8.4.1`, mas não o `libcublas.so` sem versão
contra o qual `-lcublas` resolve. O script abaixo contorna as duas coisas.

<code-tabs name="parser" />

Depois aponte `custom-lib-path` no config gerado para o
`libnvdsinfer_custom_impl_Yolo.so` compilado. O valor gerado é o caminho relativo
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, que resolve quando
o `deepstream-app` roda a partir do checkout do `DeepStream-Yolo` e precisa ser
editado nos outros casos.

## Rodar o pipeline

Confira que o container enxerga a GPU antes de gastar tempo com qualquer outra
coisa. Essa foi a primeira checagem que a execução de validação fez, em uma placa
Blackwell sob WSL2.

<code-tabs name="gpu" />

A execução de validação tocou o `deepstream-app` com uma fonte de arquivo, sem
sink de display, com o on-screen display ligado e com `gie-kitti-output-dir`
definido para que as detecções de cada frame caíssem em disco como texto KITTI.
Um config com esses ajustes:

<code-tabs name="run" />

O `nvinfer` constrói o engine do TensorRT a partir do ONNX na primeira execução e
o guarda em cache ao lado do modelo, então a primeira execução paga o build do
engine e as seguintes carregam o cache.

## O config gerado

Os dois configs abaixo foram escritos pelo exportador para a execução de
validação, sem edição depois.

| Chave | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Os dois configs diferem em três pontos: `maintain-aspect-ratio`, `cluster-mode`, e
se `nms-iou-threshold` está presente ou não. O config do D-FINE omite essa chave
por completo, que é o que o `cluster-mode=4` pede.

Cabeças que emitem no máximo uma predição por objeto recebem `cluster-mode=4`,
então o DeepStream não roda clustering sobre elas; o clustering fundiria detecções
genuinamente distintas. Isso cobre `rfdetr`, `dfine`, `deim`, `deimv2`, `ec`,
`rtdetr`, `rtdetrv2`, `rtdetrv4` e `yolo9_e2e`. Cabeças de grid e com âncoras
recebem `cluster-mode=2` mais `nms-iou-threshold`.

Os configs de detecção também carregam
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, que entrega a construção do
engine à biblioteca do parser. É isso que fixa o nome do arquivo de cache do
engine, e é a origem da colisão descrita nas armadilhas conhecidas.

## Tarefas e famílias suportadas

Quarenta e três combinações de família e tarefa exportam.
`deepstream_supported_tasks()` e `deepstream_supported_families(task)` em
`libreyolo/export/deepstream.py` devolvem as mesmas listas em tempo de execução.

| Tarefa | `network-type` | Biblioteca do parser | Famílias |
|---|---|---|---|
| Detecção | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Classificação | 1 | Nenhuma | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentação semântica | 2 | Nenhuma | pidnet, eomt, dinov2, lingbotvision |
| Segmentação de instâncias | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | Nenhuma | yolo9, yolonas, rfdetr, ec |
| Profundidade | 100 | Nenhuma | depth_anything, zipdepth |
| Restauração | 100 | Nenhuma | nafnet, realesrgan, swinir |
| Matting | 100 | Nenhuma | birefnet |
| Gaze | 100 | Nenhuma | l2cs |

`network-type=100` quer dizer que o DeepStream não tem pós-processador para a
tarefa. Esses configs definem `output-tensor-meta=1`, as saídas nativas do grafo
passam sem alteração, e a aplicação as decodifica a partir dos metadados do tensor.
Grafos com várias saídas não dão problema ali: toda camada de saída chega aos
metadados com os mesmos nomes de saída e os mesmos eixos dinâmicos de uma
exportação ONNX comum.

As linhas de segmentação de instâncias são a linha de detecção seguida da máscara
daquela instância, achatada em `(netH / 4, netW / 4)`, que é a resolução que o
parser de segmentação tem fixada no código, como probabilidades para
`segmentation-threshold`.

Classificação e gaze rodam como inferência secundária. Defina `process-mode=2` e
`operate-on-gie-id` no config gerado para colocar um classificador atrás de um
detector. O gaze é um contrato só de cabeça, um recorte de rosto por entrada,
então precisa de um detector de rostos na frente.

Três famílias estão ausentes de propósito. `segformer` não está ligada ao contrato
compartilhado de exportação semântica e não consegue exportar para ONNX em formato
nenhum. RTMDet-Ins e YOLO9 têm a exportação de segmentação de instâncias bloqueada
dentro da própria LibreYOLO. `depth_anything3` não tem implementação de exportação.

Duas linhas da tabela têm lacunas de checkpoint por trás. Só o checkpoint semântico
`l` do EoMT está publicado, e a classificação com DINOv2 não tem checkpoint
publicado nenhum, então essa combinação precisa de pesos ajustados por você.

## Diferenças de pré-processamento

O `nvinfer` calcula `net-scale-factor * (x - offsets)` por canal com uma escala
escalar, que não consegue expressar desvio padrão por canal. As famílias que
precisam de um (`rfdetr`, `ec`, os tamanhos de `deimv2` com backbone DINO,
`rtmdet`, `picodet` e todas as famílias de classificação) têm a normalização
embutida no grafo exportado, e o config gerado alimenta o grafo com o espaço de
entrada bruto correspondente.

A geometria é onde os pipelines em Python da própria LibreYOLO e o `nvinfer` ainda
divergem:

- Famílias com letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) preenchem com cinza nativamente. O `nvinfer` preenche com
  preto.
- A detecção com `yolonas` nativamente redimensiona o lado mais longo para 636
  dentro do seu canvas de 640. O `maintain-aspect-ratio` do `nvinfer` usa os 640
  inteiros.
- A classificação nativamente redimensiona o lado mais curto e depois recorta pelo
  centro. O `nvinfer` estica o frame ou a ROI do objeto até a entrada da rede,
  então sujeitos com recorte apertado saem diferentes.
- O EoMT nativamente roda tiles de janela deslizante para segmentação semântica. O
  grafo exportado é um único canvas esticado, o que é mais rápido e menos preciso.
- `pidnet` emite um mapa de classes a 1/8 da resolução de entrada e
  `lingbotvision` a 1/16. O DeepStream faz upsampling do mapa de classes para
  exibição.

A checagem de paridade do ONNX alimenta tensores já pré-processados, então ela
confere as saídas do grafo e não pega uma ordem de cor ou uma política de padding
erradas no config. Valide com os seus próprios dados antes de fazer deploy de uma
carga que exija paridade exata.

## Armadilhas conhecidas

### Dois modelos de detecção em um diretório carregam o engine um do outro

Todo config de detecção carrega a mesma linha:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

O construtor de engines do parser exige esse nome base e ele não varia por modelo.
Exporte um segundo modelo de detecção para o mesmo diretório e a segunda execução
carrega o engine em cache do primeiro modelo. Nada quebra; os boxes é que saem
errados. Dê a cada modelo de detecção o seu próprio diretório. A execução de
validação teve que isolar o D-FINE em um antes de conseguir testá-lo.

### Um box só pode carregar uma classe

O formato de linha do `nvinfer` é `[x1, y1, x2, y2, score, class_id]`, uma classe
por box, então a exportação colapsa as pontuações de classe no argmax delas. Um box
que o `predict` reporta sob duas classes sobrevive sob uma. Caso medido: a LibreYOLO
reporta `vase 0.773` e `bottle 0.383` no mesmo box, e o grafo do DeepStream fica com
`vase`. Isso decorre do formato de linha do parser e não dá para mudar sem sair
desse contrato, então é comportamento esperado, não uma regressão.

## Validado

O `deepstream-app` rodou até o EOS com `App run successful` nos dois tipos de
cabeça de detector, sobre o `sample_1080p_h264.mp4` que a NVIDIA traz (1443
frames), com os dumps KITTI por frame ligados.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Tipo de cabeça | grid | um para um |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frames com detecções | 1443 | 1443 |
| Total de detecções | 18031 | 71105 |

Os histogramas de classe sobre os 1443 frames colocam carros em primeiro e pessoas
em segundo nos dois modelos, o que está certo para uma cena de rua. A diferença de
quatro vezes na contagem de detecções é a diferença de `cluster-mode` fazendo o seu
trabalho: o D-FINE em `cluster-mode=4` não roda clustering, então toda query acima
do limiar sobrevive, quase duplicadas incluídas.

Dois modelos treinados de forma independente colocam o objeto dominante no mesmo
lugar:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Essa execução estabelece cinco coisas: o TensorRT constrói um engine a partir do
ONNX exportado em sm_120, o `nvinfer` aceita toda chave do config gerado, o
`NvDsInferParseYolo` lê o layout do tensor corretamente, os boxes caem em
coordenadas 1920x1080 da resolução de origem, e os rótulos resolvem contra o
arquivo de rótulos gerado.

O ambiente em que ela rodou:

| Componente | Valor |
|---|---|
| SO do host | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Capacidade de computação | 12.0 (Blackwell, sm_120) |
| Runtime de container | Docker Desktop 29.4.3, backend WSL2 |
| Imagem do DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Versão do DeepStream | 8.0.0 |
| CUDA do container | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` no HEAD |

Junto com a execução do pipeline, `tests/unit/test_deepstream_export.py` cobre os
adaptadores de grafo e as chaves do config gerado, e seus 35 testes passam neste
commit.

## Não validado

Dito para que o escopo acima não seja lido como mais amplo do que é.

- Jetson e aarch64. O contrato de exportação não depende da arquitetura, mas o
  pipeline só foi rodado em uma GPU discreta x86.
- Quarenta e uma das 43 combinações. Só detecção com `yolo9` e detecção com
  `dfine` passaram pelo DeepStream. Classificação, segmentação semântica,
  segmentação de instâncias e as tarefas de tensor bruto estão cobertas por testes
  unitários e checagens de paridade ONNX, não por uma execução do pipeline.
- FP16 e INT8. Só `network-mode=0` foi exercitado.
- Multi-stream e batching. Uma fonte, `batch-size=1`.
- Acurácia contra um dataset de ground truth. As detecções foram conferidas por
  plausibilidade semântica e por concordância entre modelos, não pontuadas como mAP
  através do DeepStream.
