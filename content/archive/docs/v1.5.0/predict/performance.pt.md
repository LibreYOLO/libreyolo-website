---
title: Desempenho da inferência
seo_title: Inferência mais rápida na LibreYOLO
description: >-
  Grafos CUDA, meia precisão, batching, inferência por blocos e test-time
  augmentation na hora de predizer, com os padrões reais e quais famílias
  suportam cada um.
lead: >-
  Cinco controles em tempo de predição mudam o throughput ou a acurácia:
  reprodução de grafos CUDA, precisão, batching, tiling e test-time
  augmentation. Cada um se aplica a um conjunto específico de famílias, e dois
  deles custam acurácia ou latência em vez de economizá-la.
keywords:
  - cuda graphs pytorch inferência
  - inferência em batch yolo python
  - inferência fp16
  - inferência por tiles objetos pequenos
  - inferência fatiada imagens grandes
  - test time augmentation detecção
  - capture_graph
  - predizer pasta de imagens yolo
last_verified: 1.5.0
verification: >-
  Padrões dos argumentos de InferenceRunner.__call__ em
  libreyolo/models/base/inference.py. API de grafos CUDA de
  BaseModel.capture_graph, graph_info, release_graphs e cuda_graph_scope em
  libreyolo/models/base/model.py; a adesão por família, da variável de classe
  SUPPORTS_CUDA_GRAPH. Comportamento de meia precisão de NOOP_PREDICT_KWARGS em
  libreyolo/utils/predict_args.py, o aviso da CLI em
  libreyolo/cli/commands/predict.py, e CAST_RECIPES mais SUPPORTED_FAMILIES em
  libreyolo/quant/api.py. Condições de batching de
  InferenceRunner._process_in_batches e _predict_batch. Tiling de _predict_tiled
  e _merge_tile_detections. Test-time augmentation de BaseModel._predict_augment
  e _merge_tta, com TTA_ENABLED, TTA_SCALES e TTA_FIXED_SIZE lidos em
  libreyolo/models/.
snippets:
  batch:
    - label: Inferência em batch sobre uma pasta
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # Um forward empilhado por chunk de 4 nas famílias que suportam.
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: 'Streaming, para a lista nunca se materializar'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: Capturar de antemão e depois reproduzir (precisa de CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Pague o warmup e a captura uma vez, fora da primeira requisição.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Capturar só quando um shape se repete (precisa de CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" espera um shape ser visto duas vezes, então trabalho de
        # uma única execução nunca paga pela captura.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Instalar o extra de exportação
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Exportar e carregar de volta, na precisão padrão'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Exportação FP16 (gere e rode isto em uma máquina com CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'FP16 no PyTorch, via uma receita de cast (precisa de CUDA)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Uma receita de cast não lê dados de calibração.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Inferência por blocos em uma imagem grande
      language: python
      code: >
        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O tiling só entra em ação quando a imagem é maior que o tamanho de
        entrada.

        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))

        large.save("large.jpg")


        model = LibreYOLO("LibreYOLO9s.pt")


        result = model("large.jpg", tiling=True, overlap_ratio=0.2)

        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Test-time augmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Os controles e seus padrões

Cada um deles é um argumento de `predict`, e todos vêm desligados por padrão.

| Argumento | Padrão | Efeito |
|---|---|---|
| `batch` | `1` | Imagens por forward pass, para fontes do tipo pasta e lista |
| `cuda_graph` | `False` | Reproduz o forward a partir de um grafo CUDA capturado |
| `tiling` | `False` | Divide uma imagem grande em blocos sobrepostos |
| `overlap_ratio` | `0.2` | Sobreposição entre blocos quando `tiling` está ligado |
| `augment` | `False` | Roda visões espelhadas e as mescla |
| `half` | | Aceito, avisado e ignorado |
| `device` | `None` | Move o modelo antes de predizer |

`imgsz` também afeta o custo, já que define a resolução em que o modelo roda,
mas é antes de tudo um argumento de acurácia e pertence ao modelo, não a esta
página.

## Batching

<code-tabs name="batch" />

`batch` vale para fontes do tipo pasta e lista. Com `batch=1`, cada imagem roda
um forward pass. Acima de `1`, cada chunk é pré-processado, empilhado em um
único tensor, rodado uma vez e depois fatiado de volta, para que o
pós-processamento de imagem única que cada família já tem receba o que espera.

O caminho empilhado só é usado quando tudo isto vale:

- `batch` é maior que `1`
- `tiling` está desligado
- o test-time augmentation não está ativo
- a família define `SUPPORTS_BATCHED_PREDICT`
- a rede subjacente não está em modo de treinamento

A última condição não é um detalhe técnico. Uma rede em modo de treinamento
normalizaria o chunk empilhado com estatísticas de batch entre imagens, deixando
que imagens do mesmo chunk alterassem as predições umas das outras, então essas
execuções continuam sequenciais.

`SUPPORTS_BATCHED_PREDICT` tem `true` como padrão. Estas famílias optam por
ficar de fora e rodam uma imagem por forward, independentemente de `batch`:
Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net,
LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body,
SwinIR, YOLOv1, ZipDepth, todo detector de vocabulário aberto e todo modelo de
visão e linguagem.

Há mais um fallback. Se o pré-processamento não devolver tensores
`(1, C, H, W)` uniformes, com shape, dtype e device iguais ao longo do chunk, o
chunk roda sequencialmente em vez de empilhar, então a correção nunca depende de
as imagens por acaso terem o mesmo tamanho.

Combine `batch` com `stream=True` em uma pasta grande para ter forwards em batch
sem manter todos os resultados na memória.

## Grafos CUDA

<code-tabs name="graphs" />

Um grafo CUDA grava um forward pass uma vez e o reproduz como um único launch.
Detectores pequenos gastam boa parte do tempo de batch 1 lançando kernels, então
juntar esses lançamentos é um ganho de throughput, e a saída da reprodução é bit
a bit idêntica à da execução eager.

`cuda_graph` aceita três valores. `False` é o padrão e não faz nada. `True`
captura no primeiro uso de cada shape de entrada. `"auto"` espera um shape se
repetir antes de capturar, então trabalho de uma única execução ou com shapes
variáveis nunca paga o custo da captura.

`capture_graph(imgsz=None, batch=1, dtype=None)` tira esse custo da primeira
requisição. Um grafo só vale para o shape exato que capturou, então o `batch`
aqui precisa bater com a forma como `predict` será chamado depois.

`graph_info()` informa os grafos capturados, as contagens de reprodução e
qualquer motivo pelo qual a execução caiu de volta para eager.
`release_graphs()` libera os grafos e seus buffers estáticos.

A captura exige CUDA e uma família que tenha aderido via `SUPPORTS_CUDA_GRAPH`,
porque precisa de um forward sem trabalho visível ao host, e isso é verificado
família por família. Pedir a captura em uma família que não aderiu levanta
`NotImplementedError` em vez de rodar eager silenciosamente.

Um grafo grava endereços de memória, não valores, então qualquer coisa que
realoque parâmetros o descarta. Trocar de device por `predict(device=...)`,
quantizar e desquantizar invalidam os grafos capturados.

A matriz completa de suporte por família, as divisões nas costuras e o contrato
de numéricos estão em [Grafos CUDA](/docs/reference/cuda-graphs).

## Precisão

<code-tabs name="precision" />

`half=True` na hora de predizer não faz nada. É aceito por compatibilidade com a
linha de comando, dispara um aviso dizendo que é um no-op e é descartado antes de
chegar a qualquer família. A flag `--half` da CLI imprime o mesmo aviso para um
modelo `.pt`.

Há dois caminhos reais para baixar a precisão.

Para um artefato exportado, a precisão é escolhida na hora da exportação com
`export(format=..., half=True)`, e o arquivo resultante volta a ser carregado por
`LibreYOLO()` sem mudanças.

Para a execução em PyTorch, `model.quantize(recipe="fp16")` faz o cast do modelo
para float16 e instala hooks que mantêm float32 nas entradas e saídas do modelo.
`"bf16"` faz o mesmo com bfloat16. Nenhum dos dois casts lê dados de calibração,
então `calib` é ignorado para eles. A quantização cobre hoje quatro famílias:
YOLOv9, RF-DETR, BiRefNet e FeyNobg. Um cast em um device de CPU registra um
aviso de que será lento, então essas receitas são feitas para GPU.

Os dois caminhos mudam os numéricos. Nenhum deles garante as mesmas detecções
como substituição direta, então valide antes de fazer deploy.

## Inferência por blocos

<code-tabs name="tiling" />

O tiling recorta uma imagem grande em blocos quadrados sobrepostos, prediz em
cada um e mescla os resultados. É a opção para objetos pequenos em imagens de
alta resolução, onde redimensionar a imagem inteira encolhe os alvos abaixo do
que o modelo consegue resolver.

O tamanho do bloco é o tamanho de entrada do modelo, ou `imgsz` quando
informado, e precisa ser quadrado. `overlap_ratio` tem `0.2` como padrão. Blocos
que se sobrepõem são reconciliados com supressão não máxima por classe no limiar
`iou`, e a lista mesclada é então truncada em `max_det`. Ou seja, `iou` tem
efeito sobre predições por blocos até para famílias que não rodam NMS próprio.

O tiling é pulado, não apenas barato, quando a imagem já cabe: se as duas
dimensões estão em ou abaixo do tamanho de entrada, roda um forward comum no
lugar. Também é pulado para classificação, segmentação semântica e a tarefa
`embed`, que caem para uma única passada porque o tiling não faz sentido ali.

Ele levanta erro para tarefas cuja saída não pode ser costurada de volta:
máscaras de segmentação de instâncias, caixas orientadas, pontos, profundidade,
bordas e normais. Não pode ser combinado com `augment`.

O resultado carrega `result.tiled` e `result.num_tiles`. Com `save=True`,
execuções por blocos escrevem um diretório em `runs/tiled_detections` com cada
bloco, a imagem anotada, uma visualização em grade e um `metadata.json`
registrando o tamanho do bloco, a sobreposição e os limiares, com
`result.tiles_path` e `result.grid_path` apontando para eles.

## Test-time augmentation

<code-tabs name="tta" />

`augment=True` roda a imagem mais de uma vez e mescla as detecções com supressão
não máxima por classe no limiar `iou`. Como no tiling, isso faz `iou` pesar para
famílias que de outro modo o ignoram.

Na prática, isso é espelhamento horizontal. A lista de escalas `TTA_SCALES` tem
como padrão uma única escala de `1.0` e nenhuma família distribuída a
sobrescreve, então toda família roda duas passadas: a imagem original e seu
espelho. Famílias marcadas com `TTA_FIXED_SIZE` redimensionam para um quadrado
fixo, o que de todo jeito torna o multiescala um no-op para elas.

A segmentação semântica e a panóptica usam outra mesclagem. A visão espelhada
delas é espelhada de volta e as duas distribuições softmax têm sua média
calculada antes do argmax, em vez de serem mescladas como caixas.

O test-time augmentation não está disponível para toda tarefa. Ele levanta erro
para caixas orientadas, pose, pontos, profundidade, normais, bordas,
restauração, OCR e modelos de embedding, e não pode ser combinado com tiling.

Estas famílias o desabilitam de vez, então `augment=True` roda uma única passada
comum: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS,
NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED,
toda variante do SAM, todo detector de vocabulário aberto e todo modelo de visão
e linguagem.

## Medição

Nada nesta página traz um número de latência, porque um milissegundo sem seu
hardware, runtime, precisão e tamanho de batch não é um fato. Números medidos em
diferentes hardwares e runtimes são publicados em
[visionanalysis.org](https://www.visionanalysis.org), e `libreyolo profile` mede
um modelo específico na máquina que está na sua frente.
