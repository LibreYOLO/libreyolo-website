---
title: API do modelo
seo_title: Métodos e assinaturas do objeto de modelo do LibreYOLO
description: >-
  Todos os métodos de um modelo LibreYOLO carregado: predict, embed, track, val,
  train, export, save, quantize, info e os controles de CUDA graph, com os
  defaults reais.
lead: >-
  Um modelo LibreYOLO carregado é uma instância de BaseModel. Esta página lista
  os métodos que essa instância carrega, com as assinaturas e os defaults lidos
  de libreyolo/models/base/model.py.
keywords:
  - métodos do modelo libreyolo
  - argumentos do predict libreyolo
  - argumentos do val libreyolo
  - exportar modelo libreyolo onnx
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Assinaturas e defaults lidos de libreyolo/models/base/model.py e
  libreyolo/models/base/inference.py na v1.5.0. As classes de família podem
  restringir ou estender esses argumentos; train() é definido por família e aqui
  só está documentado o wrapper cfg= compartilhado.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True devolve um gerador, um Results por frame ou imagem.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Construção

A factory devolve uma instância da classe de família. Construir essa classe
diretamente recebe os mesmos argumentos, exceto que `size` é obrigatório:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` seleciona CUDA quando disponível, depois MPS, depois CPU. Um
inteiro ou uma string de dígito é lido como ordinal CUDA, então `device=0` e
`device="0"` significam `cuda:0`. `task` é validado contra o `SUPPORTED_TASKS`
da família. Passar `model_path=None` constrói a arquitetura e a deixa em modo
de treinamento; passar um `dict` carrega esse state dict diretamente.

## predict e \_\_call\_\_

`predict` é um alias para `__call__`.

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

| Argumento | Default | Significado |
|---|---|---|
| `source` | `None` | Imagem, lista ou tupla de imagens em memória, diretório, arquivo de vídeo, ou uma fonte de tela como `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Limiar de confiança |
| `iou` | `0.45` | Limiar de IoU para o NMS |
| `imgsz` | `None` | Sobrescreve o tamanho de entrada; `None` usa o tamanho nativo do modelo |
| `device` | `None` | Sobrescreve o dispositivo para esta chamada |
| `classes` | `None` | Mantém apenas estes IDs de classe |
| `max_det` | `300` | Máximo de detecções por imagem |
| `augment` | `False` | Data augmentation em tempo de teste |
| `save` | `False` | Escreve uma imagem ou vídeo anotado |
| `batch` | `1` | Imagens por forward pass para fontes de diretório e lista |
| `stream` | `False` | Devolve um gerador em vez de uma lista materializada |
| `stream_buffer` | `False` | Mantém todo frame capturado ao vivo em vez de só o mais recente |
| `vid_stride` | `1` | Processa a cada N-ésimo frame de vídeo ou de tela |
| `show` | `False` | Exibe os frames anotados em uma janela |
| `output_path` | `None` | Caminho de saída quando `save=True` |
| `color_format` | `"auto"` | Dica de formato de cor para arrays em memória |
| `tiling` | `False` | Inferência por blocos (tiles) para imagens grandes |
| `overlap_ratio` | `0.2` | Proporção de sobreposição dos blocos |
| `output_file_format` | `None` | `"jpg"`, `"png"` ou `"webp"` |
| `cuda_graph` | `False` | `True` captura no primeiro uso por formato de entrada, `"auto"` espera um formato se repetir |

Uma fonte de imagem única devolve um `Results`. Uma lista, uma tupla ou um
diretório devolve uma lista deles, e `stream=True` devolve um gerador em todos
os casos.

Fontes de stream ao vivo são ilimitadas e exigem `stream=True`. `tiling` e
`augment` não podem ser combinados. O data augmentation em tempo de teste
levanta erro para as tarefas `embed`, `point` e `edge`.

<code-tabs name="usage" />

Com `batch > 1`, as famílias cujo `SUPPORTS_BATCHED_PREDICT` é verdadeiro
rodam um forward empilhado por bloco; `batch=1` mantém um forward por imagem.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Um wrapper de conveniência sobre `predict` que empilha cada linha de embedding
em um único tensor `(N_total, D)`. O modelo precisa ter sido construído com
`task="embed"`, caso contrário ele levanta `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Emite um `Results` por frame com `track_id` preenchido. `tracker` é
`"bytetrack"`, `"botsort"`, `"ocsort"` ou `"deepocsort"`, e é ignorado quando
`tracker_config` é informado, porque o tipo da config seleciona o tracker.
`track_conf` mapeia para `track_high_thresh` no ByteTrack e no BoT-SORT e para
`det_thresh` no OC-SORT e no Deep OC-SORT. `output_path` tem como default
`runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Devolve um dicionário de métricas cujas chaves dependem da tarefa; detecção
devolve `metrics/precision`, `metrics/recall`, `metrics/mAP50` e
`metrics/mAP50-95`. `imgsz` aceita um int quadrado ou uma tupla
`(height, width)` e tem como default o tamanho de entrada nativo do modelo.
`plots` é um alias para `save_plots`. `allow_download_scripts` controla o
Python embutido que o YAML de um dataset pode carregar no seu campo
`download`.

`faster_coco_eval` é aceito através de `**kwargs` e tem `True` como default,
caindo de volta para o pycocotools quando o pacote não está instalado. O
backend que rodou é reportado em `model.last_eval_backend`.

A validação com data augmentation levanta erro para as tarefas `obb` e `pose`.

## train

`train` é definido por família, então seus argumentos diferem. Dois
comportamentos são compartilhados, porque a classe base envolve o `train` de
toda família:

- `cfg=` recebe um caminho de YAML cujas chaves são mescladas na chamada. Os
  argumentos nomeados explícitos ganham do arquivo.
- `pretrained=False` em uma família do grupo de cobertura `g0` ou `g1`
  reinicializa o modelo do zero antes do treinamento, e não pode ser combinado
  com `resume=True`.

Quais controles de data augmentation uma família realmente respeita é uma
questão por família; veja a
[matriz de data augmentation](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Devolve o caminho do artefato escrito. `format` é resolvido pelo registro de
exportadores, onde `engine` é um alias para `tensorrt` e `litert` é um alias
para `tflite`. Argumentos compartilhados por todos os exportadores:

| Argumento | Default | Significado |
|---|---|---|
| `output_path` | `None` | Caminho do arquivo de saída; gerado em `weights/` quando omitido |
| `imgsz` | `None` | Tupla `(height, width)` ou um único int; default é o tamanho nativo |
| `opset` | `None` | Versão do opset ONNX |
| `simplify` | `True` | Roda a simplificação do grafo ONNX |
| `dynamic` | `True` | Habilita eixos dinâmicos |
| `half` | `False` | Precisão FP16 |
| `int8` | `False` | Precisão INT8 |
| `batch` | `1` | Tamanho de batch embutido no artefato |
| `device` | `None` | Dispositivo em que traçar |
| `data` | `None` | data.yaml para a calibração INT8 |
| `fraction` | `1.0` | Fração do dataset de calibração a usar |
| `allow_download_scripts` | `False` | Permite Python embutido nos downloads do YAML do dataset |
| `verbose` | `False` | Log detalhado do exportador |

Combinações bloqueadas levantam `NotImplementedError` no preflight, antes do
traçado. A cobertura e suas regras estão na página da
[matriz de exportação](/docs/reference/export-matrix). Quando há adaptadores
LoRA ativos, eles são fundidos nos pesos densos, e essa fusão acontece apenas
depois de toda rejeição de requisição.

## save

```python
model.save(path) -> str
```

Escreve um checkpoint LibreYOLO de esquema v1.0: o state dict mais os
metadados descritos no
[esquema de checkpoint](/docs/reference/checkpoint-schema). Um modelo
quantizado carrega adicionalmente o seu manifesto `quant`, de modo que
`LibreYOLO(path)` restaura a estrutura e as escalas quantizadas.

## quantize, quant_info e dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantiza in place e devolve o modelo. `recipe` é um dos casts `fp16` e `bf16`,
das receitas de Conv e Linear `int8` e `fp8`, ou das receitas só de Linear
`w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2`, que famílias transformer como a
RF-DETR suportam. `int2` exige QAT. `calib` recebe um caminho de data.yaml ou
o nome de um dataset embutido e lê as imagens apenas para a frente; as labels
nunca são lidas. Passe `calib=None` para pular a calibração. `algorithm` é
`"minmax"`, `"percentile"` ou `"auto"`.

`model.quant_info()` devolve o resumo do estado de quantização, ou `None` para
um modelo float. `model.dequantize()` restaura os módulos float in place
mantendo os pesos mestres treinados com quantização, o que é a ponte do QAT
para `export(format="onnx", int8=True, data=...)`.

## info e camadas

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` devolve um dicionário amigável a JSON e loga um resumo legível por
humanos quando `verbose` é verdadeiro. `get_available_layer_names` lista as
camadas que uma config de destilação ou de extração de características pode
nomear.

## CUDA graphs

Disponível nas famílias cujo atributo de classe `SUPPORTS_CUDA_GRAPH` é
verdadeiro. O replay é bit a bit idêntico à execução eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Um grafo capturado só é válido para o formato exato em que foi capturado,
então `batch` e `imgsz` precisam bater com a chamada posterior de `predict`.
`capture_graph` tira o custo da captura da primeira requisição. `mode` aceita
`True` ou `"on"` para capturar no primeiro uso, `"auto"` para esperar até um
formato se repetir, e `False` para não fazer nada. `capture_graph` levanta
`NotImplementedError` quando a família não optou por esse suporte e
`CudaGraphUnavailable` quando a captura falha.

## Dispositivo e dtype

Os objetos `Results` carregam `.to()`, `.cpu()`, `.cuda()` e `.numpy()`; veja
[Tipos de Results](/docs/reference/results-types). O próprio modelo é movido
passando `device=` para `predict`, ou no momento da construção.
