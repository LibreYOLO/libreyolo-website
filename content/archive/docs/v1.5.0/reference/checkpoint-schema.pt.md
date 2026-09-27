---
title: Esquema de checkpoint
seo_title: Esquema de metadados de checkpoint do LibreYOLO v1.0
description: >-
  Os metadados que todo checkpoint .pt do LibreYOLO carrega: chaves
  obrigatórias, acréscimos por tarefa, chaves de runtime de exportação,
  manifestos de quantização e campos de treinamento.
lead: >-
  Um arquivo .pt do LibreYOLO é um dicionário plano salvo com torch.save. A
  chave model guarda o state dict; as demais chaves de primeiro nível são
  metadados que identificam o checkpoint sem precisar analisar o nome do arquivo
  nem farejar o state dict.
keywords:
  - esquema de checkpoint libreyolo
  - metadados de checkpoint pytorch
  - schema_version 1.0
  - model_family
  - manifesto quant quantização
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Espelha docs/checkpoint_schema.md no repositório libreyolo na v1.5.0,
  conferido com libreyolo/utils/serialization.py e BaseModel.save.
snippets:
  usage:
    - label: Ler os metadados de um checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Baixe um checkpoint e salve de novo para que exista um caminho local.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Esquema v1.0

Todo checkpoint `.pt` oficial do LibreYOLO contém:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Chave | Tipo | Significado |
|---|---|---|
| `model` | state dict | Os pesos do modelo |
| `schema_version` | str | Versão do contrato de metadados; a v1.0 usa a string `"1.0"` |
| `libreyolo_version` | str | A versão que produziu o checkpoint |
| `model_family` | str | Uma família registrada, como `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variante dentro da família, como `t`, `s`, `r18`, `atto` |
| `task` | str | Nome canônico da tarefa |
| `nc` | int | Contagem positiva de classes |
| `names` | dict | `dict[int, str]` com chaves em `0..nc-1` |
| `imgsz` | int | Resolução de entrada quadrada positiva, ou o escalar legado para um contrato retangular |

`task` é um destes: `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` ou `mesh`.

Os checkpoints oficiais escrevem todas as chaves de `names`. Os leitores podem
preencher chaves ausentes com rótulos `class_i` para mapeamentos esparsos
legados, mas chaves fora do intervalo são inválidas.

Checkpoints retangulares mantêm um `imgsz` escalar para leitores legados,
definido como `max(imgsz_h, imgsz_w)`, e escrevem adicionalmente `imgsz_h` e
`imgsz_w` com as dimensões reais. Um leitor que entenda os campos retangulares
deve preferi-los ao escalar. Famílias com contrato retangular fixo, como o pose
da HRNet, rejeitam tamanhos de runtime incompatíveis.

O esquema é deliberadamente plano, e `model` é deliberadamente um state dict.

<code-tabs name="usage" />

## Acréscimos de pose

Pose costuma ser de classe única, `nc: 1` com `person`, mas a cabeça de pose do
YOLO-NAS também suporta pose multiclasse com um único esqueleto de keypoints
compartilhado, caso em que `nc` e `names` descrevem as classes como na detecção.
Exportações de pose em runtime emitem `scores` com shape `[batch, anchors, nc]`.

| Chave | Significado |
|---|---|
| `num_keypoints` | Contagem positiva de keypoints usada pela cabeça de pose |
| `keypoint_dim` | `2` para labels `x,y` ou `3` para labels `x,y,visibility`; as saídas do modelo sempre expõem `x,y,visibility` |
| `oks_sigmas` | Sigmas OKS opcionais por keypoint; o padrão da tarefa para `num_keypoints` é usado quando ausente |
| `num_keypoints_per_class` | Contagens opcionais de keypoints por classe para cabeças no estilo GroupPose cujo tensor de keypoints é preenchido por classe; `0` para classes sem keypoints |

## Acréscimos de mesh

Checkpoints de mesh usam `task: "mesh"`, `nc: 1` e `names: {0: "person"}`. Os
layouts de parâmetros diferem entre modelos de corpo, então as dimensões são
registradas em vez de presumidas.

| Chave | Significado |
|---|---|
| `body_model` | A parametrização, como `mhr`; obrigatória, e usada para interpretar todos os campos abaixo |
| `num_betas` | Contagem de coeficientes de identidade e forma; 45 para MHR |
| `num_body_pose` | Largura do bloco de parâmetros de pose do corpo; 130 para MHR. Um vetor plano, não um trio por junta, porque as juntas do rig têm graus de liberdade diferentes |
| `num_vertices` | Contagem de vértices que o decodificador emite; 18439 para MHR |
| `num_joints` | Contagem de juntas que o decodificador emite; 127 para MHR |
| `rotation_format` | Como as rotações são codificadas, por exemplo `euler_zyx` para MHR ou `axis_angle`. Nunca inferido do shape do tensor, já que um vetor de 3 é ambíguo |

## Placeholders de tarefas densas

Várias tarefas predizem mapas densos em vez de classes, então os slots
parecidos com classe existem apenas por compatibilidade de esquema.

| Tarefa | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Predições de bordas são mapas densos de probabilidade float32 em `[0, 1]`.

Checkpoints de restore podem adicionar `degradation`, um rótulo curto de
corrupção como `deblur`, `denoise` ou `super-resolution`; `dataset`, um rótulo
de proveniência como `GoPro` ou `SIDD`; e `scale`, um fator inteiro positivo de
aumento de escala da saída em relação à entrada, por exemplo `4` para um modelo
de super-resolução x4. Ausente ou `1` significa que a imagem restaurada mantém a
resolução de entrada. O runtime também deriva a escala a partir da família e do
tamanho, então `scale` é metadado de proveniência e não um requisito de
carregamento.

## Acréscimos de OCR

A família `ppocr` distribui um checkpoint composto por tier, cujo state dict de
`model` guarda dois submodelos sob os namespaces de chave `det.*` e `rec.*`.

| Chave | Significado |
|---|---|
| `charset` | O alfabeto CTC completo na ordem dos índices de saída: o índice 0 é o blank do CTC, depois o dicionário de reconhecimento, depois o caractere de espaço. Os loaders precisam lê-lo do checkpoint, nunca de um arquivo à parte |
| `pipeline` | Padrões do pipeline fixados no momento da conversão: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Argumentos de runtime podem sobrescrevê-los a cada chamada |
| `components` | Reservado para etapas opcionais do pipeline, como orientação de documento, desdobramento e rotação de linha de texto. Vazio na v1 |

## Metadados de runtime de exportação

Os artefatos exportados usam a mesma convenção de escrita dupla para o caso
retangular: `imgsz_h` e `imgsz_w` são escritos ao lado do escalar legado
`imgsz`, e um leitor que não entenda os campos retangulares não deve tratar
silenciosamente o escalar como um contrato quadrado.

O suporte retangular em runtime tem escopo por família e por formato.
Exportações da família YOLO9, HRNet, NAFNet e Real-ESRGAN podem usar `imgsz_h` e
`imgsz_w` não quadrados nos formatos suportados; famílias ou formatos sem
suporte retangular explícito rejeitam os metadados em vez de pré-processar esses
artefatos como quadrados. As exportações de HRNet são cabeças fixas, de batch
um, FP32 e sobre recorte de pessoa, em que a W32 aceita 256x192 e a W48 aceita
384x288, e o detector de pessoas não está embutido no grafo.

Exportações com NMS embutido podem adicionar estas chaves planas:

| Chave | Significado |
|---|---|
| `nms` | Booleano em string; `"true"` significa que o grafo inclui uma saída de pós-processamento embutida |
| `nms_conf` | Limiar de confiança fixado na saída embutida |
| `nms_iou` | Limiar de IoU fixado na saída embutida |
| `max_det` | Máximo de linhas de detecção pós-NMS que a saída embutida emite |
| `nms_raw_output` | Booleano em string; `"true"` significa que o grafo também expõe uma saída auxiliar bruta do detector |

Para exportações ONNX de detecção da YOLO9 com `nms=true`, a saída `0` (chamada
`output`) é o tensor pós-NMS autônomo nos limiares definidos na exportação.
Quando `nms_raw_output=true`, a saída `1` (chamada `raw`) fica reservada para os
backends do LibreYOLO, para que eles possam aplicar o recorte nativo no canvas
original e a semântica de runtime de `predict(conf=..., iou=..., max_det=...)`.
Consumidores de terceiros devem usar a primeira saída.

Exportações de pose podem adicionar `num_keypoints`; `keypoint_dim`, em que
exportações brutas no estilo GroupPose podem usar valores maiores, como `8`,
quando o tensor inclui campos de precisão ou de logits de classe;
`num_keypoints_per_class` como uma lista codificada em JSON, em que os slots de
classe com zero keypoints precisam ser preservados porque definem o esquema; e
`pose_input`, em que `"person_crop"` significa que o grafo consome um recorte já
extraído e não contém detector. As exportações de runtime da HRNet exigem esse
valor.

Exportações de classificação podem adicionar `crop_pct`, uma razão float de
recorte central cujo alvo de redimensionamento antes do recorte é
`round(imgsz / crop_pct)` e que assume `0.875` por padrão quando ausente, e
`interpolation`, `"bilinear"` ou `"bicubic"`, com padrão `"bilinear"`.

Exportações ExecuTorch escrevem os metadados planos em um sidecar
`<program>.pte.json` obrigatório. O contrato da v1 é CPU, FP32, batch 1 e canvas
de entrada fixo, e exige adicionalmente `executorch_version`,
`executorch_delegate` igual a `"xnnpack"` e um `executorch_delegate_partitions`
positivo. O loader rejeita um sidecar que declare outro delegate, shapes
dinâmicos ou precisão diferente de FP32.

Exportações MNN escrevem os metadados planos em um sidecar `<model>.mnn.json`
obrigatório. O contrato da v1 é CPU, FP32, somente detecção e shape de entrada
NCHW fixo, e exige adicionalmente `mnn_version`, `mnn_backend` igual a `"cpu"`,
`mnn_input_names` e `mnn_output_names` ordenados e não vazios, `mnn_input_shape`
como quatro inteiros positivos na ordem `[batch, channels, height, width]`, e
`mnn_batch` igual a `mnn_input_shape[0]`. O loader rejeita metadados de shape
dinâmicos, não FP32, que não sejam de detecção, de família não suportada ou
inconsistentes.

Um `.pte` e um `.mnn` são artefatos específicos de backend, não checkpoints do
PyTorch.

## Checkpoints quantizados

Um modelo quantizado adiciona uma chave plana opcional, `quant`, que guarda um
dict de manifesto com `schema`, `recipe`, `keep_high_precision`, `execution`,
proveniência de calibração, `module_count` e `state`. Manifestos FP8 também
podem carregar `fp8_tensorwise_weights`, a lista exata de nomes de módulos
`QuantLinear` cuja escala de peso é por tensor em vez de por canal de saída. Um
loader que veja `quant` reconstrói a estrutura dos módulos quantizados e a
política de escalonamento antes do `load_state_dict`.

`state` distingue as duas formas de artefato.

`"prepared"`, o padrão, guarda pesos mestres FP32 mais buffers de escala `_q_*`
e é treinável. Um leitor sem suporte a quantização pode ignorar a chave `quant`
e carregar os mestres como um modelo float.

`"finalized"` é a forma de deploy escrita por `export(format="pt")`. Os mestres
são removidos e cada módulo quantizado carrega, em vez deles, pesos empacotados:

| Receita | Tensores empacotados | Dequantização |
|---|---|---|
| int8 | `weight_packed` int8 no shape original do peso, `_q_w_scale` FP32 por canal | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn no shape original, `_q_w_scale` FP32 com uma entrada por canal de saída | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, dois códigos de 4 bits por byte, nibble baixo primeiro, código `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, grupo 128 ao longo de in_features | Escala por grupo |
| int2 | Quatro códigos de 2 bits por byte, código `q + 2`, grupo 64 | Escala por grupo |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, código `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 por tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Como nvfp4, mas com blocos de 32 elementos, mais `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Os buffers de faixa de ativação `_q_act_lo`, `_q_act_hi` e `_q_calibrated` são
mantidos para int8. O manifesto registra `remainder`, `"fp16"` ou `"fp32"`, para
os tensores não quantizados. O desempacotamento reproduz a simulação bit a bit,
então a inferência finalizada bate exatamente com a inferência preparada no
dispositivo em que a finalização ocorreu. Esse layout é o contrato estável para
exportadores e runtimes externos.

## Checkpoints de treinamento

Os checkpoints do trainer usam o mesmo núcleo de metadados obrigatórios e podem
adicionar campos planos de treinamento e de retomada:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` declara se o `model` de primeiro nível é suavizado por EMA.
Quando a EMA está habilitada, `train_model`, `ema` e `ema_updates` preservam o
estado de retomada. Pesos publicados para inferência devem ser enxutos e não
devem incluir optimizer, epoch, config, loss nem estado de retomada de EMA, a
menos que sejam distribuídos intencionalmente como checkpoints de treinamento.

Por compatibilidade entre versões, os leitores aceitam os aliases legados de
melhor métrica `best_mAP50_95`, `best_mAP50`, `best_metric` e
`best_metric_name`.

## Snapshots externos

O esquema rege os arquivos `.pt` escritos pelo LibreYOLO. Ele não renomeia nem
envolve snapshots upstream de múltiplos arquivos usados pelos tiers de modelo
separados.

O LibreMODUS de tamanho `14b-a7b` é uma exceção explícita: o alias resolve, via
`LibreVLM(...)`, para um diretório de arquivos upstream fixados, e o LibreYOLO
não adiciona metadados v1.0 a ele nem o republica como um `.pt`.

## Pesos legados e de terceiros

Novos escritores validam de forma estrita e precisam emitir metadados v1.0.
Quando os metadados estão ausentes ou incompletos, checkpoints legados com cara
de LibreYOLO carregam pelo caminho de compatibilidade, com um aviso e instruções
de conversão, e checkpoints upstream de terceiros são roteados para a conversão
automática. Veja
[checkpoints upstream](/docs/reference/upstream-checkpoints).

## Helpers

Os helpers do esquema ficam em `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` não altera nada e retorna a lista de erros; com
`strict=True` ele levanta `CheckpointMetadataError` em vez disso.
`model.save(path)` é a forma suportada de escrever um checkpoint conforme.
