---
title: DEIM
families: [deim]
seo_title: "DEIM e DEIMv2 no LibreYOLO"
description: "Use DEIM e DEIMv2 no LibreYOLO para detecção de objetos. Instale, faça predições, treine, valide e exporte, a partir de um tamanho de meio milhão de parâmetros."
lead: "Um transformer de detecção treinado com correspondência densa um-para-um, que converge em muito menos épocas que as receitas DETR sobre as quais é construído. O LibreYOLO inclui duas versões dele, distinguidas pelo checkpoint que você carrega."
keywords: [DEIM, DEIMv2, DINOv3, "transformer de detecção", DETR, "detecção de objetos", "detecção de objetos em tempo real", "detecção de objetos python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A versão faz parte do nome do arquivo, e a fábrica roteia com base
        # no checkpoint, então as duas carregam do mesmo jeito.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice
        # de webcam, stream RTSP ou uma lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml baixa uma amostra de 128 imagens no primeiro uso.
        # Aponte `data` para o YAML do seu próprio dataset em uma execução real.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Se não forem definidos, epochs, batch, imgsz e lr0 vêm da receita
        # publicada para o tamanho que foi carregado.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Precisa do extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() retorna um dict simples, não um objeto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Contra o COCO
      language: bash
      code: |
        # coco-val-only.yaml busca as 5000 imagens de val2017 e pula o
        # conjunto de treinamento. Ele traz um script de download embutido,
        # então precisa de permissão explícita a menos que o dataset já esteja
        # local.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Precisa do extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A fábrica roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalação

Nenhuma das duas versões precisa de um extra opcional. Tudo o que elas importam
está na instalação base.

```bash
pip install libreyolo
```

O fine-tuning por adaptadores com `lora=True` é a exceção, e precisa do extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que toda família retorna, então trocar
por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram uma
decodificação top-k sobre queries e classes; não há um passo de NMS para
ajustar, e `iou` é aceito mas não usado. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Variantes

A versão 1 traz cinco tamanhos, todos no mesmo tamanho de entrada. A versão 2
mantém esses cinco nomes e acrescenta três menores, `atto`, `femto` e `pico`, os
dois primeiros nativos em um tamanho de entrada menor que o resto. Cinco códigos
de tamanho existem, portanto, nas duas versões e nomeiam modelos diferentes; a
versão está escrita no nome do arquivo do checkpoint.

<benchmark-table task="detect" />

<va-embed />

A versão 1 mantém a arquitetura do D-FINE e troca seu objetivo de classificação
pela loss ciente de matchability da receita densa um-para-um, então as duas
famílias compartilham quase todas as chaves do state dict e são distinguidas
pelos metadados do checkpoint. A versão 2 mantém esse contrato de treinamento e
mistura backbones: HGNetv2 abaixo de `s`, e um vision transformer DINOv3 com um
adaptador de tuning espacial em `s` e acima. Esse backbone é o que coloca uma
segunda licença nesses quatro checkpoints, então leia
[licenciamento](#licensing) antes de colocar um deles em produção.

## Treinamento

O treinamento parte de um checkpoint publicado. `pretrained` nunca chega ao
treinador: a versão 1 avisa que a chave é desconhecida e a ignora, a versão 2 a
remove. Nenhuma das duas entrega um modelo inicializado aleatoriamente.

<code-tabs name="train" />

Passe `lr0` você mesmo na versão 1. A assinatura Python do `train()` dela usa
`4e-4` como padrão, a taxa da receita COCO publicada, enquanto a configuração de
treinamento da família carrega `1e-4` como seu padrão de fine-tune, e é esse
valor menor que a CLI resolve quando o argumento está ausente. A configuração
registra a medição por trás disso: nos tamanhos de batch que um fine-tune de
fato usa, em datasets pequenos, a taxa do COCO degradou a transferência de forma
mensurável.

A versão 2 resolve esses padrões sozinha. Deixar `epochs`, `batch`, `imgsz` e
`lr0` sem definir faz com que ela leia cada um da receita publicada para o
tamanho que foi carregado, então os tamanhos pequenos treinam na própria
resolução de entrada sem que ninguém precise dizer, e um valor que você passa
sobrescreve a receita. `imgsz` é o argumento que ela restringe: precisa ser um
múltiplo positivo de 32, e a versão 2 lança um erro antes de a execução começar
caso contrário.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

As linhas da tabela de benchmark acima vêm do harness de benchmark do
LibreYOLO; a nota abaixo dessa tabela registra qual dataset as produziu e liga
os registros das execuções.

## Exportação

<export-matrix />

A matriz cobre as duas versões em uma única página: onde elas discordam sobre um
formato, a célula mostra a mais fraca das duas, então nada aqui fica
supervalorizado para a versão que você carregar.

Um artefato exportado volta a carregar por `LibreYOLO()` pelo sufixo do arquivo,
então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e retorna o
mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todo arquivo de pesos publicado desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>
Os quatro tamanhos do DEIMv2 de S para cima pegam seu backbone do DINOv3, então
seus repositórios de pesos carregam tanto a Apache-2.0 quanto a DINOv3 License
da Meta, e o LibreYOLO distribui o código do backbone DINOv3 sob esse mesmo
acordo. O resto desta família, incluindo todo tamanho do DEIMv2 abaixo de S, é
Apache-2.0 apenas.
</provenance-box>

## Citação

<citation-block />

O DEIMv2 é um artigo separado e tem seu próprio bloco de citação em
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
cite esse se você usou um checkpoint da versão 2.
