---
title: D-FINE
families: [dfine]
seo_title: "D-FINE: faça fine-tuning, valide e exporte sob a MIT"
description: "Use o D-FINE no LibreYOLO para detecção de objetos e segmentação de instâncias. Instale, rode predições, faça fine-tuning, valide e exporte, com código licenciado sob a MIT."
lead: "Um transformer de detecção que reformula a regressão de caixas como uma distribuição de probabilidade sobre cada borda da caixa, refinada ao longo das camadas do decoder. O LibreYOLO o suporta para detecção e segmentação de instâncias."
keywords: [D-FINE, "transformer de detecção", "detecção de objetos em tempo real", "segmentação de instâncias", "fine-tuning D-FINE", DETR]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O sufixo -seg no nome do arquivo seleciona a cabeça de máscaras, então
        # nenhum argumento task é necessário aqui.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentação de instâncias
      language: bash
      code: |
        # Continua a partir de pesos de segmentação publicados, cabeça de máscaras incluída.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentação a partir de pesos de detecção
      language: bash
      code: |
        # Os pesos de detecção não trazem cabeça de máscaras, então isto é uma
        # transferência explícita: a cabeça começa sem treinamento e só serve depois
        # de treinada. Pedir task=segment aqui é o que autoriza a transferência.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentação de instâncias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # caixas
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640
        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640 half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalação

O D-FINE não precisa de nenhum extra opcional. Tudo que ele importa já está na
instalação base.

```bash
pip install libreyolo
```

O fine-tuning com adaptadores via `lora=True` é a exceção, e precisa do extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Um nome de arquivo com
`-seg` já resolve sozinho para a tarefa de segmentação, e aí `result.masks`
carrega as máscaras de instância junto com as caixas. `conf` e `max_det` filtram
a seleção de queries; `iou` é aceito por paridade de API mas não tem efeito,
porque o decoder é um preditor de conjuntos sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Cinco tamanhos. Todos rodam na mesma resolução de entrada, então a tabela os
separa por número de parâmetros e acurácia.

<benchmark-table task="detect" />

<va-embed />

A segmentação reaproveita o backbone, o encoder e o decoder da detecção e
acrescenta uma cabeça de máscaras, então um checkpoint `-seg` aceita os mesmos
argumentos que o irmão de detecção. A família RT-DETRv4 do LibreYOLO é escrita
como uma subclasse do wrapper do D-FINE: ela herda essa linhagem de decoder e
depois fixa sua lista de tarefas de volta em detecção, porque não traz cabeça de
máscaras.

## Treinamento

O treinamento parte de um checkpoint publicado, para as duas tarefas.

<code-tabs name="train" />

Sem mexer em nada, o trainer roda 132 épocas com `lr0=2e-4` e `amp=False`, um
batch de 16 e early stopping após 50 épocas sem melhora. Os pesos de detecção são
um ponto de partida válido para treinar segmentação, mas só como transferência
explícita, já que a cabeça de máscaras começa sem treinamento e de outro modo
devolveria máscaras sem sentido. Passar `task=segment` para o CLI é o que
autoriza isso. O caminho pelo Python é mais estreito: `LibreDFINE` precisa ser
construído diretamente com `allow_detect_to_segment_transfer=True`, porque a
factory `LibreYOLO()` não aceita esse argumento, e a construção direta não baixa
nada, então o arquivo de pesos já precisa estar em disco.

`lora=True` vale para detecção. O treinamento de segmentação o rejeita e aponta
para `freeze='backbone'` no lugar, porque a cabeça de máscaras não foi testada
com adaptadores. No Apple silicon o trainer move a execução inteira para a CPU: o
backward pass do matmul por bins do Integral esbarra em uma falha de compilação
do Metal. A inferência em MPS não é afetada.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário indexado pelo nome da métrica, e imprime os
resultados por classe quando `verbose` fica ligado.

<code-tabs name="val" />

Contra um checkpoint `-seg`, a chave `metrics/mAP50-95` pura contém a pontuação
das máscaras, e a mesma execução também reporta as caixas em `(B)` e as máscaras
em `(M)`, então as duas ficam disponíveis em uma única passada.

## Exportação

<export-matrix />

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Os caminhos de OpenVINO, Paddle, MNN e Core AI
exportam com um canvas fixo em vez de formas dinâmicas.
[Exportação](/docs/export) lista os argumentos que todo formato aceita e os
extras que alguns deles acrescentam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

Os pesos de segmentação têm um segundo upstream: o decoder de máscaras, o
matching de máscaras e a loss de máscaras vêm do ArgoHA/D-FINE-seg, também
Apache-2.0, cujo maintainer aprovou o reuso com atribuição.

</provenance-box>

## Citação

<citation-block />
