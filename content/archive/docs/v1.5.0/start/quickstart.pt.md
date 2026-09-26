---
title: Início rápido
seo_title: Início rápido do LibreYOLO
description: >-
  Rode um detector em uma imagem, faça fine-tuning com um dataset pequeno e
  exporte para TorchScript ou ONNX, tudo na CPU, em cerca de dez linhas de
  Python.
lead: >-
  O caminho mais curto pelo LibreYOLO: prediga em uma imagem, treine com um
  dataset pequeno e depois exporte o resultado. Todos os comandos desta página
  rodam na CPU.
keywords:
  - libreyolo início rápido
  - tutorial libreyolo
  - libreyolo predict
  - treinar libreyolo
  - exportar libreyolo onnx
  - exemplo yolo python
  - detecção de objetos python
last_verified: 1.5.0
meta:
  - label: Instalação
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: A CPU basta para tudo nesta página
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Baixa o checkpoint no primeiro uso e depois o guarda em cache em
        weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Uma única imagem retorna um objeto Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo e streams
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True gera um Results por quadro em vez de montar uma lista.
        # Troque o caminho por um índice de webcam, uma URL RTSP ou uma pasta.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 é um dataset de 8 imagens que vem com a biblioteca. Ele é
        baixado

        # de uma URL no primeiro uso, então nenhum script precisa ser executado.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() retorna um dict simples, não um objeto.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() retorna o caminho que escreveu.
        path = model.export(format="torchscript")
        print(path)

        # A fábrica roteia pelo sufixo do arquivo, então o artefato é carregado
        # de volta como um checkpoint e retorna o mesmo objeto Results.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Instalação

```bash
pip install libreyolo
```

Isso é tudo o que as seções de predição e treinamento abaixo precisam. Exportar
para ONNX adiciona um extra; veja [instalação](/docs/install) para a lista
completa.

## Predição

<code-tabs name="predict" />

`LibreYOLO()` é uma fábrica. Ela lê o arquivo, descobre a qual família os pesos
pertencem e retorna o modelo dessa família, então trocar por outro detector é
uma mudança de uma linha. Passar `LibreYOLO9t.pt` sem diretório faz com que
`weights/LibreYOLO9t.pt` seja procurado em relação ao diretório de trabalho e
baixado ali quando estiver faltando. Veja [checkpoints e pesos](/docs/weights)
para as regras de download e como trabalhar offline.

`save=True` escreve uma cópia anotada em `runs/detect/`, dentro de um diretório
`predict` que é incrementado a cada execução. O `Results` retornado carrega
`boxes`, e `names` mapeia um índice de classe para o seu rótulo. O caminho de
uma única imagem retorna um `Results`; um diretório, uma lista de imagens ou
`stream=True` retornam uma lista ou um gerador deles.

## Treinamento

<code-tabs name="train" />

`data` é um YAML de dataset. `coco8.yaml` vem com a biblioteca, e é por isso que
o snippet roda como está colado; um nome que não venha incluído é lido como um
caminho. Os datasets são resolvidos em `~/datasets`, ou em
`LIBREYOLO_DATASETS_DIR` quando essa variável está definida.

Uma execução escreve em `project/name`, por padrão um diretório abaixo de
`runs/train`, com `weights/best.pt` e `weights/last.pt` dentro dele. `train()`
retorna um dicionário que inclui `save_dir`, `best_checkpoint`,
`last_checkpoint`, as losses por época e as métricas de validação por época. O
checkpoint treinado é carregado através de `LibreYOLO()` exatamente como o
pré-treinado.

Nem toda família é treinável. Quando uma família traz apenas inferência,
`train()` lança `NotImplementedError` e diz isso. [Conceitos
básicos](/docs/concepts) explica o que significa cada nível de suporte.

## Exportação

<code-tabs name="export" />

O TorchScript não precisa de nada além da instalação base. Os outros destinos
têm cada um o seu próprio extra, e a cobertura é por família e por tarefa, não
uniforme: veja [exportação e deploy](/docs/export).

Os argumentos aceitos por todos os formatos incluem `imgsz` (um int, ou um par
de altura e largura), `batch` (padrão 1), `half`, `int8` com um YAML de `data`
para calibração, `dynamic` (padrão True), `simplify` (padrão True), `opset`,
`device` e `output_path`. Quando `output_path` é omitido, o arquivo é escrito em
`weights/` com um nome derivado do checkpoint.

## Próximos passos

- [Conceitos básicos](/docs/concepts) para tarefas, famílias, tamanhos e nomes de checkpoints.
- [Checkpoints e pesos](/docs/weights) para o download automático, o uso offline e a segurança ao carregar.
- [Importar pesos existentes](/docs/migrate) se você já tem um checkpoint de um projeto upstream.
- [Todos os modelos](/docs/models) para a família que se encaixa no seu problema.
- [Treinamento](/docs/train), [Predição](/docs/predict) e [Exportação](/docs/export) para os fluxos de trabalho completos.
