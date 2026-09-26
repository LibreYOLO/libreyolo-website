---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: super-resolução de imagens em 4x no LibreYOLO'
description: >-
  Use o SwinIR no LibreYOLO para super-resolução de imagens em 4x. Instale, faça
  predições, valide e exporte os checkpoints lightweight, médio e grande.
lead: >-
  Uma rede Swin Transformer para restauração de imagens. O LibreYOLO inclui
  inferência e validação para seus checkpoints de super-resolução em 4x: o
  gerador lightweight oficial e os geradores médio e grande para mundo real.
keywords:
  - SwinIR
  - Swin Transformer
  - super-resolução de imagens
  - restauração de imagens
  - aumentar resolução de imagem python
  - melhorar qualidade de foto
  - ampliar imagem sem perder qualidade
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'Por blocos (tiles), para imagens grandes'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile divide o forward pass em tiles sobrepostos e funde as
        # emendas de volta; tile_pad é o halo adicionado ao redor de cada
        # tile antes de ele ser recortado de volta. Ambos são argumentos
        # nomeados só do Python, não flags da CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # quando omitido, imgsz assume um tamanho de patch interno pequeno,
        # não a sua resolução de trabalho, então passe o tamanho que o seu
        # deploy realmente entrega ao modelo.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Instalação

O SwinIR não precisa de nenhum extra opcional. Tudo o que ele importa está na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

Um resultado de restauração não carrega boxes; `result.restored` é uma imagem
RGB densa `(H, W, 3)` uint8, em um canvas 4x o da entrada em cada dimensão.
`save=True` grava essa imagem direto, em vez de um plot anotado. A entrada
recebe padding até um múltiplo de 8 em vez de ser redimensionada, então a
predição roda na resolução original da foto; uma fonte maior do que a memória
permite pode ser dividida com `tile` e `tile_pad`, que fundem as emendas dos
tiles de volta na saída. Veja [predição](/docs/predict) para fontes, streaming
e tratamento de resultados.

## Variantes

Três tamanhos, todos fixos em um aumento de escala de 4x. `s` é o gerador
lightweight oficial, com quatro estágios de residual Swin Transformer block
(RSTB) e upsampling pixel-shuffle direto. `m` e `l` são os geradores médio e
grande para mundo real, com seis e nove estágios RSTB e um upsampler de
vizinho mais próximo mais convolução, feito para degradações do mundo real e
não apenas para redução bicúbica.

## Validação

`val()` mede PSNR e SSIM entre a saída restaurada e uma imagem alvo limpa,
ambas calculadas em RGB no canvas original, sem recorte de borda e sem
redimensionamento. O SSIM usa uma janela gaussiana de 11x11 com sigma 1.5, com
média sobre os três canais de cor.

<code-tabs name="val" />

O argumento de dataset é um YAML que pareia um diretório de imagens de entrada
degradadas com um diretório de imagens alvo limpas de resolução equivalente;
veja [formatos de dataset](/docs/reference/dataset-formats) para as chaves
exatas.

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. O ExecuTorch e todo formato que a matriz marca como
bloqueado não estão disponíveis para esta família; ONNX, TorchScript, TensorRT,
OpenVINO e TFLite estão. [Exportação](/docs/export) lista os argumentos que todo
formato aceita e os extras que alguns poucos adicionam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
