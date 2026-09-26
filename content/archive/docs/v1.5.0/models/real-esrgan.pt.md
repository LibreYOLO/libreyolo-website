---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: super-resolução de imagens no LibreYOLO'
description: >-
  Use o Real-ESRGAN no LibreYOLO para super-resolução de imagens prática em 4x,
  2x e um nível 4x rápido. Instale, faça predições, valide e exporte.
lead: >-
  Um upscaler de super-resolução cega voltado para uso prático, treinado com
  degradações sintéticas em vez de apenas redução bicúbica. O LibreYOLO inclui
  inferência e validação para seus checkpoints 4x, 2x e 4x rápido.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - super-resolução de imagens
  - melhorar qualidade de imagem python
  - aumentar resolução de imagem
  - ampliar imagem sem perder qualidade
  - restauração de imagens
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'Por blocos (tiles), para imagens grandes'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile divide o forward pass em tiles sobrepostos e funde as
        # emendas de volta; tile_pad é o halo adicionado ao redor de cada
        # tile antes de ele ser recortado de volta. Ambos são argumentos
        # nomeados só do Python, não flags da CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # quando omitido, imgsz assume um tamanho de patch interno pequeno,
        # não a sua resolução de trabalho, então passe o tamanho que o seu
        # deploy realmente entrega ao modelo.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Instalação

O Real-ESRGAN não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

Um resultado de restauração não carrega boxes; `result.restored` é uma imagem
RGB densa `(H, W, 3)` uint8, em um canvas `Results.restore_scale` vezes o da
entrada em cada dimensão. `save=True` grava essa imagem direto, em vez de um
plot anotado. A entrada é convertida para RGB e qualquer canal alfa é
descartado. Uma fonte maior do que a memória permite pode ser dividida com
`tile` e `tile_pad`, que fundem as emendas dos tiles de volta na saída. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Três checkpoints, nomeados pelo seu fator de aumento de escala. `x4` é RRDBNet
(`RealESRGAN_x4plus`), 23 blocos densos residual-in-residual, o padrão de
qualidade em 4x. `x2` é a mesma arquitetura RRDBNet em 2x. `x4t` é
SRVGGNetCompact (`realesr-general-x4v3`), um gerador menor e mais rápido,
feito para vídeo e usos de menor latência em 4x. O modelo de propósito geral
do projeto original também inclui uma rede pareada de força de remoção de
ruído, misturada no momento da inferência; esse controle de força não faz
parte deste port, que roda o gerador `x4t` base.

## Validação

`val()` mede PSNR e SSIM entre a saída restaurada e uma imagem alvo limpa,
ambas calculadas em RGB no canvas original, sem recorte de borda e sem
redimensionamento. O SSIM usa uma janela gaussiana de 11x11 com sigma 1.5,
com média sobre os três canais de cor.

<code-tabs name="val" />

O argumento de dataset é um YAML que pareia um diretório de imagens de entrada
degradadas com um diretório de imagens alvo limpas de resolução equivalente;
veja [formatos de dataset](/docs/reference/dataset-formats) para as chaves
exatas.

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint
e devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita e os extras que alguns poucos adicionam.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
