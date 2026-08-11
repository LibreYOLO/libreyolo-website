---
title: Normais de superfície
seo_title: Estimativa de normais de superfície no LibreYOLO
description: >-
  Preveja um campo denso de normais de superfície a partir de uma única imagem
  no LibreYOLO. Conheça a convenção de referencial de câmera, valide o erro
  angular e exporte um modelo.
lead: >-
  A estimativa de normais de superfície prevê a direção para a qual cada
  superfície visível aponta. O LibreYOLO expõe isso como a tarefa normal, que
  devolve um campo denso de vetores unitários sobre o canvas da imagem original.
keywords:
  - estimativa de normais de superfície python
  - gerar mapa de normais de uma imagem
  - geometria monocular
  - métrica de erro angular
  - normal map python
last_verified: 1.5.0
snippets:
  predict:
    - label: Prever um campo de normais
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # vetores unitários (H, W, 3) float32

        normals.assert_normalized()    # levanta erro se algum pixel não for
        unitário
    - label: Ler um pixel
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # Referencial de câmera do OpenCV: +x à direita, +y para baixo, +z para

        # dentro da cena. Uma superfície voltada para a câmera fica perto de (0,
        0, -1).

        field = result.normals.data

        h, w = field.shape[:2]

        print(field[h // 2, w // 2])
    - label: Salvar a visualização
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # plot() renderiza o campo; está definido para resultados de normais e
        bordas.

        result.plot().save("normals.png")
  val:
    - label: Validar e ler as chaves das métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # graus
        print(metrics["metrics/median_angular_error"])   # graus
        print(metrics["metrics/within_11_25"])           # porcentagem de pixels
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Rodar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definição

A tarefa `normal` prevê um vetor unitário de três componentes por pixel a partir
de uma única imagem RGB: a direção para a qual a superfície naquele pixel aponta.
Diferente da profundidade, a saída não tem escala livre, então duas predições são
diretamente comparáveis sem alinhamento.

Uma predição preenche `result.normal_map`, um payload `NormalMap` que guarda um
array `(H, W, 3)` float32 sobre o canvas da imagem original, também acessível
como `result.normals`. Os vetores usam o referencial de câmera do OpenCV do
LibreYOLO, com `+x` à direita, `+y` para baixo e `+z` para dentro da cena, e
apontam para a câmera, então uma superfície fronto-paralela é `(0, 0, -1)`.
`.assert_normalized()` verifica que todo pixel é finito e unitário dentro de uma
tolerância. `result.boxes` fica vazio, então `conf`, `iou` e `max_det` não têm
efeito, e `Results.plot()` cobre esta tarefa.

## Modelos

Duas famílias atendem `normal`.

[MoGe-2](/docs/models/moge-2) é a dedicada: um modelo de geometria monocular de
uma única passagem em três tamanhos de encoder. O LibreYOLO não copia esses
checkpoints para a sua própria organização; carregar um baixa o tamanho
correspondente dos repositórios oficiais em uma revisão fixada e o confere contra
um SHA-256 registrado.

[LibreMODUS](/docs/models/libremodus) produz normais como um dos alvos de um
modelo any-to-any, e pode receber um mapa de profundidade em vez de uma imagem
RGB como entrada. Precisa do extra `modus` e da sua própria conta autenticada no
Hugging Face, e não oferece nem `val()` nem `export()`, então não participa das
seções de validação e exportação abaixo.

## Predição

Os pesos do MoGe-2 são baixados no primeiro uso e ficam em cache localmente.

<code-tabs name="predict" />

`imgsz` precisa ser divisível pelo tamanho de patch do encoder ViT, o que o
LibreYOLO confere antes de a execução começar. Predizer uma lista de imagens roda
um forward pass por imagem; esta tarefa não tem caminho rápido de batch
empilhado. Veja [predição](/docs/predict) para fontes, streaming e tratamento de
resultados.

## Formato do dataset

A validação de normais emparelha cada imagem com um PNG de 16 bits e três canais,
de mesmo nome-base e mesma resolução, mais uma máscara de validade opcional.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

O PNG alvo é exatamente `uint16` de três canais, com os canais armazenados como
RGB. A decodificação é `n = png / 65535 * 2 - 1` seguida da renormalização de
cada vetor, e os vetores decodificados usam o mesmo referencial de câmera do
OpenCV das predições. Um pixel de máscara conta como válido quando é diferente de
zero; sem arquivo de máscara, todo vetor decodificado finito e diferente de zero
é válido. Pixels de alvo inválidos e de padding são mantidos internamente como
`(0, 0, 0)` e nunca contribuem para uma métrica. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

Nenhuma das duas famílias de normais tem implementação de treinamento: `train()`
levanta `NotImplementedError` nas duas. A página do MoGe-2 indica seus
checkpoints oficiais fixados para predição, validação e exportação.

## Validação

`val()` mede o ângulo entre cada vetor predito e o seu vetor de ground truth,
sobre os pixels que o dataset marca como válidos.

<code-tabs name="val" />

`metrics/mean_angular_error` e `metrics/median_angular_error` são esse ângulo em
graus, e quanto menor, melhor. `metrics/within_11_25`, `metrics/within_22_5` e
`metrics/within_30` são a porcentagem de pixels válidos cujo erro angular fica
dentro de 11.25, 22.5 e 30 graus, então quanto maior, melhor. Repare na unidade:
esses três são porcentagens, não frações. `fitness` é `metrics/within_11_25`
dividido por 100, o que coloca a seleção do melhor checkpoint na mesma escala
`[0, 1]` de todas as outras tarefas.

## Exportação

Um modelo de normais exportado é carregado de volta por `LibreYOLO()` pelo sufixo
do arquivo, então um arquivo `.onnx` se comporta como um checkpoint e devolve o
mesmo `Results`.

<code-tabs name="export" />

A exportação de normais usa um contrato de runtime de resolução fixa e batch 1:
`dynamic` e um `batch` diferente de 1 são rejeitados, e `imgsz` precisa ser
divisível pelo tamanho de patch do encoder. A cobertura por formato está na
[página do MoGe-2](/docs/models/moge-2) e na
[matriz completa de exportação](/docs/reference/export-matrix).
[Exportação](/docs/export) lista os argumentos que cada formato aceita.
