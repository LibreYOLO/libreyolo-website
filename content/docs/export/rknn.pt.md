---
title: RKNN
seo_title: "Exportar para RKNN em NPUs Rockchip"
description: "Compile um detector LibreYOLO para um artefato .rknn da Rockchip: o SDK do fabricante que você mesmo instala, as quatro variantes RK3588 validadas e a paridade no simulador."
lead: "RKNN é o formato compilado de NPU da Rockchip. O LibreYOLO exporta um ONNX intermediário de opset 19, compila com o SDK RKNN Toolkit2 e consegue comparar o grafo compilado com o ONNX Runtime no simulador de host do Toolkit2, sem placa nenhuma."
keywords:
  - exportar yolo rknn
  - npu rockchip
  - rk3588
  - rknn-toolkit2
  - paridade simulador rknn
  - inferência rockchip orange pi
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Escreve
    value: "Um arquivo .rknn, um sidecar .rknn.metadata.json e um relatório .rknn.parity.json quando verify=True"
  - label: Extra
    value: "Nenhum no PyPI. O rknn-toolkit2 é um SDK do fabricante que você mesmo instala."
  - label: Recarrega com
    value: "Não pelo LibreYOLO. O artefato roda na placa com o runtime da Rockchip."
  - label: Formas
    value: "Quadrada fixa, batch 1, opset 19. Os três são exigidos."
  - label: Precisão
    value: "O build de ponto flutuante do fabricante. half=True e int8=True são rejeitados."
  - label: Escopo
    value: "Quatro variantes de detecção no RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s e YOLO-NAS-s"
verification: "Lido de libreyolo/export/rknn.py, libreyolo/export/exporter.py, libreyolo/export/support.py e docs/rknn.md no branch dev. Os números de paridade medidos vêm do registro de validação de 2026-08-04 em docs/rknn.md."
snippets:
  install:
    - label: Lado do LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: SDK do fabricante, instalado por você
      language: bash
      code: |
        # O rknn-toolkit2 é um SDK da Rockchip sob licença separada. O LibreYOLO
        # não o embute nem o instala. Somente Linux x86_64; no Windows, use
        # WSL2 ou um container Linux.
        #
        # O Toolkit2 2.3.2 precisa de setuptools<81 e falha no ONNX 1.19 ou mais
        # novo, cuja remoção de onnx.mapping o compilador dele ainda importa.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Depois instale o wheel correspondente do rknn-toolkit2 a partir do
        # repositório de wheels da própria Rockchip e confirme que ele importa:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Escreve weights/LibreYOLO9t.rknn e weights/LibreYOLO9t.rknn.metadata.json
        path = model.export(format="rknn", name="rk3588", imgsz=640, verify=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # plataforma alvo; target= e target_platform= também funcionam
            imgsz=640,         # precisa bater com o canvas registrado da variante
            batch=1,           # qualquer outro valor levanta NotImplementedError
            dynamic=False,     # True levanta ValueError
            opset=19,          # qualquer outro valor levanta NotImplementedError
            verify=False,      # True roda o simulador de PC e barra pela paridade
        )
  parity:
    - label: Paridade sem placa contra um artefato ONNX existente
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Conferir uma família e uma tarefa antes de compilar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Instalação

A compilação precisa do RKNN Toolkit2 da Rockchip, que é distribuído como um SDK
do fabricante sob a licença própria da Rockchip e não é uma dependência do
LibreYOLO. Não existe um extra `libreyolo[rknn]`, e nada nesse formato se instala
com uma única linha.

<code-tabs name="install" />

Não é preciso ter uma placa para compilar nem para conferir a paridade numérica.
Uma placa RK3588 é necessária para medir latência, consumo e comportamento
térmico, e nenhuma dessas medidas foi registrada.

## Exportação

<code-tabs name="export" />

A requisição é validada contra uma lista de variantes de modelo exatas antes de
qualquer compilação, e o canvas também é validado: passar um `imgsz` diferente
daquele com que a variante foi registrada levanta um erro em vez de compilar em
silêncio algo não testado. O LibreYOLO escreve um ONNX intermediário de opset 19,
compila esse ONNX, opcionalmente o simula e depois remove o intermediário.

Os metadados ficam em um sidecar chamado `<model>.rknn.metadata.json`, porque o
formato RKNN não tem um campo de metadados portável.

`verify=True` roda o simulador de PC do Toolkit2 dentro da mesma sessão que
compilou o artefato, compara cada saída com o ONNX Runtime na mesma entrada e
escreve `<model>.rknn.parity.json` com métricas de erro por saída. Os limiares são
similaridade de cosseno de no mínimo 0.9999 e RMSE normalizado de no máximo 0.02,
aplicados a qualquer saída que já não esteja próxima elemento a elemento; o build
de ponto flutuante do fabricante rebaixa os tensores internos para meia precisão,
então um `allclose` estrito não vale nem mesmo quando os bounding boxes
decodificados estão estáveis. Uma execução que falha escreve
`<model>.rknn.failed.parity.json`, descarta o candidato e deixa intacta qualquer
exportação anterior bem-sucedida naquele caminho.

Para comparar um artefato ONNX que você já tem, sem exportar de novo:

<code-tabs name="parity" />

O simulador do Toolkit2 roda o grafo em memória produzido por `load_onnx` e
`build`. Ele não consegue recarregar um arquivo `.rknn` específico de um target
sem uma placa, e é por isso que `verify=True` faz compilação, exportação e
simulação em uma única sessão.

## Executar o artefato

Não há nenhuma entrada de RKNN em `libreyolo/backends`, então `LibreYOLO()` não
carrega um arquivo `.rknn`. O artefato compilado recebe deploy na placa e é
executado pelo runtime da própria Rockchip, e ali o pré-processamento, a
decodificação, o NMS e o reescalonamento de coordenadas são responsabilidade da
aplicação.

`<model>.rknn.metadata.json` carrega os nomes das classes, o tamanho de entrada, a
tarefa e a plataforma alvo, que é o que uma aplicação precisa para reproduzir o
pós-processamento do LibreYOLO. Distribua esse arquivo junto com o modelo
compilado.

Para uma verificação no host que não precise da placa, guarde um artefato ONNX na
mesma forma fixa e compare no simulador, como acima.

## Restrições

Quatro combinações compilam, e elas são variantes de modelo em vez de famílias:

| Variante | Tarefa | Canvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Todo o resto é recusado antes da compilação, com a mensagem de que o RKNN nesta
versão se limita às variantes de detecção exatas testadas no simulador. Existem
resultados de somente compilação para outros modelos, mas eles deliberadamente
não são apresentados como suporte: na mesma rodada de medição, o RF-DETR deixou
dois nós `GridSample` do decoder sem rebaixar, e D-FINE, RT-DETR, RT-DETRv2,
RT-DETRv4, DEIM, DEIMv2 e EC compilaram e simularam com saídas decodificadas
materialmente erradas.

Batch 1, formas estáticas, opset 19. `half=True` é rejeitado, porque o RKNN não
expõe o contrato `half` do LibreYOLO, e `int8=True` é rejeitado até que existam
uma calibração representativa e resultados de acurácia por tarefa.

Os demais targets da Rockchip são rejeitados: `rk3588` é a única plataforma
validada.

Para a grade completa de famílias e tarefas, veja
[a matriz de exportação](/docs/reference/export-matrix). Para uma combinação:

<code-tabs name="support" />
