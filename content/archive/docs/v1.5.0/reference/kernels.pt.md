---
title: Kernels
seo_title: Registro de kernels e kernels do Hub no LibreYOLO
description: >-
  Como o LibreYOLO seleciona implementações aceleradas: o registro de kernels em
  libreyolo/kernels, o kernel opcional de MS-deform-attn no Hugging Face Hub e a
  chave de ativação da atenção fundida.
lead: >-
  Toda operação acelerada no LibreYOLO tem um padrão portátil e, às vezes, uma
  variante mais rápida registrada por cima dele. A seleção acontece em tempo de
  execução por predicado, uma dependência opcional ausente vira fallback em vez
  de erro, e um grafo exportado sempre segue o caminho portátil.
keywords:
  - kernels libreyolo
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - extra hub-kernels
  - kernel ms_deform_attn
  - set_fused_attention
  - acelerar inferência libreyolo cuda
last_verified: 1.5.0
verification: >-
  API do registro lida de libreyolo/kernels/__init__.py na v1.5.0, API de
  atenção de libreyolo/kernels/attention/__init__.py e sdpa.py, provider do Hub
  de libreyolo/kernels/attention/ms_deform_attn.py, incluindo sua revisão fixada
  e o predicado de elegibilidade. Estrutura de diretórios listada a partir de
  libreyolo/kernels/. Definição do extra vinda de pyproject.toml. Notas de
  comportamento e números de benchmark de docs/kernels.md. O histórico de gating
  da v1.4.0 vem do commit que ligou o slot no RF-DETR e da entrada do CHANGELOG
  1.5.0.
meta:
  - label: Pacote
    value: libreyolo.kernels
    mono: true
  - label: Extra opcional
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Forçar referência
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Ver o que foi selecionado
      language: python
      code: |
        import libreyolo.kernels as kernels

        # Slot de op para o nome da implementação selecionada, ou "unavailable".
        print(kernels.active())
    - label: Forçar o caminho de referência
      language: bash
      code: |
        # off e reference significam a mesma coisa, e também pulam
        # completamente a importação dos providers acelerados.
        LIBREYOLO_KERNELS=off python train.py
    - label: Desligar os kernels do Hub sem desinstalar
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Mudar uma família para atenção fundida
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Retorna quantos módulos de atenção foram trocados.
        print(set_fused_attention(model))
    - label: Registrar a sua própria
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## O registro

`libreyolo/kernels/` é um pequeno registro em tempo de execução de
implementações plugáveis. Um slot de op é um nome como `fake_quant_fp8` ou
`ms_deform_attn`. Quem chama pede um slot ao registro e recebe de volta a
primeira implementação registrada cujo predicado passa, com o registro mais
recente ganhando, caindo na implementação de referência quando nada mais se
aplica.

Essa estrutura existe para que uma dependência opcional nunca seja um requisito
rígido. Uma máquina sem Triton, sem CUDA ou sem o pacote `kernels` roda o mesmo
código e produz os mesmos números, só que mais devagar.

| Função | Propósito |
|---|---|
| `active()` | Slot de op para o nome da implementação selecionada, ou `"unavailable"` |
| `resolve(op)` | O callable que rodaria, ou `None` |
| `register(op, impl, *, name, predicate=None)` | Adiciona uma implementação, a mais nova primeiro |
| `unregister(op, name)` | Remove uma |
| `clear_cache()` | Descarta a resolução memoizada |

<code-tabs name="usage" />

Um predicado que levanta exceção é capturado e avisado, nunca propagado, então
uma implementação de terceiros quebrada degrada para o caminho portátil em vez
de quebrar a predição.

### Estrutura

A árvore é organizada primeiro por propósito e depois por backend, de modo que
um slot é encontrado pelo que ele calcula, e não pela biblioteca que por acaso o
implementa hoje.

| Diretório | Conteúdo |
|---|---|
| `kernels/quant/simulate/` | Kernels Triton de fake-quantization, com backward straight-through, em qualquer dispositivo. Usados tanto pelo QAT quanto pela quantização pós-treinamento simulada |
| `kernels/quant/execute/` | Caminhos de precisão real apenas para modelos finalizados, sem backward: o GEMM FP8 em tensor cores, seu prólogo e epílogo fundidos em Triton, e os kernels de unpack de pesos empacotados |
| `kernels/attention/` | Ops de atenção compartilhadas entre famílias: o slot `ms_deform_attn` e a política de SDPA fundida |

A fronteira entre `simulate` e `execute` é se o modelo está finalizado, não se
ele está treinando ou em deploy. As implementações de referência ficam em
`libreyolo/quant/`, que define o que os números significam; `kernels/` só os
deixa rápidos. O empacotamento de pesos não tem variante nenhuma, porque é o
contrato do checkpoint.

Os slots de GEMM e de atenção não têm implementação de referência. Quem chama
precisa conferir que `resolve()` retornou algo e manter o próprio caminho
portátil, e é por isso que grafos ONNX, TensorRT e `torch.export` sempre contêm
a matemática portátil.

### Sobrescrever a seleção

`LIBREYOLO_KERNELS=off` ou `=reference` força as implementações de referência e
curto-circuita completamente a importação dos providers acelerados. Qualquer
outro valor restringe a seleção às implementações registradas sob aquele nome.
`LIBREYOLO_QUANT_KERNELS` é aceito como alias legado da época em que o registro
morava em `libreyolo/quant/`, e só é lido quando `LIBREYOLO_KERNELS` não está
definida. Ambas aparecem junto com as demais em
[configurações](/docs/reference/settings).

## Kernels do Hub

Kernels CUDA compilados publicados no Hugging Face Hub são carregados em tempo
de execução pelo pacote opcional `kernels`. Nada é embutido no LibreYOLO; o
artefato é baixado e mantido em cache por esse pacote, e cada provider fixa uma
revisão de commit auditada, então mexer em um pin exige uma rodada de paridade
em GPU antes de entrar.

Instalar o extra é o opt-in:

```bash
pip install "libreyolo[hub-kernels]"
```

Sem o pacote nada muda e nenhuma requisição de rede é feita.
`LIBREYOLO_HUB_KERNELS=0` desativa o download sem desinstalar nada. Um kernel
que falha ao carregar ou ao rodar se desativa pelo resto do processo e cai no
fallback com um único aviso.

Hoje um slot é servido pelo Hub: `ms_deform_attn`, o forward e o backward
compilados da atenção deformável multiescala do Deformable DETR, sob Apache 2.0.
Ele está ligado a toda a linhagem deformável: RF-DETR, Deformable DETR,
DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM,
DEIMv2, EC e OV-DEIM. Como o backward também é compilado, o treinamento se
beneficia tanto quanto a predição.

A elegibilidade é estreita de propósito. As entradas precisam ser CUDA e
float32, e a execução precisa ser eager: o provider recusa sob
`torch.jit.is_tracing()`, `torch.compiler.is_compiling()`,
`torch.compiler.is_exporting()` e `torch.onnx.is_in_onnx_export()`. Dois layouts
de entrada também caem no caminho portátil, uma contagem de pontos por nível que
varia entre os níveis, e a amostragem por índice inteiro discreto. A variante de
pose do EC não está ligada.

### Este kernel passou a ser alcançável agora

Leia isto antes de instalar o extra em um projeto já existente.

Na v1.4.0 o slot era consultado de dentro de um helper, atrás de uma condição
que exigia a ausência dos pares de spatial shapes. O RF-DETR sempre passa esses
pares pelo decoder, então a condição nunca se sustentava e o kernel nunca
executava em nenhum forward eager. A consulta mudou de lugar na v1.5.0, e agora
o kernel de fato roda.

A consequência prática é que atualizar para a v1.5.0 *e* instalar
`libreyolo[hub-kernels]` em CUDA faz com que o RF-DETR e sua linhagem tirem o
forward de um binário compilado pela primeira vez. Como resultado, predições e
métricas podem variar dentro da tolerância de float. Uma instalação padrão, sem
o extra, não é afetada. Se você estiver comparando métricas antes e depois da
atualização, mantenha o extra fixo ou defina `LIBREYOLO_HUB_KERNELS=0` dos dois
lados.

## Atenção fundida

A atenção fundida do tipo scaled dot-product não precisa de dependência
opcional, só do PyTorch padrão, então ela é governada por política e não por
disponibilidade. Valem duas regras.

Primeiro, uma captura de grafo nunca a usa. Todo ponto de chamada trocado mantém
disponível a equação em ops primitivas atrás de uma checagem de exportação,
cobrindo a exportação para ONNX, cujo opset padrão não tem symbolic para SDPA, e
`torch.jit.trace`, por onde passam TorchScript, CoreML e NCNN. As capturas do
Dynamo ficam deliberadamente fora dessa trava, porque o `torch.compile` faz um
lowering de SDPA melhor do que a matemática manual, e tanto o Core AI quanto o
ExecuTorch decompõem SDPA em core ATen por conta própria.

Segundo, o critério de paridade para torná-la o padrão é ser exata byte a byte.
As famílias que passam nesse critério usam SDPA por padrão: SegFormer, Depth
Anything e MoGe-2, BERT, Grounding DINO, SwinIR e PP-OCR. As que não passam
mantêm a matemática manual e expõem uma flag `fused_attn` no lugar, que é o que
`set_fused_attention(model)` vira: Swin, o backbone Swin do DINO-DETR, BiRefNet
e FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth e MobileSAM. ViT e DeiT carregam a
mesma flag, mas com o padrão ligado, seguindo o upstream, então a mesma chamada
com `enabled=False` os desliga.

Vale a pena onde se aplica. Em uma RTX 5070 Ti sob autocast fp16, a atenção por
janelas do Swin vai de 1.278 ms para 0.721 ms, um ganho de 1.77x, e a atenção de
visão do OWLv2 vai de 6.483 ms para 1.735 ms, 3.74x.

## Hardware

| Plataforma | Comportamento |
|---|---|
| CPU e MPS | Todo predicado de CUDA e de Triton falha, então tudo roda em referência |
| NVIDIA CUDA | Kernels Triton e os kernels elegíveis do Hub e de GEMM entram em ação |
| AMD ROCm | Triton pode entrar em ação, já que as wheels do ROCm trazem o backend AMD do Triton, mas a paridade só é exercitada em NVIDIA no CI |

## Adicionar uma implementação

Chame `register()` com um nome e um predicado. Kernels compilados fora da árvore
podem ser distribuídos como um pacote `libreyolo_kernels` separado que se
registra na importação, o que mantém um backend privado inteiramente fora da
árvore do LibreYOLO.

A paridade é a trava para qualquer coisa dentro da árvore: um forward que bate
exatamente com a referência, e gradientes dentro de 1e-6 do estimador
straight-through, sobre o conjunto de shapes que a suíte de testes carrega.

A seleção de kernels interage com os [grafos CUDA](/docs/reference/cuda-graphs):
a matriz de paridade de inferência rodou sem o pacote `kernels` instalado, então
a segurança de captura com um kernel compilado ativo não está coberta por ela.
