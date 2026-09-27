---
title: 커널
seo_title: LibreYOLO 커널 레지스트리 및 허브 커널
description: >-
  LibreYOLO가 가속화된 구현을 선택하는 방법: libreyolo/kernels 아래의 커널 레지스트리, 선택적 Hugging Face
  Hub MS-deform-attn 커널, 그리고 융합된 어텐션 스위치.
lead: >-
  LibreYOLO의 모든 가속화된 연산에는 이동 가능한 기본값이 있으며, 때로는 그 위에 더 빠른 변형이 등록되어 있습니다. 선택은 런타임에
  조건(predicate)에 따라 이루어지며, 누락된 선택적 의존성은 오류가 아니라 대체 수단으로 처리되고, 내보낸 그래프는 항상 이동 가능한
  경로를 사용합니다.
keywords:
  - libreyolo 커널
  - 리브레욜로_커널
  - 리브레욜로_허브_커널
  - 허브-커널 추가
  - ms_deform_attn 커널
  - 융합 주의 설정
  - 리브레욜로 트라이톤 커널
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/kernels/__init__.py로부터 읽은 레지스트리 API,
  libreyolo/kernels/attention/__init__.py 및 sdpa.py로부터의 어텐션 API, 고정된 리비전과 적격성
  조건을 포함한 libreyolo/kernels/attention/ms_deform_attn.py로부터의 허브 프로바이더.
  libreyolo/kernels/.로부터 나열된 디렉터리 레이아웃, pyproject.toml로부터의 추가 정의.
  docs/kernels.md.로부터의 동작 노트와 벤치마크 수치. RF-DETR 슬롯 배선 커밋에서의 v1.4.0 게이팅 히스토리와
  1.5.0 CHANGELOG 항목.
meta:
  - label: 패키지
    value: libreyolo.kernels
    mono: true
  - label: 추가 선택
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: 참조 강제
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: 선택된 항목 보기
      language: python
      code: |
        import libreyolo.kernels as kernels

        # 선택한 구현 이름에 대한 Op 슬롯 또는 '사용 불가'.
        print(kernels.active())
    - label: 참조 경로 강제하기
      language: bash
      code: |
        # off와 reference는 둘 다 같은 의미이고, 또한 skip도 같다
        # 가속화된 공급자를 전혀 가져오지 않습니다.
        LIBREYOLO_KERNELS=off python train.py
    - label: 허브 커널을 제거하지 않고 끄기
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: 계열을 융합 주의로 전환
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # 몇 개의 어텐션 모듈이 전환되었는지 반환합니다.
        print(set_fused_attention(model))
    - label: 직접 등록하십시오
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

## 등록부

`libreyolo/kernels/`는 플러그형 구현의 작은 런타임 레지스트리입니다. op 슬롯은 `fake_quant_fp8` 또는 `ms_deform_attn`와 같은 이름입니다. 호출자는 레지스트리에 슬롯을 요청하고, 등록된 구현 중에서 먼저 조건을 만족하는 구현을 받습니다. 가장 최신 등록이 우선하며, 다른 것이 적용되지 않으면 참조 구현으로 넘어갑니다.

그 구조는 선택적 의존성이 결코 필수 요구사항이 되지 않도록 존재합니다. Triton, CUDA, 또는 `kernels` 패키지가 없는 기계도 동일한 코드를 실행하고 동일한 숫자를 생성하지만, 단지 더 느리게 수행됩니다.

| 기능 | 목적 |
|---|---|
| `active()` | 선택한 구현 이름 또는 `"unavailable"`에 슬롯 |
| `resolve(op)` | 실행될 호출 가능 객체, 또는 `None` |
| `register(op, impl, *, name, predicate=None)` | 구현을 추가하되 최신순으로 |
| `unregister(op, name)` | 하나 제거 |
| `clear_cache()` | 메모이제이션된 해상을 제거하다 |

<code-tabs name="usage" />

발생하는 술어는 잡혀서 경고를 받고, 절대 전달되지 않으므로, 손상된 서드파티 구현도 예측을 깨뜨리지 않고 휴대 가능한 경로로 실패합니다.

### 레이아웃

트리는 먼저 목적에 따라, 그 다음에 백엔드에 따라 조직되므로, 슬롯은 오늘날 어떤 라이브러리가 구현했는지보다는 무엇을 계산하는지에 따라 찾아집니다.

| 디렉토리 | 내용 |
|---|---|
| `kernels/quant/simulate/` | 임의 장치에서 직통 역전파를 사용하는 페이크 양자화 Triton 커널. QAT와 시뮬레이션된 사후 학습 양자화 모두에서 사용됨 |
| `kernels/quant/execute/` | 최종 모델에 대한 실수 정밀 경로만, 역전파 없음: FP8 텐서 코어 GEMM, 그 융합된 Triton 서두 및 후두, 그리고 패킹된 가중치 언팩 커널 |
| `kernels/attention/` | 계열 간에 공유되는 주의 작업: `ms_deform_attn` 슬롯 및 융합 SDPA 정책 |

`simulate`와 `execute`의 경계는 모델이 최종화되었는지 여부이지, 학습 중인지 배포 중인지는 아닙니다. 참조 구현은 `libreyolo/quant/`에 남아 있으며 여기서 숫자의 의미를 정의합니다. `kernels/`는 이를 빠르게 만들 뿐입니다. 가중치 패킹에는 전혀 변형이 없는데, 이는 체크포인트 계약이기 때문입니다.

GEMM과 어텐션 슬롯에는 참조 구현이 없습니다. 호출자는 `resolve()`가 무언가를 반환했는지 확인하고 자체적인 포터블 경로를 유지해야 합니다. 이것이 ONNX, TensorRT 및 `torch.export` 그래프에 항상 포터블 수학이 포함되는 이유입니다.

### 선택 덮어쓰기

`LIBREYOLO_KERNELS=off` 또는 `=reference`는 참조 구현을 강제로 사용하게 하며 가속 제공자의 가져오기를 완전히 단축시킵니다. 다른 값은 해당 이름으로 등록된 구현으로 선택을 제한합니다. `LIBREYOLO_QUANT_KERNELS`는 레지스트리가 `libreyolo/quant/` 아래에 있을 때의 이전 별칭으로 존중되며, `LIBREYOLO_KERNELS`가 설정되지 않은 경우에만 읽힙니다. 두 값 모두 나머지 항목과 함께 [설정](/docs/reference/settings)에 나열되어 있습니다.

## 허브 커널

Hugging Face Hub에 게시된 컴파일된 CUDA 커널은 선택적 `kernels` 패키지를 통해 런타임에 로드됩니다. LibreYOLO에는 어떤 것도 포함되어 있지 않으며, 아티팩트는 해당 패키지에 의해 가져와지고 캐시됩니다. 각 공급자는 감사된 커밋 리비전을 고정하므로, 핀을 올리려면 그것이 적용되기 전에 GPU 동등성 실행이 필요합니다.

추가 기능 설치는 선택 사항입니다:

```bash
pip install "libreyolo[hub-kernels]"
```

패키지가 없으면 아무것도 변경되지 않으며 네트워크 요청도 이루어지지 않습니다. `LIBREYOLO_HUB_KERNELS=0`는 아무것도 제거하지 않고 가져오기를 비활성화합니다. 로드나 실행에 실패한 커널은 나머지 프로세스 동안 스스로를 비활성화하고 하나의 경고와 함께 백업됩니다.

오늘 하나의 슬롯이 허브에 의해 지원됩니다: `ms_deform_attn`, Deformable DETR에서 컴파일된 다단계 변형 주의 순전파 및 역전파로, Apache 2.0 하에 있습니다. 이는 전체 변형 계열에 연결되어 있습니다: RF-DETR, Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM, DEIMv2, EC 및 OV-DEIM. 역전파도 컴파일되어 있기 때문에, 예측뿐만 아니라 학습에도 혜택을 줍니다.

자격 요건은 의도적으로 좁습니다. 입력은 CUDA 및 float32여야 하며, 실행은 즉시 실행(eager)이어야 합니다: 공급자는 `torch.jit.is_tracing()`, `torch.compiler.is_compiling()`, `torch.compiler.is_exporting()` 및 `torch.onnx.is_in_onnx_export()`에서는 거절합니다. 두 개의 입력 레이아웃도 휴대 가능한 경로로 넘어갑니다, 레벨마다 다른 포인트 수와, 이산 정수 인덱스 샘플링이 있습니다. EC 포즈 변형은 연결되어 있지 않습니다.

### 이 커널은 새로 접근 가능

기존 프로젝트에 추가 기능을 설치하기 전에 이것을 읽으십시오.

v1.4.0에서는 슬롯이 헬퍼 내부에서 참조되었고, 공간-모양 쌍이 없는 경우라는 조건 뒤에 있었습니다. RF-DETR는 항상 이러한 쌍을 디코더를 통해 전달하므로, 조건이 성립하지 않아 커널이 어느 이거 포워드에서도 실행되지 않았습니다. v1.5.0에서 참조 위치가 이동되었고, 이제 커널이 실제로 실행됩니다.

실질적인 결과는 v1.5.0으로 업그레이드하고 CUDA에 `libreyolo[hub-kernels]`를 설치하면 RF-DETR과 그 계열이 처음으로 컴파일된 바이너리에서 순방향을 수행한다는 것입니다. 그 결과 예측과 지표가 부동 소수점 허용 오차 범위에서 변동할 수 있습니다. 추가 설치 없이 기본 설치만 있는 경우에는 영향을 받지 않습니다. 업그레이드 전후로 지표를 비교할 때는 추가 설치를 고정하거나 양쪽에 `LIBREYOLO_HUB_KERNELS=0`를 설정하십시오.

## 융합 주의

융합된 스케일드 닷-프로덕트 어텐션은 선택적 종속성이 필요 없고, 기본 PyTorch만 필요하므로 가용성이 아니라 정책에 따라 적용됩니다. 두 가지 규칙이 적용됩니다.

첫째, 그래프 캡처는 결코 그것을 사용하지 않습니다. 교환된 모든 호출 사이트는 ONNX 내보내기를 포함하여 기본 연산(opset)에서 SDPA 상징이 없고, TorchScript, CoreML 및 NCNN이 모두 거치는 `torch.jit.trace`를 포함한 내보내기 검사를 통해 원시 연산 방정식을 유지합니다. 다이나모 캡처는 일부러 게이트 밖에 두는데, `torch.compile`가 수동 수학보다 SDPA를 더 잘 낮추기 때문이며, Core AI와 ExecuTorch 모두 SDPA를 자체적으로 핵심 ATen으로 분해합니다.

둘째, 기본값으로 만들기 위한 패리티 바는 바이트 단위로 정확합니다. 이를 지우는 계열는 기본적으로 SDPA를 사용합니다: SegFormer, Depth Anything, MoGe-2, BERT, Grounding DINO, SwinIR, PP-OCR. 수동 계산을 유지하지 않고 대신 `fused_attn` 플래그를 노출하는 계열도 있는데, 이 플래그를 `set_fused_attention(model)`가 전환합니다: Swin, DINO-DETR의 Swin 백본, BiRefNet, FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth, MobileSAM. ViT와 DeiT도 동일한 플래그를 가지고 있지만 기본적으로 켜져 있으며, 업스트림을 따릅니다. 따라서 `enabled=False`로 동일한 호출을 하면 꺼집니다.

적용되는 곳에서는 해볼 가치가 있습니다. fp16 오토캐스트 하에서 RTX 5070 Ti에서 Swin 윈도우 어텐션은 1.278 ms에서 0.721 ms로, 1.77배 향상되며, OWLv2 비전 어텐션은 6.483 ms에서 1.735 ms로, 3.74배 향상됩니다.

## 하드웨어

| 플랫폼 | 행동 |
|---|---|
| CPU 및 MPS | 모든 CUDA 및 Triton 조건이 실패하여 모든 것이 참조로 실행됩니다 |
| 엔비디아 쿠다 | 트라이톤 커널과 적격한 허브 및 GEMM 커널이 작동한다 |
| AMD ROCm | Triton은 ROCm 휠이 Triton의 AMD 백엔드를 제공하기 때문에 작동할 수 있지만, 동등성 검사는 CI에서 NVIDIA에서만 수행됩니다. |

## 구현 추가

`register()`에 이름과 술어를 사용하여 호출하십시오. 트리 외부에서 컴파일된 커널은 가져오기 시 자체적으로 등록되는 별도의 `libreyolo_kernels` 패키지로 제공될 수 있으며, 이는 개인 백엔드를 LibreYOLO 트리 외부에 완전히 유지합니다.

패리티는 트리 내 어떤 것에 대한 관문입니다: 참조에 대한 정확한 정방향 일치와 테스트 스위트가 가진 형태 집합에 대해 스트레이트 스루 추정기와 1e-6 이내의 그래디언트입니다.

커널 선택은 [CUDA 그래프](/docs/reference/cuda-graphs)와 상호작용합니다: 추론 파리티 매트릭스는 `kernels` 패키지가 설치되지 않은 상태에서 실행되었으므로, 컴파일된 커널이 활성화된 상태에서의 캡처 안전성은 이에 의해 보장되지 않습니다.
