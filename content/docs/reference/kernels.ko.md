---
title: 커널
seo_title: LibreYOLO 커널 레지스트리 및 Hub 커널
description: >-
  LibreYOLO가 가속 구현을 선택하는 방법: libreyolo/kernels 하의 커널 레지스트리, 선택적 Hugging Face Hub
  MS-deform-attn 커널, 그리고 결합된 어텐션 스위치
lead: >-
  LibreYOLO의 모든 가속 연산에는 휴대 가능한 기본값이 있으며, 때로는 그 위에 더 빠른 변형이 등록됨. 선택은 런타임의
  조건(predicate)에 따라 이루어지며, 선택적 종속성이 없으면 오류가 아닌 대체(fallback)를 사용하고, 내보낸 그래프는 항상
  휴대 가능한 경로를 따름
keywords:
  - libreyolo 커널
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels 추가
  - ms_deform_attn 커널
  - set_fused_attention
  - libreyolo triton 커널
last_verified: 1.5.0
verification: >-
  레지스트리 API는 v1.5.0에서 libreyolo/kernels/__init__.py에서 읽었으며, 주의 API는
  libreyolo/kernels/attention/__init__.py와 sdpa.py에서, 허브 공급자는
  libreyolo/kernels/attention/ms_deform_attn.py에서 그 고정된 개정과 적격성 조건을 포함합니다. 디렉토리
  레이아웃은 libreyolo/kernels/.에서 나열됩니다. 추가 정의는 pyproject.toml에서 제공됩니다. 동작 노트와 벤치마크
  수치는 docs/kernels.md.에서 확인할 수 있습니다. v1.4.0 게이팅 기록은 RF-DETR 슬롯-배선 커밋에서와 1.5.0
  CHANGELOG 항목에서 확인할 수 있습니다.
meta:
  - label: 패키지
    value: libreyolo.kernels
    mono: true
  - label: 옵트인 추가
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

        # 선택된 구현 이름으로의 Op 슬롯, 또는 "사용 불가".
        print(kernels.active())
    - label: 참조 경로 강제
      language: bash
      code: |
        # off와 reference는 같은 의미이며, 또한 건너뛰기를 의미합니다
        # 가속 공급자 가져오기를 전혀 수행하지 않습니다.
        LIBREYOLO_KERNELS=off python train.py
    - label: 허브 커널을 제거하지 않고 끄기
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: 계열을 융합된 주의로 전환
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # 몇 개의 어텐션 모듈이 전환되었는지 반환합니다.
        print(set_fused_attention(model))
    - label: 자신만의 것을 등록합니다
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

## 레지스트리

`libreyolo/kernels/`는 플러그인 가능한 구현의 작은 런타임 레지스트리입니다. 연산 슬롯(op slot)은 `fake_quant_fp8` 또는 `ms_deform_attn`와 같은 이름입니다. 호출자는 레지스트리에 슬롯을 요청하고 처음으로 조건(predicate)을 통과한 등록된 구현을 돌려받습니다. 가장 최근 등록된 것이 우선하며, 다른 것이 적용되지 않으면 참조 구현(reference implementation)으로 넘어갑니다.

이 구조는 선택적 의존성이 절대적인 조건이 되지 않도록 존재합니다. Triton, CUDA 또는 `kernels` 패키지가 없는 머신에서도 동일한 코드를 실행하고 동일한 결과를 생성하지만 단지 느리게 실행됩니다.

| 함수 | 목적 |
|---|---|
| `active()` | 선택된 구현 이름 또는 `"unavailable"`에 대한 연산 슬롯 |
| `resolve(op)` | 실행될 호출 가능 객체, 또는 `None` |
| `register(op, impl, *, name, predicate=None)` | 구현 추가, 최신순 |
| `unregister(op, name)` | 하나 제거 |
| `clear_cache()` | 메모이제이션된 해상도 제거 |

<code-tabs name="usage" />

예외를 발생시키는 술어는 잡혀 경고가 표시되며, 전파되지 않으므로, 손상된 서드파티 구현은 예측을 깨뜨리는 대신 포터블 경로로 전환된다.

### 레이아웃

트리는 목적 우선, 백엔드 후순으로 조직되므로 슬롯은 현재 어떤 라이브러리가 구현했는지보다는 계산하는 내용에 따라 찾아진다.

| 디렉토리 | 내용 |
|---|---|
| `kernels/quant/simulate/` | 모든 장치에서 직통 역전파를 사용하는 가짜 양자화 Triton 커널. QAT와 시뮬레이션된 사후 학습 양자화 모두에서 사용 |
| `kernels/quant/execute/` | 최종 모델 전용 실수 정밀 경로, 역전파 없음: FP8 텐서 코어 GEMM, 그 융합된 Triton 프로로그 및 에필로그, 그리고 패킹된 가중치 언팩 커널 |
| `kernels/attention/` | 패밀리 간 공유되는 어텐션 연산: `ms_deform_attn` 슬롯 및 융합-SDPA 정책 |

`simulate`와 `execute` 사이의 경계는 모델이 최종화되었는지 여부이지, 학습 중인지 배포 중인지 여부가 아님. 참조 구현은 `libreyolo/quant/`에 남아 있으며, 이는 숫자의 의미를 정의함; `kernels/`는 단지 이를 빠르게 만듦. 가중치 패킹에는 변형이 전혀 없음, 왜냐하면 그것은 체크포인트 계약이기 때문임.

GEMM과 어텐션 슬롯에는 참조 구현이 없음. 호출자는 `resolve()`가 뭔가를 반환했는지 확인하고 자체적인 포터블 경로를 유지해야 하며, 이것이 ONNX, TensorRT 및 `torch.export` 그래프가 항상 포터블 수학을 포함하는 이유임.

### 선택 무시

`LIBREYOLO_KERNELS=off` 또는 `=reference`는 참조 구현을 강제로 지정하며 가속 제공자의 가져오기를 완전히 우회합니다. 다른 값은 해당 이름으로 등록된 구현으로 선택을 제한합니다. `LIBREYOLO_QUANT_KERNELS`는 레지스트리가 `libreyolo/quant/` 아래에 있었을 때의 이전 별칭으로 유효하며, `LIBREYOLO_KERNELS`가 설정되지 않은 경우에만 읽힙니다. 두 값 모두 [설정](/docs/reference/settings)의 나머지 항목과 함께 나열됩니다.

## 허브 커널

Hugging Face Hub에 게시된 컴파일된 CUDA 커널은 선택적 `kernels` 패키지를 통해 런타임에 로드됩니다. LibreYOLO에는 어떤 것이 포함되어 있지 않으며, 산출물은 해당 패키지에서 가져오고 캐시하며, 각 제공자는 감사된 커밋 리비전을 고정하므로 고정 버전을 올리려면 배포 전에 GPU 일치 실행이 필요합니다.

추가 설치는 선택적입니다:

```bash
pip install "libreyolo[hub-kernels]"
```

패키지가 없으면 아무 것도 변경되지 않으며 네트워크 요청도 이루어지지 않습니다. `LIBREYOLO_HUB_KERNELS=0`는 아무 것도 제거하지 않고 fetch를 비활성화합니다. 로드하거나 실행하지 못하는 커널은 나머지 프로세스에서 스스로 비활성화되며 하나의 경고와 함께 대체됩니다.

오늘날 하나의 슬롯이 Hub 기반입니다: `ms_deform_attn`, Apache 2.0 하에 Deformable DETR에서 컴파일된 다중 스케일 변형 가능 주의(attention) 순방향(forward) 및 역방향(backward)입니다. 이는 전체 변형 가능 계열(RF-DETR, Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM, DEIMv2, EC 및 OV-DEIM)에 연결되어 있습니다. 역방향 또한 컴파일되어 있기 때문에 학습뿐만 아니라 예측에도 이점이 있습니다.

적격성은 의도적으로 좁습니다. 입력은 CUDA와 float32여야 하며, 실행은 즉시 수행(eager)되어야 합니다: 제공자는 `torch.jit.is_tracing()`, `torch.compiler.is_compiling()`, `torch.compiler.is_exporting()` 및 `torch.onnx.is_in_onnx_export()` 하에서는 거부됩니다. 두 가지 입력 레이아웃 또한 이동식 경로(portable path)로 넘어가며, 레벨마다 달라지는 각 레벨별 점(point) 수와 개별 정수 인덱스 샘플링이 포함됩니다. EC 포즈(EC pose) 변형은 연결되어 있지 않습니다.

### 이 커널은 새로 접근 가능해졌습니다.

기존 프로젝트에 추가 기능을 설치하기 전에 이 내용을 읽으세요.

v1.4.0에서는 슬롯이 조건 뒤에 있는 헬퍼 내부에서 확인되었습니다. 그 조건은 공간-형상 쌍(spatial-shape pairs)이 없을 경우를 요구했습니다. RF-DETR은 항상 디코더를 통해 해당 쌍을 전달하므로 조건이 충족되지 않았고, 커널은 어떤 즉시 실행(forward)에서도 실행되지 않았습니다. v1.5.0에서 확인 위치가 이동되었고, 이제 커널이 실제로 실행됩니다.

실제적인 결과는 v1.5.0으로 업그레이드하고 CUDA에 `libreyolo[hub-kernels]`를 설치하면 RF-DETR 및 그 계열이 처음으로 컴파일된 바이너리에서 순방향 연산을 수행한다는 것입니다. 그 결과, 예측 및 성능 지표가 부동 소수점 허용 오차 범위 내에서 변경될 수 있습니다. 추가 항목 없이 기본 설치를 한 경우에는 영향이 없습니다. 업그레이드 전후로 지표를 비교할 경우, 추가 항목을 고정하거나 양쪽 모두에 `LIBREYOLO_HUB_KERNELS=0`를 설정하십시오.

## 융합 주의(fused attention)

융합 스케일 점곱 주의(fused scaled dot-product attention)는 선택적 의존성이 필요 없고, 기본 PyTorch만 필요하므로, 가용성보다는 정책에 따라 결정됩니다. 두 가지 규칙이 적용됩니다.

먼저, 그래프 캡처는 절대로 그것을 사용하지 않습니다. 모든 교환된 호출 지점은 ONNX 내보내기를 포괄하는 내보내기 확인 뒤에 원시 연산 방정식을 계속 유지합니다. 기본 opset에는 SDPA 심볼릭이 없고, `torch.jit.trace`는 TorchScript, CoreML, NCNN 모두가 통과합니다. 다이너모 캡처는 의도적으로 게이트 밖에 위치하는데, `torch.compile`가 수동 수학보다 SDPA를 더 잘 낮추고, Core AI와 ExecuTorch 모두 SDPA를 자체적으로 핵심 ATen으로 분해하기 때문입니다.

둘째, 기본값으로 설정하기 위한 패리티 바는 바이트 단위로 정확합니다. 이를 지우는 패밀리는 기본적으로 SDPA를 사용합니다: SegFormer, Depth Anything, MoGe-2, BERT, Grounding DINO, SwinIR, PP-OCR. 수동 계산을 유지하지 않고 대신 `fused_attn` 플래그를 노출하는 패밀리도 있으며, `set_fused_attention(model)`가 이를 전환합니다: Swin, DINO-DETR의 Swin 백본, BiRefNet, FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth, MobileSAM. ViT와 DeiT는 같은 플래그를 가지고 있지만 업스트림을 따라 기본값을 켜놓기 때문에 `enabled=False`와 함께 같은 호출을 하면 이를 끕니다.

적용 가능한 곳에서는 실행할 가치가 있습니다. fp16 자동 캐스트 하의 RTX 5070 Ti에서 Swin 윈도우 어텐션은 1.278 ms에서 0.721 ms로, 1.77배 향상되며, OWLv2 비전 어텐션은 6.483 ms에서 1.735 ms로, 3.74배 향상됩니다.

## 하드웨어

| 플랫폼 | 동작 |
|---|---|
| CPU 및 MPS | 모든 CUDA 및 Triton 조건이 실패하므로 모든 것이 참조(reference)로 실행됩니다 |
| NVIDIA CUDA | Triton 커널 및 적격한 Hub와 GEMM 커널이 작동합니다 |
| AMD ROCm | ROCm 휠이 Triton의 AMD 백엔드를 제공하므로 Triton이 작동할 수 있지만, CI에서는 NVIDIA에서만 동등성이 확인됩니다 |

## 구현 추가

이름과 조건과 함께 `register()`를 호출하세요. 트리 외부에서 컴파일된 커널은 자체를 가져오기 시 등록하는 별도의 `libreyolo_kernels` 패키지로 제공될 수 있으며, 이를 통해 LibreYOLO 트리에서 완전히 개인 백엔드를 분리할 수 있습니다.

동등성(parity)은 트리 내부의 모든 것에 대한 관문입니다: 참조와 정확히 동일한 순방향 매치(forward match)와, 테스트 슈트가 가진 형태 집합 상에서 스트레이트-스루 추정기(straight-through estimator) 기준으로 1e-6 이내의 그래디언트.

커널 선택은 [CUDA 그래프](/docs/reference/cuda-graphs)와 상호작용합니다: 추론 패리티 매트릭스는 `kernels` 패키지가 설치되지 않은 상태에서 실행되었으므로, 컴파일된 커널이 활성화된 상태에서의 캡처 안전성은 이를 포함하지 않습니다.
