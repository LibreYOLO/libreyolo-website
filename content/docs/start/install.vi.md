---
title: Cài đặt
seo_title: Cài đặt LibreYOLO
description: >-
  Cài LibreYOLO từ PyPI, chọn các gói bổ sung mà họ mô hình hoặc đích xuất cần,
  rồi xác nhận PyTorch nhận diện GPU.
lead: >-
  LibreYOLO được phát hành trên PyPI với tên libreyolo. Package cơ sở hỗ trợ dự
  đoán, huấn luyện, xác thực và các họ mô hình không cần gì ngoài PyTorch; các
  gói bổ sung cung cấp phần còn lại.
keywords:
  - cài libreyolo
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo gpu
  - yêu cầu libreyolo
last_verified: 1.5.0
meta:
  - label: Package
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 trở lên
  - label: Giấy phép mã nguồn
    value: MIT
  - label: Dependency cốt lõi
    value: PyTorch 2.4 trở lên
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Kèm gói bổ sung
      language: bash
      code: |
        # Phân cách bằng dấu phẩy để kết hợp nhiều gói trong một lần cài.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Tất cả
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Từ mã nguồn
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, mọi GPU nhìn thấy và các
        # package tùy chọn đã được cài.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Danh mục mô hình
      language: bash
      code: |
        # Mọi họ đã đăng ký cùng tác vụ, kích thước và độ phân giải
        # đầu vào. Họ còn thiếu gói bổ sung được liệt kê kèm lệnh pip
        # để kích hoạt.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Cài đặt

<code-tabs name="install" />

Yêu cầu Python 3.10 trở lên. Bản cài cơ sở lấy PyTorch, torchvision, NumPy,
Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools, typer, click,
safetensors và SciPy, nên YOLOv9 cùng các họ không cần thêm thành phần nào sẽ
hoạt động ngay sau `pip install libreyolo`.

Bản clone checkout nhánh `release`, là nhánh ổn định có mã khớp với tài liệu
này. Nhánh tích hợp chứa công việc chưa phát hành là `dev`.

## Gói bổ sung tùy chọn

Gói bổ sung là tên trong ngoặc vuông để thêm dependency mà một họ mô hình hoặc
một đích xuất cần. Không có gì khác thay đổi: API giống nhau dù gói bổ sung có
được cài hay không.

### Họ mô hình

| Gói bổ sung | Thành phần thêm vào |
|---|---|
| `rfdetr` | `transformers`, cung cấp backbone RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, cung cấp encoder ViT-L/16 và EfficientNet-Lite3 của MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate`, và `bitsandbytes` ngoài macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` và `ftfy`, cần cho tokenizer văn bản CLIP vendored |
| `siglip2` | `sentencepiece`, cần cho tokenizer SigLIP 2 đa ngôn ngữ |
| `gaze` | `gdown`, kích hoạt tự động tải checkpoint L2CS |
| `rtdetr` | Không có. RT-DETR không cần dependency bổ sung; tên được giữ ổn định |

### Xuất và runtime

| Gói bổ sung | Thành phần thêm vào |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 và `pycuda`, ngoài macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, chỉ macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` cùng `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` và `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` cùng `MNN` |
| `ncnn` | `pnnx` và `ncnn` |
| `paddle` | `libreyolo[onnx]` cùng `paddlepaddle` 2.6.2 và `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` cho inference V2 qua HTTP và HTTPS |

### Huấn luyện, đánh giá và logging

| Gói bổ sung | Thành phần thêm vào |
|---|---|
| `lora` | `libreyolo[rfdetr]` cùng `peft`, cho tinh chỉnh `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, backend đánh giá COCO bằng C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` là tùy chọn thay vì dependency bắt buộc để nền tảng không có wheel
dựng sẵn vẫn cài đặt bình thường. Khi package không có mặt, đánh giá COCO quay
về pycocotools và quá trình chạy tiếp tục.

### Công cụ

| Gói bổ sung | Thành phần thêm vào |
|---|---|
| `stream` | `yt-dlp`, chỉ cần để phân giải URL trang YouTube |
| `tracking` | Không có. Mọi dependency tracking đã là dependency cốt lõi |
| `label` | `libreyolo[sam]`, kích hoạt hỗ trợ click-to-mask trong `libreyolo label` |
| `hub-kernels` | `kernels`, loader tùy chọn cho Hub kernel đã biên dịch. Xem [kernel](/docs/reference/kernels), phần này lưu ý việc cài có thể làm thay đổi dự đoán RF-DETR trong sai số float |
| `clip-convert` | `libreyolo[clip]` cùng `open_clip_torch`, để chuyển đổi trọng số và kiểm tra parity |
| `siglip2-convert` | `libreyolo[siglip2]` cùng `transformers`, với cùng mục đích |

Webcam, RTSP, RTMP, TCP, UDP, HLS và danh sách multi-stream cục bộ không cần
gói bổ sung. Chỉ URL trang YouTube cần.

### Gói bổ sung tổng hợp

`libreyolo[all]` cài các gói bổ sung cho mô hình, xuất, tracking và logging
trong một lệnh. Một số gói được cố ý để ngoài. `neptune` bị loại vì bản
`neptune-scale` ổn định cần protobuf dưới 7 trong khi pipeline TFLite cần
protobuf 7. `executorch` bị loại vì ExecuTorch giới hạn phiên bản PyTorch đi
kèm, còn `coreai` vì `coreai-torch` cố định PyTorch ở 2.11.x và sẽ kéo toàn bộ
môi trường sang phiên bản đó. `fast-eval`, `hub-kernels`, `clip-convert` và
`siglip2-convert` cũng không được đưa vào. Hãy cài riêng theo tên nếu cần.

## Ràng buộc nền tảng

Ba gói bổ sung được giới hạn nền tảng bằng dependency marker, vì vậy cài đặt
thành công ở mọi nơi và chỉ cài ít thành phần hơn khi wheel không tồn tại.

| Gói bổ sung | Ràng buộc |
|---|---|
| `coreai` | Chỉ macOS. Toolchain Core AI không chuyển đổi hoặc chạy ở nơi khác |
| `tensorrt` | Bỏ qua trên macOS vì không có CUDA |
| `tflite`, `litert` | `onnx2tf` và `ai-edge-litert` cần Python 3.12 trở lên |

`sensenova` bỏ qua `bitsandbytes` trên macOS vì không có wheel được công bố;
phần còn lại của gói bổ sung vẫn cài bình thường.

Nếu ổ đĩa là giới hạn, phần lớn dung lượng thuộc về PyTorch, và phần lớn
PyTorch là payload CUDA đi kèm wheel mặc định. Wheel chỉ dùng CPU loại bỏ phần
đó mà không mất tính năng. Với detection ONNX trên máy không được có torch,
xem [cài đặt gọn nhẹ](/docs/lightweight-install).

## GPU và CUDA

Thiết bị được chọn khi dựng mô hình. Giá trị mặc định `device="auto"` dùng CUDA
khi `torch.cuda.is_available()` là true, tiếp theo dùng Metal Performance
Shaders khi `torch.backends.mps.is_available()` là true, còn lại dùng CPU.
Không phần nào khác trong thư viện kiểm tra phần cứng, nên nếu PyTorch không
nhìn thấy GPU thì LibreYOLO cũng không thấy.

Để chỉ định thiết bị, truyền `device` cho mô hình hoặc cho `predict`, `train`,
`val` và `export`. Nó nhận `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, một số
nguyên như `0` hoặc chuỗi chữ số như `"0"`; hai dạng cuối được mở rộng thành
`cuda:<n>`.

Hãy bắt đầu với `libreyolo checks`, lệnh in phiên bản Torch, phiên bản CUDA và
cuDNN dùng để build Torch, cùng mọi GPU nhìn thấy và bộ nhớ của chúng. Khi lệnh
không báo CUDA trên máy có card NVIDIA, wheel PyTorch mà pip phân giải là bản
CPU. Trước tiên hãy cài bản CUDA từ index PyTorch, rồi cài LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Đây cũng là index mà repo cố định cho môi trường do uv quản lý trên Linux và
Windows. Nó cần driver NVIDIA 555 trở lên, là yêu cầu runtime CUDA 12.8. macOS
giữ wheel PyPI vì máy chủ tải PyTorch không công bố bản build Darwin.

## Kiểm tra bản cài

<code-tabs name="verify" />

`libreyolo models` là cách nhanh nhất để xem gói bổ sung đã có hiệu lực chưa:
họ còn thiếu dependency được in kèm lệnh pip chính xác để kích hoạt. Cả hai
lệnh cũng nhận `--json`, in cùng dữ liệu dưới dạng object máy đọc được ra
stdout.
