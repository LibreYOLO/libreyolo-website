---
title: NVIDIA Jetson
seo_title: Cài LibreYOLO và PyTorch trên NVIDIA Jetson
description: >-
  Cài LibreYOLO trên NVIDIA Jetson: bốn thư viện CUDA mà JetPack bỏ sót, bước
  --no-deps mà PyTorch cần, và số đo thực tế trên Orin Nano.
lead: >-
  Các bo mạch NVIDIA Jetson chạy LibreYOLO bằng wheel PyTorch aarch64 tiêu
  chuẩn. Không cần bản build torch riêng cho Jetson, nhưng JetPack thiếu bốn thư
  viện mà torch liên kết tới, và quá trình cài phải bổ sung chúng.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - cài PyTorch trên Jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - TensorRT trên Jetson
  - wheel aarch64
last_verified: 1.4.0
meta:
  - label: Bo mạch
    value: 'Jetson Orin Nano Super Developer Kit, 8 GB, GPU compute capability 8.7'
  - label: Nền tảng
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Stack đã kiểm thử
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, vào ngày 2026-07-27
  - label: Thiếu trong JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Đã benchmark
    value: >-
      223 lần chạy đã kiểm chứng trên bo mạch này, 58 mô hình thuộc 12 họ, trên
      PyTorch, ONNX Runtime và TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Theo dõi tại
    value: Phần Jetson của issue 648
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Công thức cài đặt và kết quả mong đợi lấy từ lần chạy cài đặt ngày 2026-07-27
  trên một Jetson Orin Nano Super. Các dòng độ trễ và độ chính xác lấy từ bản
  chụp kết quả đã kiểm chứng phía sau visionanalysis.org, lọc theo phần cứng
  jetson_orin, đo vào tháng 6 năm 2026 trên libreyolo 1.2.0.dev0. Hành vi xuất
  mô hình và nạp mô hình đọc từ libreyolo/export/exporter.py,
  libreyolo/export/tensorrt.py và libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Gói hệ thống và một môi trường ảo
      language: bash
      code: |
        # JetPack không cài sẵn pip hay module venv
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: 'PyTorch, từ index wheel CUDA 13'
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: Bốn thư viện JetPack không kèm theo
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Nếu pip đòi cuda-toolkit 13.0.3, hãy cài bằng --no-deps'
      language: bash
      code: >
        # --no-deps nghĩa là các phụ thuộc Python của torch cũng phải liệt kê
        bằng tay

        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Chỉ đích danh thư viện còn thiếu tiếp theo thay vì đoán mò
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Mọi thứ còn thiếu trên toàn bộ thư viện của torch, trong một lượt:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Cài LibreYOLO sau torch, không phải trước'
      language: bash
      code: |
        # torch đã được thỏa mãn nên pip giữ nguyên bản build CUDA
        pip install libreyolo

        # Extra ONNX chỉ cần khi xuất mô hình. Một bản xuất TensorRT đi qua
        # ONNX, nên hãy cài nó trước phần xuất mô hình bên dưới
        pip install "libreyolo[onnx]"
  verify:
    - label: Phiên bản và thiết bị
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Rồi chạy một kernel thật
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Tự tải checkpoint ở lần dùng đầu tiên
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Ghi ra libreyolo9s.onnx, rồi build libreyolo9s.engine từ đó

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Engine được nạp lại qua cùng một entry point

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Chế độ nguồn và xung nhịp
      language: bash
      code: >
        sudo nvpmodel -q      # bo mạch này có những chế độ nào, và chế độ đang
        bật

        sudo nvpmodel -m 0    # chế độ cao nhất trên bo mạch được thử ở đây

        sudo jetson_clocks


        tegrastats            # tải theo thời gian thực; nvidia-smi bị hạn chế
        trên Tegra
source_hash: c07ff908503e89b5
---

## Trang này ghi lại điều gì

Trang này ghi lại một cấu hình đã được kiểm chứng từ đầu đến cuối, không phải một
ma trận hỗ trợ. Bo mạch là Jetson Orin Nano Super Developer Kit với 8 GB bộ nhớ,
chạy JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python 3.12.3), và stack
dựng lên được trên đó là `libreyolo 1.4.0` với `torch 2.13.0+cu130`, OpenCV
5.0.0 và NumPy 2.5.1. `torch.cuda.is_available()` trả về `True` và GPU tự báo
tên là `Orin`.

Các bản JetPack khác, các bo mạch Jetson khác và các phiên bản CUDA khác đều
chưa được kiểm thử. Công thức bên dưới là công thức đã chạy được trên tổ hợp đó.

Lần chạy đó diễn ra ngày 2026-07-27 với LibreYOLO 1.4.0, và chưa được lặp lại
trên phần cứng chạy 1.5.0: đây là trang duy nhất trong nhánh 1.5.0 vẫn mang một
kết quả kiểm chứng của 1.4.0, nên front matter của nó ghi `last_verified: "1.4.0"`.
Không có thay đổi nào trong 1.5.0 động tới đường cài đặt, bốn thư viện còn thiếu
hay các cờ xuất mô hình mô tả ở đây, nên các lệnh được kỳ vọng là vẫn đúng,
nhưng số phiên bản trong phần kết quả bên dưới là những gì 1.4.0 in ra, không
phải một phép đo trên 1.5.0.

Có hai điểm ở đây đi ngược lại những gì phần lớn hướng dẫn về Jetson nói. Các
wheel là bản build aarch64 thông thường phát hành cho CUDA 13, nên không cần bản
build torch riêng cho Jetson. Và JetPack không kèm theo bốn thư viện mà các
wheel đó liên kết tới, nên `import torch` lỗi lần lượt từng thư viện một cho tới
khi cài đủ cả bốn.

## Cài đặt

Bản image JetPack không có sẵn pip và cũng không có module `venv`, nên cả hai
được cài trước.

<code-tabs name="prep" />

Bo mạch 8 GB là chật với các checkpoint lớn hơn. Thêm swap trên ổ NVMe trước khi
nạp chúng sẽ tránh bị kill vì hết bộ nhớ giữa chừng.

Tiếp đến là PyTorch. Index CUDA 13 chứa các wheel aarch64; index bổ sung cung
cấp những phụ thuộc thuần Python từ PyPI.

<code-tabs name="torch" />

Bốn wheel `nvidia-*-cu13` là phần dễ bỏ sót. JetPack cung cấp driver GPU, chứ
không cung cấp cuDNN, NCCL, cuSPARSELt hay NVSHMEM, và torch không chịu import
khi thiếu chúng. Cài cả bốn cùng lúc thì nhanh hơn là phát hiện ra chúng qua
từng exception một.

Đoạn mã thứ ba xử lý một lỗi cụ thể: metadata phụ thuộc của torch cho bản build
CUDA 13 đòi `cuda-toolkit==13.0.3`, thứ không có wheel aarch64 trên PyPI, nên
bước giải phụ thuộc thất bại trước khi tải về bất cứ thứ gì. `--no-deps` bỏ qua
bộ giải phụ thuộc, đồng nghĩa mọi phụ thuộc phải được nêu tên trên dòng lệnh.

LibreYOLO được cài sau cùng. Cài nó trước sẽ để pip tự chọn torch của nó, mà
trên nền tảng này thì đó không phải bản build CUDA.

<code-tabs name="install" />

Mọi phụ thuộc còn lại đều giải ra một wheel aarch64 dựng sẵn, gồm cả OpenCV,
NumPy, SciPy, pycocotools và safetensors. Không có gì phải biên dịch từ mã nguồn.

## Kiểm tra CUDA có hoạt động không

<code-tabs name="verify" />

Đoạn mã thứ hai quan trọng ngang đoạn thứ nhất. Một wheel được build cho sai
kiến trúc GPU vẫn báo `torch.cuda.is_available() == True` rồi lỗi ngay ở phép
toán thật đầu tiên với `CUDA error: no kernel image is available for execution
on the device`. Một phép nhân ma trận trên thiết bị là phép kiểm tra bắt được
điều đó.

## Chạy một lần dự đoán

<code-tabs name="predict" />

`predict` trả về cùng đối tượng `Results` như trên mọi nền tảng khác, nên các
trang mô hình vẫn áp dụng nguyên vẹn.

## Xuất sang TensorRT

Trên bo mạch này, TensorRT nhanh hơn cả PyTorch lẫn ONNX Runtime ở toàn bộ 55 mô
hình được đo trên mọi runtime.

<code-tabs name="export" />

`format="tensorrt"` ghi ra một đồ thị ONNX trước rồi build engine từ đó, nên
phải cài extra `onnx`. `LibreYOLO()` điều hướng theo phần mở rộng của tệp, nên
một tệp `.engine` được nạp bằng đúng lời gọi như một checkpoint `.pt`.

Đừng dùng extra pip `tensorrt` trên Jetson. Nó ghim `tensorrt-cu12`, một bản
build CUDA 12, trên một nền tảng CUDA 13. Hãy dùng bản TensorRT do JetPack cài
sẵn. Nếu `import tensorrt` lỗi bên trong môi trường ảo trong khi vẫn chạy được
bên ngoài, hãy tạo lại môi trường với `--system-site-packages` để module hệ
thống nhìn thấy được.

Engine TensorRT đã serialize gắn chặt với thiết bị, kiến trúc GPU và phiên bản
TensorRT đã build ra chúng. Một engine build trên máy trạm sẽ không nạp được
trên Jetson, nên bước build phải chạy ngay trên bo mạch.

## Số đo trên bo mạch này

Độ trễ mỗi ảnh, kích thước batch 1, từ đầu đến cuối gồm cả tiền xử lý và hậu xử
lý, trên COCO val2017 (tập con 500 ảnh) với `conf=0.001` và `max_det=300`. Năm
mô hình trong số 58 mô hình đã đo:

| Mô hình | Đầu vào (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

Cột mAP là điểm của chính lần chạy TensorRT FP16. Trên 55 mô hình được đo ở cả
bốn runtime, khoảng cách lớn nhất giữa điểm PyTorch FP32 và điểm TensorRT FP16
là 0.59 điểm, ở DEIMv2-X. Các runtime khác nhau về tốc độ, không khác nhau về độ
chính xác.

TensorRT FP32 nhanh hơn cả PyTorch lẫn ONNX Runtime ở toàn bộ 55 mô hình đó.
TensorRT FP16 cũng nhanh hơn PyTorch FP32 ở cả 55 mô hình, từ 1.68x đến 6.22x,
với trung vị 3.39x. ONNX Runtime mới là thứ dao động: nó chậm hơn PyTorch ở 23
trong số 55 mô hình, trong đó có dòng RT-DETR-r18.

Điều kiện đằng sau mọi con số: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, đo vào tháng 6 năm
2026. Độ trễ trên Jetson còn phụ thuộc vào chế độ nguồn đang bật, thứ mà các bản
ghi benchmark không lưu lại.

<code-tabs name="power" />

Toàn bộ 223 lần chạy, gồm 53 mô hình còn lại và đầy đủ các cột độ chính xác,
được công bố trên
[trang Jetson Orin tại Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Khắc phục sự cố

### import torch lỗi và nêu tên một thư viện dùng chung

Một trong bốn thư viện ở trên đang thiếu. Thay vì đoán xem là cái nào, hãy đọc
thẳng nó ra từ file nhị phân:

<code-tabs name="ldd" />

Mỗi mục còn thiếu ứng với một wheel:

| Thư viện còn thiếu | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch cảnh báo rằng không có bản build nào hỗ trợ GPU này

Lời gọi CUDA đầu tiên trên cấu hình chạy được sẽ in ra thế này:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

Cảnh báo này chỉ mang tính hình thức trên bo mạch này. Wheel mang theo kernel
`sm_80` và Orin chạy được chúng. Đúng cảnh báo đó cũng xuất hiện với wheel cũ
hơn từ index đó, chính là wheel đã tạo ra mọi dòng benchmark bên trên. Hãy xác
nhận bằng phép nhân ma trận trong phần kiểm tra CUDA thay vì tin hay không tin
thông báo.

### CUDA error: no kernel image is available for execution on the device

Wheel đã cài được build cho một kiến trúc GPU khác. Đây là điều xảy ra với các
wheel từ index `sbsa` của NVIDIA, vốn nhắm tới GPU ARM cho máy chủ chứ không
phải chip Jetson. Hãy cài lại từ index CUDA 13 trong phần cài đặt.

### pip không tìm thấy cuda-toolkit 13.0.3

Không có wheel aarch64 cho nó. Hãy dùng dạng `--no-deps` trong phần cài đặt và
nêu tên các phụ thuộc của torch một cách tường minh.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Wheel torch aarch64 liên kết tới NVIDIA Performance Libraries cho các phép toán
trên CPU. Hãy cài chúng và đưa chúng vào library path:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Index đó dùng tốt cho hai thư viện CPU này. Chính các bản build torch của nó mới
là thứ gây ra lỗi "no kernel image" ở trên.

### Các nguồn wheel không phù hợp với JetPack 7.2

| Nguồn | Kết quả trên Orin Nano Super |
|---|---|
| torch từ `pypi.jetson-ai-lab.io/sbsa/cu130` | Được build cho GPU ARM máy chủ. Import được, báo CUDA khả dụng, rồi lỗi với "no kernel image is available for execution on the device". |
| torch từ `pypi.jetson-ai-lab.io/jp6/*` | Các bản build CUDA 12 và Python 3.10. Chúng không cài được trên Python 3.12 của image này. |
| Container PyTorch của JetPack 6 | Khởi tạo CUDA thất bại với lỗi 801 trên một host chạy JetPack 7. |
| Tự build torch từ mã nguồn | Chạy được, nhưng mất nhiều giờ trên một bo mạch 8 GB và là không cần thiết một khi đã cài các wheel CUDA 13. |

## DeepStream

Nếu cần một pipeline video hoàn chỉnh thay vì một vòng lặp Python, hãy xuất với
`deepstream=True` và chạy đồ thị qua `nvinfer`. Hướng đó có trang riêng, gồm cả
file cấu hình `nvinfer` được sinh ra, cách build bộ phân tích bounding box và
những cái bẫy đã biết: [DeepStream](/docs/export/deepstream).

Bản thân pipeline DeepStream được kiểm chứng trên một GPU rời x86, không phải
trên Jetson. Hợp đồng xuất mô hình không phụ thuộc vào kiến trúc, nhưng lần chạy
pipeline trên aarch64 thì vẫn còn để ngỏ.

## Chưa kiểm chứng

- Các bản JetPack khác 7.2, và các bản L4T khác R39.2.
- Các bo mạch Jetson khác Orin Nano Super 8 GB.
- Huấn luyện trên bo mạch. Suy luận (inference) và xuất mô hình đã được thử; một
  lần chạy huấn luyện thì chưa.
- Engine INT8. Bo mạch này chỉ có các dòng FP32 và FP16.
- Kích thước batch lớn hơn 1. Mọi phép đo bên trên đều là batch 1.
