---
title: TensorRT
seo_title: Xuất sang TensorRT từ LibreYOLO
description: >-
  Build một engine TensorRT từ mô hình LibreYOLO: bước trung gian ONNX, build
  FP16 và INT8, optimization profile cho batch động, và giới hạn về tính khả
  chuyển của engine.
lead: >-
  TensorRT biên dịch một đồ thị thành một engine được tối ưu cho đúng một GPU.
  LibreYOLO xuất một bước trung gian ONNX trước, parse nó bằng bộ parser ONNX
  của TensorRT, build engine, rồi ghi metadata của mô hình ra ngay bên cạnh dưới
  dạng một sidecar JSON.
keywords:
  - xuất yolo sang tensorrt
  - engine tensorrt
  - trt fp16
  - hiệu chuẩn int8 tensorrt
  - optimization profile
  - batch động tensorrt
  - hardware compatibility level tensorrt
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tensorrt")
    mono: true
  - label: Kết quả ghi ra
    value: Một tệp .engine cùng một sidecar metadata .engine.json
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Hình dạng
    value: >-
      Mặc định là cố định; dynamic=True thêm một optimization profile theo trục
      batch
  - label: Precision
    value: 'FP32, FP16 (half=True), INT8 (int8=True kèm data=)'
  - label: Yêu cầu
    value: >-
      Một GPU NVIDIA lúc build và lúc chạy. Engine không dùng lại được giữa các
      kiến trúc GPU.
verification: >-
  Đọc từ libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py và pyproject.toml
  trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        # Engine được build từ một bước trung gian ONNX, nên cần cả hai extra
        pip install "libreyolo[onnx,tensorrt]"
    - label: Xác nhận toolchain trước khi build
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Ghi ra weights/LibreYOLO9t_fp16.engine và
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Tham số
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # bắt buộc khi int8=True
            dynamic=False,
            workspace=4.0,                  # GiB bộ nhớ tạm lúc build
            min_batch=1,                    # giới hạn của profile động
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # hoặc "ampere_plus"
            gpu_device=0,                   # thiết bị build trên máy nhiều GPU
            verbose=False,
        )
  dynamic:
    - label: Engine với batch động
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Bước trung gian ONNX cần có trục batch động thì profile
        # mới có thứ để gắn vào
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 với dữ liệu hiệu chuẩn
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # bắt buộc: định dạng này không có giá trị mặc định
            fraction=1.0,
        )
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT trần
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Tên các lớp đối tượng, tác vụ và kích thước đầu vào nằm trong sidecar,
        không nằm trong engine

        # Việc cấp phát buffer, tiền xử lý và hậu xử lý ở đây là việc của bạn

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi build
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Cài đặt

Cả bước build lẫn bước chạy đều cần một GPU NVIDIA với stack CUDA hoạt động được.
Định dạng này không có đường lui về CPU.

<code-tabs name="install" />

Extra `tensorrt` ghim `tensorrt-cu12` và `pycuda`, và marker loại bỏ cả hai trên
macOS. Trên Jetson, đừng dùng extra đó: nó ghim một bản build CUDA 12 lên một nền
tảng CUDA 13. Hãy dùng bản TensorRT do JetPack cài đặt, như mô tả ở
[NVIDIA Jetson](/docs/export/jetson).

## Xuất mô hình

<code-tabs name="export" />

Việc xuất chạy theo hai bước. Bước một ghi một bước trung gian ONNX ra một đường
dẫn tạm, bước hai parse nó và build engine, rồi bản trung gian bị xóa sau đó.
`workspace` là bộ nhớ tạm lúc build tính bằng GiB; giá trị lớn hơn cho phép builder
thử nhiều kernel hơn và không ảnh hưởng tới bộ nhớ khi suy luận (inference).

Sidecar metadata được ghi ngay cạnh engine với tên `<engine>.json` và ghi lại
precision mà bản build thực sự đạt được. Khi GPU không có FP16 nhanh hoặc INT8
nhanh, builder cảnh báo và lui xuống mức thấp hơn, còn sidecar báo cáo precision
thực tế thu được thay vì precision đã được yêu cầu.

Ở chế độ FP16, một backbone ViT nằm trong đồ thị sẽ được phát hiện và các lớp float
của nó bị ghim về FP32. Các backbone kiểu DINOv2 tràn số trong FP16 và cho ra NaN,
nên bản build đặt `OBEY_PRECISION_CONSTRAINTS` và báo `FP16 (FP32 ViT backbone)`.
Bước này không làm gì trên các backbone CNN.

### Batch động

<code-tabs name="dynamic" />

`dynamic=True` thêm một optimization profile trải từ `min_batch` tới `max_batch`,
tối ưu tại `opt_batch`, và ghi ba giá trị đó vào sidecar. Profile chỉ được thêm khi
bước trung gian ONNX thực sự mang một chiều batch động; nếu không, bản build ghi log
rằng nó đang dùng tối ưu tĩnh và tiếp tục chạy.

### INT8

<code-tabs name="int8" />

INT8 dùng entropy calibrator của TensorRT trên một loader hiệu chuẩn của LibreYOLO,
và `data` là bắt buộc: định dạng này không có đường lui tám ảnh. Việc hiệu chuẩn cần
`cuda-python` hoặc `pycuda` cho buffer trên thiết bị. Cache hiệu chuẩn được đánh khóa
theo hash của các byte ONNX, nên scale của mô hình này không bao giờ bị dùng lại cho
một mô hình khác tình cờ ghi ra cùng đường dẫn đầu ra.

`half=True` và `int8=True` đặt cùng nhau sẽ cảnh báo và build INT8, vốn vẫn giữ một
đường lui FP16 cho các lớp mà TensorRT không thể lượng tử hóa (quantization).

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` điều hướng dựa trên phần đuôi `.engine`, đọc sidecar để lấy tên các lớp
đối tượng, tác vụ và schema pose, rồi trả về cùng đối tượng `Results` như khi dùng
checkpoint. Nó ném lỗi ngay lập tức khi không có thiết bị CUDA nào.

Snippet thứ hai là đường chạy runtime trần. Việc cấp phát buffer trên host và trên
thiết bị, tiền xử lý, giải mã, NMS và việc rescale lại tọa độ đều trở thành việc của
bạn, còn bản thân engine không mang tên các lớp đối tượng, nên sidecar phải đi kèm
theo nó.

## Giới hạn

Một engine đã serialize gắn chặt với kiến trúc GPU, stack driver và phiên bản
TensorRT đã build ra nó. Một engine build trên máy trạm sẽ không tải được trên một
kiến trúc khác, và đó là lý do bước build chạy ngay trên máy triển khai.
`hardware_compatibility="ampere_plus"` đánh đổi một phần hiệu năng để lấy tính khả
chuyển trên Ampere và các kiến trúc mới hơn. Giá trị `"same_compute_capability"` ánh
xạ về `NONE` và kèm cảnh báo: engine chỉ được tối ưu cho GPU hiện tại, và bản xuất
nói đúng như vậy thay vì tuyên bố một tính khả chuyển mà nó không hề áp dụng.

Chỉ trục batch được đưa vào profile. Một bản build với các chiều không gian động
không nằm trong hợp đồng này, và đó là lý do FCOS bị chặn: nó cần chiều cao và chiều
rộng đã pad ở dạng động để giữ nguyên phép biến đổi tỉ lệ 800 nhân 1333 của mình.

Bị chặn trước khi trace: phân đoạn với YOLO9, phân đoạn với RTMDet-Ins, phát hiện
đối tượng với SSD, Faster R-CNN và RetinaNet, cùng matting với BiRefNet hoặc FeyNobg,
những trường hợp mà TensorRT 10.16 chạm tới node ONNX `DeformConv` dùng chung và không
parse được nó vì `ModulatedDeformConv2d` vắng mặt trong plugin registry.

Ở những tổ hợp không được kiểm chứng mà cũng không bị chặn, đường chuyển đổi vẫn khả
dụng và dự án chưa ghi nhận việc đối chiếu runtime TensorRT cho tổ hợp đó. Đó là phát
biểu về bằng chứng, không phải về việc bản build có thành công hay không.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
