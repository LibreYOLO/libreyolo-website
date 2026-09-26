---
title: Core ML
seo_title: Xuất sang Core ML từ LibreYOLO
description: >-
  Xuất một mô hình phát hiện đối tượng LibreYOLO sang .mlpackage của Core ML:
  hợp đồng đầu vào ImageType, FP16, compute units, NMS nhúng sẵn và bốn họ mô
  hình được hỗ trợ.
lead: >-
  Core ML là định dạng mô hình chạy trên thiết bị của Apple. LibreYOLO trace mô
  hình phát hiện đối tượng phía sau một wrapper tiền xử lý riêng cho từng họ,
  nên đồ thị sau khi chuyển đổi luôn nhận một đầu vào ảnh RGB chuẩn, rồi ghi ra
  một .mlpackage ở định dạng ML Program kèm theo metadata của mô hình.
keywords:
  - xuất yolo sang coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - nms nhúng sẵn trong coreml
  - chạy yolo trên ios
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreml")
    mono: true
  - label: Kết quả ghi ra
    value: Một bundle .mlpackage (là một thư mục) ở định dạng ML Program
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") trên macOS
    mono: true
  - label: Hình dạng
    value: Cố định. Đầu vào là một ct.ImageType có hình dạng cứng.
  - label: Precision
    value: 'FP32, FP16 (half=True). Không có INT8.'
  - label: Họ mô hình
    value: 'Chỉ phát hiện đối tượng, cho yolox, yolo9, rtdetr và rfdetr'
verification: >-
  Đọc từ libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py và pyproject.toml
  trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Tham số
      language: python
      code: >
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True chuyển đổi với compute precision FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None ghi ra weights/<stem>.mlpackage
        )


        # dynamic được chấp nhận nhưng đầu vào là ct.ImageType có hình dạng cố
        định,

        # và metadata nhúng kèm vẫn ghi dynamic=False trong mọi trường hợp
  nms:
    - label: Nhúng lớp NMS của Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Chỉ phát hiện đối tượng với YOLOX và YOLO9, batch 1
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Qua LibreYOLO, trên macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # hoặc cpu_and_ne để ghim vào Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Dùng coremltools trực tiếp
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # Đầu vào là một ảnh tên "image" ở đúng kích thước đã xuất
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # Trên đường này letterboxing và hậu xử lý là việc của bạn
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Cài đặt

<code-tabs name="install" />

Dự đoán cần macOS. `LibreYOLO()` từ chối một `.mlpackage` trên mọi nền tảng khác,
kèm thông báo nêu tên nền tảng hiện tại, và ma trận hỗ trợ ghi nhận các tổ hợp này
là khả dụng với lý do việc đối chiếu runtime cần một máy chạy macOS.

## Xuất mô hình

<code-tabs name="export" />

Bundle được ghi vào `weights/` theo stem của checkpoint, thêm `_fp16` vào sau khi
`half=True`. Một `.mlpackage` là một thư mục, nên hãy sao chép nguyên cả cây thư mục.

Mọi họ mô hình đều được trace phía sau một wrapper tiền xử lý, nên đồ thị sau khi
chuyển đổi chỉ nhận một đầu vào chuẩn duy nhất: RGB, `scale=1/255`, không có bias,
khai báo dưới dạng `ct.ImageType`. Wrapper hấp thụ quy ước riêng của từng họ, đó là
BGR trong khoảng 0 đến 255 với YOLOX, mean và standard deviation của ImageNet với
RF-DETR, và identity với YOLO9 cùng RT-DETR. Đó là lý do một bên tiêu thụ Core ML
đưa vào một ảnh thông thường thay vì một tensor riêng của từng họ.

Việc chuyển đổi nhắm tới ML Program với deployment target tối thiểu là iOS 15.
`compute_units` được lưu trên mô hình đã chuyển đổi và có thể ghi đè lại lần nữa khi
artifact được tải.

Metadata của mô hình đi vào `user_defined_metadata` dưới dạng chuỗi, và đó là nơi
backend đọc ra họ mô hình, tác vụ, tên các lớp đối tượng, kích thước đầu vào và schema pose.

### NMS nhúng sẵn

<code-tabs name="nms" />

`nms=True` bọc mô hình trong một pipeline Core ML kết thúc bằng lớp
`NonMaximumSuppression` của Apple. Kết quả có hai đầu ra: `confidence`, hình dạng
`N` nhân với số lớp đối tượng, và `coordinates`, hình dạng `N` nhân 4 theo `xywh` đã chuẩn hóa.

Nó chỉ áp dụng cho phát hiện đối tượng với YOLOX và YOLO9, và đòi hỏi batch 1. Các họ
theo kiểu DETR bị từ chối theo tên, vì dự đoán theo tập hợp lấy top-k trên các query và
các lớp đối tượng mà không có bước IoU nào và không dùng được lớp đó. `max_det` cũng
không được phơi ra ở đây; khi số lượng detection tối đa là điều quan trọng, hãy dùng
[NMS nhúng sẵn của ONNX](/docs/export/onnx) thay thế.

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` nhận ra một thư mục có phần đuôi `.mlpackage` và trả về cùng đối tượng
`Results` như khi dùng checkpoint. `compute_units` là đối số duy nhất mà factory chuyển
tiếp cho định dạng này, và nó chấp nhận `all`, `cpu_and_gpu`, `cpu_and_ne` và
`cpu_only`. Đối số `device` bị bỏ qua, vì thay vào đó Core ML định tuyến qua compute units.

Snippet thứ hai là đường chạy runtime trần. Ở đó letterboxing, việc giải mã, NMS và
việc rescale lại tọa độ trở thành việc của bạn, còn tên các lớp đối tượng nằm trong
`user_defined_metadata`.

## Giới hạn

Bốn họ mô hình, chỉ phát hiện đối tượng: `yolox`, `yolo9`, `rtdetr` và `rfdetr`. Bất cứ
thứ gì khác đều bị từ chối ở bước preflight, vì chính wrapper tiền xử lý theo từng họ là
thứ làm cho hợp đồng đầu vào ảnh cố định trở nên đúng đắn, và một họ nằm ngoài đó sẽ
chuyển đổi với phép chuẩn hóa sai. Thông báo lỗi nêu tên ONNX và TorchScript là các
phương án thay thế.

Hình dạng đầu vào bị cố định cứng bởi `ct.ImageType`, nên `dynamic=True` không thay đổi
gì và metadata ghi `dynamic=False`. Hãy xuất một bundle thứ hai cho một độ phân giải thứ hai.

`half=True` chuyển đổi với compute precision FP16. Không có đường INT8 nào từ exporter này.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với định dạng chạy trên thiết bị
mới hơn của Apple, xem [Core AI](/docs/export/coreai). Để kiểm tra một tổ hợp:

<code-tabs name="support" />
