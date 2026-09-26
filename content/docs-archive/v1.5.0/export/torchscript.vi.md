---
title: TorchScript
seo_title: Xuất sang TorchScript từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO sang TorchScript: một archive .torchscript được
  trace, mang sẵn metadata của LibreYOLO bên trong, nạp được từ Python hoặc
  libtorch.
lead: >-
  TorchScript là định dạng đồ thị tuần tự hóa của chính PyTorch. LibreYOLO trace
  mô hình bằng torch.jit.trace và lưu kết quả kèm theo một tệp extra
  libreyolo_metadata.json, nên archive mang theo họ mô hình, tác vụ, tên các lớp
  đối tượng và kích thước đầu vào.
keywords:
  - xuất yolo sang torchscript
  - torch.jit.trace
  - torch.jit.load
  - triển khai libtorch
  - metadata torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Kết quả ghi ra
    value: Một archive .torchscript kèm một tệp extra libreyolo_metadata.json
  - label: Extra
    value: Không cần gì thêm. TorchScript đi kèm PyTorch.
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Hình dạng
    value: Cố định. Đồ thị được trace ở một hình dạng đầu vào duy nhất.
  - label: Precision
    value: 'FP32, FP16 (half=True). Không có INT8.'
verification: >-
  Đọc từ libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py và libreyolo/backends/torchscript.py trên nhánh
  dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Tham số
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int, hoặc (cao, rộng)
            batch=1,
            half=False,       # trọng số và activations ở FP16
            device=None,      # None sẽ trace trên CPU với định dạng này
            output_path=None, # None ghi ra weights/<stem>.torchscript
        )


        # dynamic được chấp nhận nhưng archive luôn là một trace hình dạng cố
        định,

        # và metadata nhúng bên trong đều ghi dynamic=False trong mọi trường hợp
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch thuần
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # Ở nhánh này, tiền xử lý và hậu xử lý là việc của bạn
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Cài đặt

<code-tabs name="install" />

TorchScript không cần gì ngoài bản cài đặt cơ bản, vì `torch.jit` đi kèm với
PyTorch. Đây là đích xuất mô hình duy nhất không có phụ thuộc tùy chọn và không
cần bộ chuyển đổi bên ngoài, nên nó là phép thử đầu tiên hữu ích khi một
toolchain dài hơn gặp lỗi.

## Xuất mô hình

<code-tabs name="export" />

Quá trình trace chạy trên CPU trừ khi có chỉ định device, và archive được ghi vào
`weights/` theo stem của checkpoint khi bỏ qua `output_path`.

Phép kiểm tra trace lại mà `torch.jit.trace` thường thực hiện đã bị tắt. Một vài
wrapper xuất mô hình lưu cache các anchor phụ thuộc hình dạng trong lần forward
đầu tiên, nên lần trace thứ hai quan sát thấy một đường đi Python khác dù đồ thị
hình dạng cố định đã ghi lại vẫn đúng. Thay vào đó, các bài kiểm tra parity xác
thực trực tiếp module đã lưu.

Metadata không nằm trong một tệp sidecar. `torch.jit.save` lưu
`libreyolo_metadata.json` bên trong archive, và `torch.jit.load` trả nó lại qua
`_extra_files`.

## Chạy tệp đã xuất

<code-tabs name="run" />

`LibreYOLO()` định tuyến theo phần mở rộng `.torchscript` và trả về cùng đối tượng
`Results` như checkpoint mà nó xuất phát. Với `device="auto"`, module được map sang
CUDA khi có, rồi đến MPS, rồi CPU.

Đoạn mã thứ hai là hướng đi dành cho người đọc không cài LibreYOLO, và cho việc
triển khai C++ qua libtorch, nơi cùng archive đó nạp được bằng
`torch::jit::load`. Ở đó, tiền xử lý, giải mã, NMS và co giãn lại tọa độ trở thành
việc của bạn. Tệp extra chứa metadata vẫn đọc được, và đó là nơi duy nhất tồn tại
tên các lớp đối tượng.

## Ràng buộc

Đồ thị là một trace ở một hình dạng đầu vào duy nhất. `dynamic=True` được chấp
nhận để giữ đối xứng về giao diện nhưng không thay đổi điều gì, và metadata nhúng
bên trong báo `dynamic=False` để một backend không bao giờ giả định một trục mà nó
không dùng được. Hãy xuất một archive thứ hai cho độ phân giải thứ hai.

`half=True` ép mô hình và đầu vào của lần trace sang FP16. Không có nhánh INT8:
`int8=True` ném `NotImplementedError` trong lúc kiểm tra hợp lệ.

`imgsz` hình chữ nhật hoạt động với các họ YOLO9, HRNet, NAFNet và Real-ESRGAN,
và bị từ chối với những họ mô hình có hợp đồng vuông cố định.

Năm tổ hợp bị từ chối trước khi trace. Phân đoạn YOLO9, vì trong LibreYOLO thì
YOLO9 chỉ làm phát hiện đối tượng. Phân đoạn RTMDet-Ins, vốn có phần giải mã mặt
nạ (mask) bằng kernel động không có hợp đồng nào cho runtime đã xuất. Phát hiện
đối tượng với SSD, Faster R-CNN và RetinaNet, vốn có đồ thị độ dài thay đổi hoặc
anchor động chỉ có bằng chứng parity qua hợp đồng của ONNX Runtime.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
