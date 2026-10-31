---
title: ExecuTorch
seo_title: Xuất mô hình sang ExecuTorch từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO sang chương trình .pte của ExecuTorch với
  delegation XNNPACK: shape cố định, batch 1, FP32, và tệp metadata sidecar mà
  nó cần.
lead: >-
  ExecuTorch chạy các chương trình PyTorch trên thiết bị biên (edge device).
  LibreYOLO capture mô hình bằng torch.export ở chế độ strict, hạ xuống XNNPACK,
  rồi ghi chương trình .pte cùng một tệp metadata JSON sidecar như một đơn vị
  duy nhất.
keywords:
  - xuất yolo sang executorch
  - chương trình .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - chạy pytorch trên thiết bị biên
last_verified: 1.5.0
meta:
  - label: Cờ
    value: export(format="executorch")
    mono: true
  - label: Kết quả ghi ra
    value: Một chương trình .pte cùng một tệp metadata sidecar .pte.json
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Shape
    value: Cố định. dynamic=True và batch != 1 bị từ chối.
  - label: Precision
    value: Chỉ FP32. half=True và int8=True bị từ chối.
  - label: Delegate
    value: 'XNNPACK, CPU. delegate=''xnnpack'' là giá trị duy nhất được chấp nhận.'
verification: >-
  Đọc từ libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py và
  pyproject.toml trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        # Cố ý để ngoài libreyolo[all]: ExecuTorch giới hạn phiên bản
        # Torch mà nó có thể đi cùng
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.pte và weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Tham số
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, hoặc (chiều cao, chiều rộng)
            batch=1,               # giá trị khác sẽ ném ValueError
            dynamic=False,         # True sẽ ném ValueError
            delegate="xnnpack",    # giá trị duy nhất được chấp nhận
            device="cpu",          # thiết bị khác sẽ ném ValueError
            output_path=None,      # None sẽ ghi ra weights/<stem>.pte
        )
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Runtime ExecuTorch thuần
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # Trên hướng này, tiền xử lý và hậu xử lý là việc của bạn

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Cài đặt

<code-tabs name="install" />

Extra này cố ý nằm ngoài `libreyolo[all]`, vì ExecuTorch ghim phiên bản Torch mà
nó làm việc cùng và việc cài nó sẽ kéo toàn bộ môi trường theo cặp phiên bản đó.
Hãy cài vào một môi trường mà bạn chấp nhận bị ràng buộc.

Trên Windows, bước lowering gọi tệp thực thi `flatc` đi kèm ExecuTorch. Nếu nó
không nằm trong `PATH`, lệnh xuất sẽ ném `RuntimeError` báo đúng điều đó, và
cách khắc phục là chạy từ Visual Studio 2022 Developer PowerShell.

## Xuất mô hình

<code-tabs name="export" />

Việc capture dùng `torch.export.export(..., strict=True)`, tức là một lần capture
đồ thị thật sự có kèm guard chứ không phải một bản trace được ghi lại. Các thao
tác đọc scalar trên host và luồng điều khiển phụ thuộc dữ liệu bị từ chối thay vì
bị gắn cứng vào đồ thị một cách âm thầm, nên vài họ mô hình thất bại ở đây dù
trace thành công ở nơi khác; lý do được ghi lại theo từng tổ hợp trong ma trận hỗ
trợ.

Bước lowering chạy `to_edge_transform_and_lower` với XNNPACK partitioner. Nếu kết
quả không chứa phân vùng delegate nào, lệnh xuất sẽ báo lỗi thay vì gán nhãn
XNNPACK cho một chương trình chỉ dùng portable kernel.

Chương trình và tệp sidecar được commit cùng nhau. Cả hai đều được chuẩn bị ở
vùng tạm, cả hai đều được thay vào, và khi có lỗi thì mọi thứ quay lại đúng trạng
thái trước đó, nên một cặp tệp dở dang không bao giờ nằm lại trên đĩa.

## Chạy tệp đã xuất

<code-tabs name="run" />

`LibreYOLO()` nhận diện theo phần mở rộng `.pte` và trả về cùng đối tượng
`Results` như khi dùng checkpoint. Tệp sidecar là bắt buộc khi tải: thiếu
`<program>.pte.json` thì backend sẽ ném `FileNotFoundError`, vì bản thân chương
trình không mang theo tên lớp đối tượng, tác vụ hay kích thước đầu vào nào.
Backend cũng kiểm tra runtime đã cài có cung cấp `XnnpackBackend` hay không trước
khi tải, và đọc chương trình từ bytes thay vì map tệp vào bộ nhớ, nhờ đó tránh
việc giữ khóa tệp trên Windows suốt vòng đời của backend.

Đoạn mã thứ hai là hướng dùng runtime thuần. Ở đó, tiền xử lý, giải mã đầu ra,
NMS và việc quy đổi lại tọa độ trở thành việc của bạn.

## Ràng buộc

Batch 1, shape cố định, FP32, CPU. `batch != 1` và `dynamic=True` đều ném
`ValueError` trước khi lệnh xuất thay đổi bất cứ thứ gì, `half=True` và
`int8=True` bị từ chối trong bước kiểm tra, còn thiết bị khác CPU thì bị khước
từ.

`delegate` chấp nhận `"xnnpack"` và không nhận gì khác trong phiên bản này.

Các bản xuất cho tác vụ phân loại mang thêm hai khóa metadata, `crop_pct` và
`interpolation`, để runtime tái tạo được chính sách resize và center-crop của họ
mô hình đó.

Các mục bị chặn nêu đích danh lỗi cụ thể chứ không chỉ ghi một hạng mục chung.
D-FINE ở tác vụ phát hiện đối tượng và phân đoạn chạm phải một thao tác đọc
`ContextVar` không được hỗ trợ trong deformable attention khi capture ở chế độ
strict, còn nếu ép dùng đường grid-sample thủ công thì mô hình serialize được
nhưng sau đó lỗi lúc chạy vì thứ tự chiều của tensor được delegate không hợp lệ.
DEIM và DEIMv2 capture, lowering và serialize đều trót lọt, rồi lỗi trong lúc
thực thi. EoMT ở tác vụ phân đoạn ngữ nghĩa lỗi vì một biểu thức ký hiệu phụ
thuộc dữ liệu trong nhánh xử lý mặt nạ (mask). BiRefNet matting capture được ở
1024 nhân 1024 nhưng không có biến thể out cho `torchvision::deform_conv2d`.
SwinIR phục hồi ảnh tải lại được rồi lỗi ở `aten::alias_copy.out` do thứ tự chiều
không khớp.

Để xem lưới đầy đủ theo họ mô hình và tác vụ, hãy xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
