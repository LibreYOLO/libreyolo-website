---
title: OpenVINO
seo_title: Xuất sang OpenVINO IR từ LibreYOLO
description: >-
  Chuyển đổi một mô hình LibreYOLO sang OpenVINO IR: cặp model.xml và model.bin,
  nén trọng số FP16, INT8 bằng NNCF, và suy luận (inference) trên CPU, GPU hoặc
  NPU.
lead: >-
  OpenVINO IR là định dạng runtime của Intel, một đồ thị model.xml nằm cạnh một
  blob trọng số model.bin. LibreYOLO xuất một bản ONNX trung gian, chuyển đổi nó
  bằng ov.convert_model, rồi ghi một metadata.yaml vào cùng thư mục.
keywords:
  - xuất yolo sang openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - lượng tử hóa int8 nncf
  - chạy openvino trên npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="openvino")
    mono: true
  - label: Kết quả ghi ra
    value: 'Một thư mục chứa model.xml, model.bin và metadata.yaml'
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Hình dạng
    value: 'Theo bản ONNX trung gian: batch động khi dynamic=True'
  - label: Precision
    value: 'FP32, nén trọng số FP16 (half=True), INT8 qua NNCF (int8=True kèm data=)'
verification: >-
  Đọc từ libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py và pyproject.toml
  trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        # IR được chuyển đổi từ một bản ONNX trung gian, nên cần cả hai extra
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 cần thêm NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra thư mục weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Tham số
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True giữ một trục batch động xuyên suốt IR
            half=False,       # True lưu trọng số FP16
            int8=False,       # True chạy lượng tử hóa sau huấn luyện của NNCF
            data=None,        # bắt buộc khi int8=True
            output_path=None, # None ghi ra weights/<stem>_openvino
        )
  int8:
    - label: INT8 với dữ liệu hiệu chuẩn
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # bắt buộc: định dạng này không có giá trị mặc định
            fraction=1.0,
        )
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Chọn thiết bị
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" và "cpu" ánh xạ tới CPU, "gpu" và "cuda" ánh xạ tới GPU,
        # mọi giá trị khác được chuyển tiếp ở dạng viết hoa, ví dụ "npu" -> NPU
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO trần
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Tên các lớp đối tượng, tác vụ và kích thước đầu vào nằm trong
        metadata.yaml cạnh IR

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Trên đường này tiền xử lý và hậu xử lý là việc của bạn
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Cài đặt

<code-tabs name="install" />

Việc chuyển đổi đi qua một bản ONNX trung gian, nên extra `onnx` là một phần bắt
buộc chứ không phải một tùy chọn đi kèm. NNCF là một gói cài riêng và chỉ cần đến
khi `int8=True`.

## Xuất mô hình

<code-tabs name="export" />

Artifact là một thư mục, không phải một tệp. `weights/LibreYOLO9t_openvino` chứa
`model.xml`, `model.bin` và `metadata.yaml`, và `_fp16` được chèn vào trước phần
đuôi khi `half=True`. Hãy di chuyển hoặc sao chép nguyên cả thư mục; ba tệp đó là
một artifact duy nhất.

`half=True` bật `compress_to_fp16` khi lưu. Đó là việc nén trọng số bên trong IR,
không phải thay đổi inference precision mà thiết bị chọn lúc chạy.

### INT8

<code-tabs name="int8" />

`int8=True` chạy lượng tử hóa (quantization) sau huấn luyện của NNCF trên một
loader hiệu chuẩn của LibreYOLO với preset mixed, và `data` là bắt buộc: định dạng
này không có phương án dự phòng tám ảnh. Nếu thiếu NNCF, một `ImportError` sẽ được
ném ra kèm theo tên lệnh cài đặt.

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` nhận ra mọi thư mục có chứa `model.xml` và trả về cùng đối tượng
`Results` như khi dùng checkpoint, đọc tên các lớp đối tượng, tác vụ, kích thước
đầu vào và schema pose từ `metadata.yaml`.

Chuỗi device được ánh xạ chứ không chuyển thẳng qua. `auto` và `cpu` đều biên dịch
cho CPU, `gpu` và `cuda` đều biên dịch cho GPU, còn mọi giá trị khác được viết hoa
rồi đưa cho OpenVINO, và đó là cách nhắm tới một NPU.

Snippet thứ ba dành cho người đọc không cài LibreYOLO. Ở đó tiền xử lý, việc giải
mã, NMS và việc rescale lại tọa độ trở thành việc của bạn, còn tên các lớp đối
tượng chỉ tồn tại trong `metadata.yaml`.

## Giới hạn

Một IR không có `metadata.yaml` vẫn tải được, nhưng khi đó backend sẽ quay về mặc
định 80 lớp đối tượng và tác vụ phát hiện đối tượng, điều này là sai với mọi thứ
khác. Hãy giữ nguyên vẹn thư mục.

Bị chặn trước khi trace: phân đoạn YOLO9, phân đoạn RTMDet-Ins, phát hiện đối tượng
với SSD, Faster R-CNN và RetinaNet, cùng matting BiRefNet hoặc FeyNobg, nơi
OpenVINO 2026.2 không hạ được phép toán ONNX chuẩn `DeformConv-19` của bộ giải mã
matte dùng chung.

Khi một tổ hợp không nằm trong nhóm đã kiểm chứng mà cũng không nằm trong nhóm bị
chặn, đường chuyển đổi vẫn khả dụng và dự án chưa ghi nhận việc đối chiếu runtime
OpenVINO cho tổ hợp đó. Một số tổ hợp được kiểm chứng kèm theo một ngữ cảnh cụ thể,
ví dụ phân đoạn ngữ nghĩa DeepLabV3 ở đầu vào cố định 520 nhân 520 trên OpenVINO
2026.2 với inference precision mặc định của CPU, và gaze L2CS ở ảnh cắt khuôn mặt
cố định 448 nhân 448. `libreyolo formats` in ra ngữ cảnh đó cho từng tổ hợp.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
