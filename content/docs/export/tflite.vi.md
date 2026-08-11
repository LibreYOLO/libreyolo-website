---
title: TFLite
seo_title: Xuất sang TFLite (LiteRT) từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO sang FlatBuffer .tflite thông qua onnx2tf: shape
  tĩnh, chỉ FP32, đầu vào NHWC, và những họ mô hình chuyển đổi trót lọt.
lead: >-
  TFLite là định dạng FlatBuffer mà LiteRT thực thi trên các đích di động và
  nhúng. LibreYOLO xuất ra một đồ thị ONNX tĩnh, chuyển đổi nó bằng onnx2tf ở
  chế độ flatbuffer-direct, rồi ghi metadata của mô hình bên cạnh artifact dưới
  dạng một tệp JSON sidecar.
keywords:
  - xuất yolo sang tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - đầu vào nhwc tflite
  - suy luận trên thiết bị biên
last_verified: 1.5.0
meta:
  - label: Cờ
    value: export(format="tflite")
    mono: true
  - label: Kết quả ghi ra
    value: Một tệp .tflite cùng một tệp metadata sidecar .tflite.json
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Shape
    value: Chỉ shape tĩnh. dynamic=True bị từ chối.
  - label: Precision
    value: Chỉ FP32. half=True và int8=True bị từ chối.
  - label: Yêu cầu
    value: >-
      Python 3.12 trở lên, vì onnx2tf 2.4.x không phát hành wheel nào cho phiên
      bản cũ hơn
verification: >-
  Đọc từ libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py và pyproject.toml
  trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        # LiteRT là tên hiện tại của Google cho TensorFlow Lite. Cả hai extra
        # đều cài cùng một toolchain và cho ra cùng một tệp .tflite
        pip install "libreyolo[tflite]"
    - label: Xác nhận phiên bản Python trước
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.tflite và weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" được chấp nhận như một bí danh và trỏ tới cùng exporter
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Tham số
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, hoặc (chiều cao, chiều rộng)
            batch=1,
            simplify=True,    # chạy onnxsim trên đồ thị ONNX trung gian
            output_path=None, # None sẽ ghi ra weights/<stem>.tflite
            verbose=False,    # True sẽ in luồng log của onnx2tf
        )

        # dynamic=True ném ValueError: bộ chuyển đổi cần shape tĩnh
        # half=True và int8=True bị từ chối trước khi trace
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT thuần
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, không phải NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Tên các lớp đối tượng, tác vụ và kích thước đầu vào nằm trong tệp
        sidecar

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Tiền xử lý, phép chuyển vị NCHW sang NHWC và hậu xử lý là việc của bạn
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Cài đặt

<code-tabs name="install" />

Extra này kéo về `onnx2tf` cho việc chuyển đổi và `ai-edge-litert` để chạy kết
quả, cả hai đều nằm sau một marker Python 3.12. Trên một interpreter cũ hơn, lệnh
xuất sẽ ném `ImportError` nêu đích danh yêu cầu về phiên bản thay vì thất bại bên
trong bộ chuyển đổi.

`libreyolo[litert]` cài đúng cùng một thứ. Chuỗi định dạng `litert` là bí danh của
`tflite`, và tệp đầu ra là `.tflite` trong cả hai trường hợp.

## Xuất mô hình

<code-tabs name="export" />

Họ mô hình và tác vụ được kiểm tra trước khi bất cứ điều gì khác diễn ra, nên một
tổ hợp không được hỗ trợ sẽ thất bại ngay lập tức kèm đúng lỗi cụ thể của bộ
chuyển đổi hoặc của runtime đã loại nó ra, chứ không phải một thông báo chung
chung. Bản thân việc chuyển đổi là một lời gọi subprocess tới `onnx2tf` ở chế độ
`flatbuffer_direct` trên một đồ thị ONNX trung gian tĩnh.

Metadata nằm ở một tệp sidecar. `weights/LibreYOLO9t.tflite.json` mang theo họ mô
hình, tác vụ, tên các lớp đối tượng, kích thước đầu vào và schema pose; bản thân
FlatBuffer không có trường metadata nào của LibreYOLO, nên hai tệp luôn đi cùng
nhau.

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` nhận diện theo phần mở rộng `.tflite` và trả về cùng đối tượng
`Results` như khi dùng checkpoint. Backend đọc tệp sidecar, chuyển vị blob NCHW
sang NHWC khi interpreter yêu cầu một đầu vào channels-last, áp dụng scale và zero
point của lượng tử hóa (quantization) từ interpreter ở những nơi có sẵn, rồi
chuyển vị các đầu ra trở lại bố cục mà bước hậu xử lý của LibreYOLO mong đợi.

Đoạn mã thứ hai là hướng dùng runtime thuần. Ở đó, tiền xử lý, phép chuyển vị bố
cục, việc giải mã, NMS và quy đổi lại tọa độ đều trở thành việc của bạn, và chi
tiết về bố cục là thứ dễ bị bỏ sót nhất: onnx2tf sinh ra đầu vào channels-last,
nên một blob có shape `(1, 3, 640, 640)` sẽ không gắn được.

## Ràng buộc

Chỉ shape tĩnh. `dynamic=True` ném `ValueError` trước khi trace, và khung xuất
(canvas) bị cố định ở đúng giá trị mà `imgsz` phân giải ra.

Chỉ FP32. `half=True` và `int8=True` đều bị từ chối trong bước kiểm tra, nên việc
triển khai ở dạng đã lượng tử hóa hiện chưa thể đạt tới từ exporter này.

Phạm vi hỗ trợ ở đây hẹp hơn so với các định dạng đồ thị, và nó được quyết định
bằng đo đạc chứ không phải theo họ mô hình. Các tổ hợp đã được kiểm chứng gồm phát
hiện đối tượng với YOLO9, YOLOX và YOLO-NAS, phân đoạn ngữ nghĩa với PIDNet, bốn
họ phân loại CNN, embedding với DINOv2 và SigLIP2, phân loại với SigLIP2, phát
hiện biên với TEED và DexiNed, và phục hồi ảnh với Real-ESRGAN và SwinIR. SwinIR
mang thêm một lưu ý: độ khớp chỉ giữ được khi kích thước ảnh nguồn trùng khớp
chính xác với canvas xuất, còn ảnh nguồn nhỏ hơn sẽ được đệm cho vừa canvas trước
khi transformer chạy, điều này có thể sai lệch so với suy luận (inference) ở kích
thước thay đổi tự do.

Các mục bị chặn nêu đích danh lỗi cụ thể, và đó là thứ đáng đọc trước khi thử tìm
cách đi vòng. Vài ví dụ: phát hiện đối tượng với RF-DETR chuyển đổi được ở canvas
gốc 384 nhưng LiteRT không cấp phát được cho nó vì `STRIDED_SLICE` nhận một đầu
vào vượt quá rank 5 chiều mà nó hỗ trợ; PicoDet bị từ chối vì một `RESHAPE` ánh xạ
19.200 phần tử đầu vào thành 9.600 phần tử đầu ra; D-FINE làm bộ chuyển đổi sập ở
khâu xử lý shape của `GatherElements`; RTMDet xuất và tải lại được với độ khớp thô
còn nguyên vẹn nhưng box công khai tụt xuống 0.911 IoU với 29.9 px độ lệch tọa độ.

Để xem lưới đầy đủ theo họ mô hình và tác vụ, hãy xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể, bao
gồm cả chuỗi lý do đằng sau một mục bị chặn:

<code-tabs name="support" />
