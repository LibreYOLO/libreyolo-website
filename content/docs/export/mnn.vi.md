---
title: MNN
seo_title: Xuất sang MNN từ LibreYOLO
description: >-
  Xuất một bộ phát hiện đối tượng LibreYOLO sang MNN qua ONNX và mnnconvert:
  hình dạng (shape) NCHW cố định, FP32 trên CPU, và một tệp sidecar metadata mà
  hợp đồng runtime đòi hỏi.
lead: >-
  MNN là công cụ suy luận (inference) nhẹ của Alibaba. LibreYOLO xuất một graph
  ONNX tĩnh, chuyển đổi nó bằng công cụ mnnconvert đi kèm gói MNN, rồi ghi ra
  một tệp sidecar JSON ghi lại tên đầu vào và đầu ra, shape đầu vào cố định và
  tên các lớp đối tượng.
keywords:
  - xuất yolo sang mnn
  - mnnconvert
  - chạy inference với mnn
  - phát hiện đối tượng trên mobile
  - shape nchw cố định
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="mnn")
    mono: true
  - label: Kết quả ghi ra
    value: Một tệp .mnn cùng một tệp sidecar metadata .mnn.json
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Hình dạng
    value: NCHW cố định. dynamic=True bị từ chối.
  - label: Precision
    value: 'Chỉ FP32, chỉ CPU.'
  - label: Tác vụ
    value: Chỉ phát hiện đối tượng trong phiên bản này
verification: >-
  Đọc từ libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py và pyproject.toml trên
  nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: >
        # Extra này đã bao gồm libreyolo[onnx]: MNN chuyển đổi từ một bản trung
        gian ONNX

        pip install "libreyolo[mnn]"
    - label: Xác nhận bộ chuyển đổi có trên PATH
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.mnn và weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Tham số
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, hoặc (chiều cao, chiều rộng)
            batch=1,          # được cố định vào artifact
            simplify=True,    # chạy onnxsim trên bản trung gian ONNX
            output_path=None, # None sẽ ghi ra weights/<stem>.mnn
            verbose=False,    # True sẽ in luồng log của mnnconvert
        )

        # dynamic=True gây ValueError. half=True và int8=True bị từ chối
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN thuần
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Tiền xử lý và hậu xử lý là việc của bạn trên đường đi này
  support:
    - label: Kiểm tra một họ mô hình và tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Cài đặt

<code-tabs name="install" />

Extra này đã bao gồm `libreyolo[onnx]`, vì quá trình chuyển đổi chạy qua một bản
trung gian ONNX. Nó cũng mang theo tệp thực thi `mnnconvert`, thứ mà trình xuất
tìm trước tiên ở cạnh trình thông dịch Python đang hoạt động và sau đó mới tìm
trên `PATH`. Nếu thiếu bộ chuyển đổi, một `ImportError` sẽ được ném ra kèm tên
lệnh cài đặt, thay vì hỏng giữa chừng quá trình chuyển đổi.

## Xuất mô hình

<code-tabs name="export" />

Trước khi bàn giao graph, trình xuất đọc hợp đồng đầu vào của ONNX và từ chối mọi
thứ nó không diễn đạt được: nhiều hơn một đầu vào ảnh, hoặc một shape đầu vào có
chiều mang tính ký hiệu. MNN ở phiên bản này đòi hỏi một shape NCHW cố định hoàn
toàn, và `batch` được cố định vào artifact chứ không được thương lượng lúc tải.

Tệp sidecar không phải là thứ ghi chép tùy chọn. `weights/LibreYOLO9t.mnn.json`
ghi lại tên đầu vào và đầu ra, shape đầu vào cố định, batch, tên các lớp đối
tượng, phiên bản MNN đã dùng, và backend mà artifact được dựng cho, và runtime
kiểm chứng từng trường trong số đó lúc tải.

Trên Windows, MNN 3.6.1 đôi khi hoàn tất việc chuyển đổi rồi lại kết thúc trong
lúc dọn dẹp tiến trình với một access violation hoặc một trạng thái fail-fast.
Trình xuất nhận ra những mã thoát cụ thể đó và coi việc chuyển đổi là thành công
khi tệp kết quả có mặt.

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` định tuyến theo phần mở rộng `.mnn` và trả về cùng đối tượng
`Results` như checkpoint. Việc tải được thiết kế nghiêm ngặt có chủ đích: tệp
sidecar phải khai báo `format=mnn`, `mnn_backend=cpu`, `dynamic=false`,
`precision=fp32`, một kích thước, một tác vụ phát hiện đối tượng, một shape NCHW
cố định và dương khớp với kích thước ảnh đã ghi lại, và tên các lớp đối tượng phủ
mọi chỉ số từ 0 đến `nc - 1`. Mọi sai lệch đều gây lỗi thay vì đoán mò.

Dự đoán ở một `imgsz` khác với giá trị mà artifact được dựng cho cũng gây lỗi, và
`device` bị bỏ qua kèm một cảnh báo, vì các bản xuất MNN ở đây chạy trên CPU.

Đoạn mã thứ hai là đường đi runtime thuần. Tiền xử lý, giải mã, NMS và việc co
giãn tọa độ trở thành việc của bạn ở đó, và tên đầu vào và đầu ra lấy từ tệp
sidecar vì bộ tải module của MNN đòi chúng một cách tường minh.

## Ràng buộc

Chỉ phát hiện đối tượng. Backend từ chối mọi tác vụ khác lúc tải, và phía xuất
cũng khớp như vậy: ngoài các tổ hợp đã ghi nhận, bước kiểm tra trước sẽ báo lỗi
với "MNN v1 has no implemented runtime contract for this family and task."

FP32, CPU, shape cố định. `dynamic=True` gây `ValueError`, còn `half=True` và
`int8=True` bị từ chối trong bước kiểm chứng.

Các họ phát hiện đối tượng đã được kiểm chứng là YOLO9, YOLO9-E2E, YOLO9-P2,
RF-DETR, EC, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM và YOLO-NAS, mỗi họ đều
được phủ bởi việc chuyển đổi, tải lại artifact mới, chạy trên CPU bằng MNN, kiểm
tra metadata và đối chiếu trùng khớp kết quả phát hiện sau NMS với mô hình
PyTorch. DEIMv2 chuyển đổi được, tải lại được, chạy được và giữ nguyên các kết
quả phát hiện sau NMS, nhưng tuyến ONNX trung gian của nó có độ trùng khớp điểm
số ở mức query chưa đầy đủ, nên nó được ghi nhận là khả dụng chứ chưa được kiểm
chứng.

Để xem lưới đầy đủ các họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
