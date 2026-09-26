---
title: Paddle
seo_title: Xuất sang PaddlePaddle từ LibreYOLO
description: >-
  Chuyển một mô hình phát hiện đối tượng LibreYOLO thành mô hình suy luận
  (inference) của PaddlePaddle qua X2Paddle: toolchain đã ghim, đồ thị tĩnh
  batch 1 FP32, và inference trên CPU.
lead: >-
  Mô hình inference của PaddlePaddle là một đồ thị model.pdmodel nằm cạnh một
  tệp trọng số model.pdiparams. LibreYOLO xuất một đồ thị ONNX tĩnh ở opset 15,
  chuyển đổi nó bằng X2Paddle, rồi đóng gói kết quả kèm một metadata.yaml để nó
  tải được qua đúng factory như mọi runtime khác.
keywords:
  - xuất yolo sang paddle
  - paddlepaddle inference
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Kết quả ghi ra
    value: 'Một thư mục chứa model.pdmodel, model.pdiparams và metadata.yaml'
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Tải lại bằng
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Hình dạng
    value: 'Tĩnh, batch 1, opset 15. Cả ba đều được áp đặt.'
  - label: Precision
    value: 'Chỉ FP32, chỉ CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 hoặc cũ hơn, được kiểm tra
      đúng chính xác
verification: >-
  Đọc từ libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md và
  pyproject.toml trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: >
        # Python 3.10 đến 3.12. WSL2 với Ubuntu 22.04 là đường đã kiểm chứng
        trên Windows

        pip install "libreyolo[paddle]"
    - label: Xác nhận các phiên bản đã ghim
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra thư mục weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Tham số
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; canvas vuông của họ mô hình này
            batch=1,          # mọi giá trị khác đều ném ValueError
            dynamic=False,    # True ném ValueError
            simplify=True,    # False ném ValueError
            opset=15,         # mọi giá trị khác đều ném ValueError
            output_path=None, # None ghi ra weights/<stem>_paddle
        )
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Dùng backend trực tiếp
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Thứ mà LibreYOLO() dựng lên cho một thư mục Paddle. Vẫn là đối tượng
        # Results đó, không có định tuyến qua factory ở giữa
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle thuần
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Trên đường này, tiền xử lý và hậu xử lý là việc của bạn
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Cài đặt

<code-tabs name="install" />

Extra này ghim đúng stack mà việc đối chiếu (parity) đã đo trên đó: PaddlePaddle
2.6.2, X2Paddle 1.6.0 và ONNX 1.17 hoặc cũ hơn. Những phiên bản ghim đó được kiểm
tra ngay lúc xuất mô hình, chứ không phải chỉ lúc cài đặt, và một phiên bản khác sẽ
ném `ImportError` nêu tên phiên bản được mong đợi. Các bản Paddle mới hơn từ chối
một phần mã tĩnh mà X2Paddle 1.6.0 sinh ra, nên hỏng sớm vẫn tốt hơn là tạo ra một
artifact chưa ai kiểm chứng.

## Xuất mô hình

<code-tabs name="export" />

Bốn tham số là cố định chứ không phải chỉ có giá trị mặc định. `dynamic` phải là
`False`, `batch` phải bằng 1, `simplify` phải là `True` để có một đồ thị chuyển đổi
hoàn toàn tĩnh, và `opset` phải bằng 15, đây là mức trần mà X2Paddle 1.6.0 chấp
nhận. Truyền bất cứ giá trị nào khác đều ném lỗi trước khi trace.

Một phép chuẩn hóa chạy trên đồ thị trung gian. ONNX định nghĩa dilation bị bỏ
trống của MaxPool là một, PyTorch ghi ra thuộc tính toàn số một một cách tường
minh, còn X2Paddle 1.6.0 thì từ chối nó, nên exporter gỡ bỏ giá trị mặc định thừa
đó và giữ nguyên phép toán đã được chỉ định.

Artifact là một thư mục: `model.pdmodel`, `model.pdiparams` và `metadata.yaml`.
Phần Python mà X2Paddle sinh ra trong lúc chuyển đổi không thuộc về nó.

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` nhận ra bất kỳ thư mục nào chứa cả `model.pdmodel` lẫn
`model.pdiparams`, đọc `metadata.yaml`, và trả về cùng đối tượng `Results` như khi
dùng checkpoint. Một device khác `auto` hoặc `cpu` sẽ ném lỗi: backend này chỉ chạy
trên CPU.

Thứ mà factory dựng lên là `PaddleBackend`, được export từ `libreyolo` và có thể
import dưới dạng `libreyolo.backends.paddle.PaddleBackend`. Hãy tự dựng nó khi bạn
muốn dùng backend mà không qua việc định tuyến theo phần đuôi của factory, ví dụ để
truyền `task=` một cách tường minh cho một thư mục có `metadata.yaml` không do bạn
viết. Hàm `predict()` của nó nhận cùng các nguồn đầu vào và trả về cùng kết quả.

Snippet runtime trần phản chiếu đúng những gì backend cấu hình, và ba tùy chọn bị
tắt là cố ý. Pipeline fusion trên CPU của Paddle 2.6 có thể crash khi tối ưu các đồ
thị gather và scatter cỡ lớn sinh ra cho deformable attention, nên đồ thị tĩnh
không fuse và có tính khả chuyển mới là thứ mà parity được đo trên đó. Tiền xử lý,
việc giải mã, NMS và việc rescale lại tọa độ trở thành việc của bạn trên đường đó.

## Giới hạn

Không có hình dạng động, không FP16, không INT8, không NMS nhúng sẵn, không có
runtime GPU.

Các tổ hợp đã được kiểm chứng là phát hiện đối tượng với YOLO9, phát hiện đối tượng
với YOLO9-E2E và YOLO9-P2, phát hiện đối tượng, ước lượng tư thế và phân đoạn với
EC, phát hiện đối tượng với RT-DETRv4, D-FINE, DEIM và DEIMv2, cùng phát hiện đối
tượng và ước lượng tư thế với YOLO-NAS. Mỗi tổ hợp đều được phủ bởi việc chuyển
đổi, một lần tải lại trên runtime CPU, đối chiếu đầu ra thô và các kết quả công bố
đã khớp.

Bị chặn, kèm lý do được ghi lại cho từng tổ hợp:

| Tổ hợp | Lý do |
|---|---|
| RF-DETR, mọi tác vụ | Cần ONNX opset 17 và GridSample; X2Paddle 1.6.0 chỉ chấp nhận opset 15 trở xuống và không có mapper cho GridSample |
| RT-DETR và RT-DETRv2 phát hiện đối tượng | Các đồ thị đã huấn luyện cần GridSample ở opset 16 trở lên |
| D-FINE phân đoạn | Chuyển đổi và tải lại được, nhưng sai số RMS tương đối của logit mặt nạ (mask) là 3.52% và IoU nhỏ nhất giữa các mask khớp cặp là 0.582 |
| YOLO9 phân đoạn | Trong LibreYOLO, YOLO9 chỉ làm phát hiện đối tượng |
| RTMDet-Ins phân đoạn | Việc giải mã mask bằng kernel động không có hợp đồng nào cho runtime đã xuất |

Bất cứ thứ gì không được liệt kê là đã kiểm chứng hay bị chặn đều bị từ chối, kèm
ghi chú rằng nó chưa được kiểm chứng qua đường chuyển đổi từ ONNX sang Paddle.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
