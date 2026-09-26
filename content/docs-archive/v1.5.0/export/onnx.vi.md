---
title: ONNX
seo_title: Xuất sang ONNX từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO sang ONNX: opset mà LibreYOLO chọn cho từng họ mô
  hình, dynamic axes, NMS nhúng sẵn, INT8, và cách graph được tải lại.
lead: >-
  ONNX là một định dạng đồ thị (graph) di động. LibreYOLO trace mô hình bằng
  torch.onnx.export, tùy chọn đơn giản hóa graph, rồi ghi họ mô hình, tác vụ,
  tên các lớp đối tượng và kích thước đầu vào vào chính metadata của tệp, để mọi
  backend LibreYOLO đều có thể dựng lại phần hậu xử lý.
keywords:
  - xuất yolo sang onnx
  - onnxruntime python
  - torch.onnx.export
  - onnx opset là gì
  - dynamic axes onnx
  - nhúng nms vào onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="onnx")
    mono: true
  - label: Kết quả ghi ra
    value: 'Một tệp .onnx, metadata được nhúng ngay trong graph'
  - label: Phụ thuộc thêm
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Hình dạng
    value: >-
      Batch động theo mặc định trong Python; các ngoại lệ theo từng tác vụ ở bên
      dưới
  - label: Precision
    value: 'FP32, FP16 (half=True), INT8 (int8=True, phát hiện đối tượng YOLO9)'
verification: >-
  Đọc từ libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py và
  libreyolo/cli/commands/export.py trên nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Tham số
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, hoặc (chiều cao, chiều rộng)
            batch=1,
            dynamic=True,     # mặc định trong Python; CLI mặc định là False
            simplify=True,    # chạy onnxsim trên graph
            opset=None,       # None chọn 13, hoặc 17 cho các họ theo kiểu DETR
            half=False,       # trọng số và activation ở FP16
            int8=False,       # QDQ INT8, chỉ cho phát hiện đối tượng YOLO9
            data=None,        # data.yaml để hiệu chỉnh, chỉ dùng cho INT8
            device=None,      # thiết bị để trace; None dùng thiết bị của mô hình
            output_path=None, # None ghi ra weights/<stem>.onnx
        )
  nms:
    - label: Nhúng NMS vào graph
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Chỉ phát hiện đối tượng YOLO9, batch 1. dynamic bị ép về False
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 với dữ liệu hiệu chỉnh
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # vài trăm ảnh đại diện
            fraction=1.0,
        )
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime thuần
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Trên đường này, tiền xử lý và hậu xử lý là việc của bạn

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Graph mang theo họ mô hình, tác vụ, tên các lớp đối tượng và kích
        thước đầu vào

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Kiểm tra một họ mô hình và tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Cài đặt

<code-tabs name="install" />

Phụ thuộc thêm này kéo về `onnx`, `onnxsim` và `onnxruntime`. Chỉ riêng `onnx` là
đủ để ghi tệp; `onnxsim` chạy bước đơn giản hóa còn `onnxruntime` chạy artifact và
thực hiện việc hiệu chỉnh (calibration) INT8.

## Xuất mô hình

<code-tabs name="export" />

Khi không có `output_path`, tệp được ghi vào `weights/` theo stem của checkpoint,
kèm `_fp16` hoặc `_int8` ở cuối khi precision đó được yêu cầu.

`dynamic` mặc định là `True` trong Python và `False` trên CLI. Khi nó được bật,
trục batch trở thành ký hiệu tượng trưng và một vài tác vụ còn mở rộng thêm: phân
đoạn ngữ nghĩa mở thêm cả chiều cao và chiều rộng của mặt nạ (mask), phục hồi ảnh
bằng Real-ESRGAN mở các trục không gian, còn các bộ phát hiện hai giai đoạn giữ
chiều cao và chiều rộng của ảnh gốc ở dạng động vì bước resize của chúng diễn ra
bên trong graph.

`opset` được chọn theo từng họ mô hình khi bị bỏ trống. Các họ theo kiểu DETR
(`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`,
`rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) cùng với `deit`, `midas` và `moge2`
nhận opset 17, vì đó là nơi `aten::scaled_dot_product` được hạ xuống. Mọi thứ còn
lại nhận 13. Matting luôn được nâng lên 19 bất kể thế nào, vì decoder của BiRefNet
cần toán tử `DeformConv`, thứ mà ONNX chỉ định nghĩa từ opset 19.

`simplify=True` chạy `onnxsim` và giữ lại graph gốc nếu bước này thất bại, nên một
lỗi đơn giản hóa chỉ là cảnh báo chứ không phải là một lần xuất mô hình thất bại.
Trên macOS arm64 với `onnx` 1.22 trở lên và `onnxsim` 0.6.5 trở xuống, bước này bị
bỏ qua hoàn toàn, vì cặp phiên bản đó có thể làm tiến trình Python bị hủy đột ngột.

### NMS nhúng sẵn

<code-tabs name="nms" />

`nms=True` chỉ dành cho phát hiện đối tượng YOLO9 và đòi hỏi batch 1; yêu cầu nó
kèm `dynamic=True` sẽ ghi ra một cảnh báo và tắt dynamic đi. Khi đó graph có hai
đầu ra: `output`, hình dạng `(batch, max_det, 6)`, và `raw`, là tensor detector
chưa giải mã mà chính backend của LibreYOLO dùng, để phần hậu xử lý giữ nguyên y
hệt như đường PyTorch.

### DeepStream

`deepstream=True` là một tùy chọn chỉ có ở ONNX. Nó xuất graph theo đúng bố cục mà
parser của NVIDIA DeepStream mong đợi và ghi kèm hai tệp phụ bên cạnh,
`config_infer_primary_<stem>.txt` và `<stem>_labels.txt`, để artifact ghép thẳng
vào một pipeline mà không cần viết tay cấu hình.

Nó loại trừ lẫn nhau với `nms=True`, và yêu cầu cả hai sẽ ném ra một `ValueError`:
DeepStream chạy bước suppression trong giai đoạn clustering của riêng nó. Truyền
nó cho bất kỳ định dạng nào khác ngoài ONNX cũng ném lỗi. Xem
[DeepStream](/docs/export/deepstream) để biết lưới họ mô hình và tác vụ được hỗ
trợ cùng cách build parser.

### INT8

<code-tabs name="int8" />

`int8=True` chạy lượng tử hóa (quantization) tĩnh của ONNX Runtime và ghi ra một
graph QDQ với đầu vào và đầu ra ở float32. Chỉ các node `Conv` và `Gemm` được lượng
tử hóa. Việc để phần giải mã của detection head ở float32 là có chủ đích: phép nối
đó trộn tọa độ hộp theo thang pixel với điểm số lớp đối tượng trong khoảng 0 đến 1,
và một activation scale duy nhất trên mỗi tensor, bị chi phối bởi độ lớn của hộp,
sẽ đẩy mọi điểm số về không.

Cờ này hiện chỉ áp dụng cho phát hiện đối tượng YOLO9, còn mọi thứ khác đều ném
`NotImplementedError` ở bước preflight. Bỏ trống `data` sẽ quay về `coco8.yaml` kèm
một cảnh báo; tám ảnh không phải là một tập hiệu chỉnh đại diện. Một mô hình đã
được lượng tử hóa sẵn trong PyTorch đi theo con đường khác, được mô tả ở
[Lượng tử hóa](/docs/export/quantization).

## Chạy artifact

<code-tabs name="run" />

`LibreYOLO()` định tuyến theo phần đuôi `.onnx` và trả về cùng đối tượng `Results`
như một checkpoint `.pt`, vì tên các lớp đối tượng, tác vụ, kích thước đầu vào và
schema pose đã được ghi vào `metadata_props` của graph ngay lúc xuất. Với
`device="auto"`, session lấy `CUDAExecutionProvider` khi ONNX Runtime báo là có
provider này, còn không thì quay về CPU.

Snippet thứ hai dành cho người đọc không cài LibreYOLO. Trên đường đó, tiền xử lý,
việc giải mã, NMS và việc rescale lại tọa độ đều trở thành việc của bạn; khối
metadata thì vẫn nằm đó để đọc.

## Giới hạn

Tên các tensor đầu ra là cố định theo từng tác vụ, và đó là thứ mà bên tiêu thụ
không đọc metadata phải khớp theo:

| Tác vụ | Tên đầu ra |
|---|---|
| Phát hiện đối tượng, head dạng grid và anchor | `output` |
| Phát hiện đối tượng, kiểu DETR | `pred_logits`, `pred_boxes` |
| Phát hiện đối tượng, RF-DETR | `dets`, `labels` |
| Phân loại | `output` |
| Phân đoạn ngữ nghĩa | `semantic_logits` |
| Độ sâu | `depth` |
| Pháp tuyến bề mặt | `normal` |
| Biên | `edges` |
| Phục hồi ảnh | `restored` |
| Matting | `matte` |
| Hướng nhìn | `yaw_logits`, `pitch_logits` |

RF-DETR cũng là họ mô hình duy nhất có tensor đầu vào mang tên `input` thay vì
`images`.

Một vài tác vụ mang theo hợp đồng runtime với độ phân giải cố định trong phiên bản
này. Độ sâu, pháp tuyến bề mặt và biên từ chối `batch != 1` và ép `dynamic=False`.
Matting ép về khung vuông 1024 gốc, vì các bảng relative-position của Swin trong
BiRefNet gắn chặt với độ phân giải của chúng. Phục hồi ảnh ép một khung cố định cho
mọi họ mô hình trừ Real-ESRGAN, vốn có generator hoàn toàn tích chập.

`imgsz` hình chữ nhật dùng được với các họ YOLO9, HRNet, NAFNet và Real-ESRGAN. Các
họ mô hình có hợp đồng vuông cố định (`clip`, `deformable_detr`, `detr`,
`dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) từ chối thẳng.

Hai tổ hợp bị từ chối trước khi trace: phân đoạn với YOLO9, vì trong LibreYOLO
YOLO9 chỉ làm phát hiện đối tượng, và phân đoạn với RTMDet-Ins, vốn có phần giải mã
mặt nạ bằng dynamic kernel không có hợp đồng nào cho runtime đã xuất.

Để xem toàn bộ lưới họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể, hãy
hỏi thẳng thư viện:

<code-tabs name="support" />
