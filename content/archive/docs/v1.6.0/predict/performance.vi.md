---
title: Hiệu năng suy luận
seo_title: Suy luận nhanh hơn trong LibreYOLO
description: >-
  CUDA graph, half precision, batching, suy luận chia tile và tăng cường dữ liệu
  khi kiểm thử trong lúc dự đoán, cùng giá trị mặc định thực tế và các họ hỗ trợ
  từng tính năng.
lead: >-
  Năm điều khiển khi dự đoán thay đổi thông lượng hoặc độ chính xác: phát lại
  CUDA graph, độ chính xác số, batching, chia tile và tăng cường dữ liệu khi
  kiểm thử. Mỗi điều khiển áp dụng cho một tập họ cụ thể, trong đó hai điều
  khiển đánh đổi độ chính xác hoặc độ trễ thay vì tiết kiệm.
keywords:
  - cuda graph suy luận pytorch
  - suy luận batch yolo python
  - suy luận fp16
  - suy luận chia tile vật thể nhỏ
  - suy luận chia lát ảnh lớn
  - tăng cường dữ liệu khi kiểm thử phát hiện
  - capture_graph
  - dự đoán batch thư mục
last_verified: 1.5.0
verification: >-
  Giá trị mặc định của đối số lấy từ InferenceRunner.__call__ trong
  libreyolo/models/base/inference.py. API CUDA graph lấy từ
  BaseModel.capture_graph, graph_info, release_graphs và cuda_graph_scope trong
  libreyolo/models/base/model.py; trạng thái bật theo họ lấy từ biến lớp
  SUPPORTS_CUDA_GRAPH. Hành vi half precision lấy từ NOOP_PREDICT_KWARGS trong
  libreyolo/utils/predict_args.py, cảnh báo CLI trong
  libreyolo/cli/commands/predict.py, cùng CAST_RECIPES và SUPPORTED_FAMILIES
  trong libreyolo/quant/api.py. Điều kiện batching lấy từ
  InferenceRunner._process_in_batches và _predict_batch. Chia tile lấy từ
  _predict_tiled và _merge_tile_detections. Tăng cường dữ liệu khi kiểm thử lấy
  từ BaseModel._predict_augment và _merge_tta, với TTA_ENABLED, TTA_SCALES và
  TTA_FIXED_SIZE được đọc trên libreyolo/models/.
snippets:
  batch:
    - label: Suy luận theo batch trên thư mục
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # Một forward xếp chồng cho mỗi nhóm 4 ảnh trên các họ hỗ trợ.
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: Streaming để không tạo toàn bộ danh sách
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: Capture trước rồi phát lại (cần CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Chạy warmup và capture một lần, tách khỏi yêu cầu đầu tiên.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Chỉ capture khi shape lặp lại (cần CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" đợi một shape xuất hiện hai lần, nên công việc chạy một lần
        # không bao giờ chịu chi phí capture.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Cài extra xuất
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Xuất và tải lại ở độ chính xác số mặc định
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Xuất FP16 (xây dựng và chạy trên máy CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 trong PyTorch qua công thức chuyển kiểu (cần CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Công thức chuyển kiểu không đọc dữ liệu hiệu chuẩn.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Suy luận chia tile trên ảnh lớn
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Chia tile chỉ hoạt động khi ảnh lớn hơn kích thước đầu vào.
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Tăng cường dữ liệu khi kiểm thử
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Các điều khiển và giá trị mặc định

Mỗi mục sau đều là đối số của `predict` và tất cả đều tắt theo mặc định.

| Đối số | Mặc định | Tác dụng |
|---|---|---|
| `batch` | `1` | Số ảnh trong mỗi forward pass cho nguồn thư mục và danh sách |
| `cuda_graph` | `False` | Phát lại forward từ CUDA graph đã capture |
| `tiling` | `False` | Chia ảnh lớn thành các tile chồng lấn |
| `overlap_ratio` | `0.2` | Mức chồng lấn tile khi bật `tiling` |
| `augment` | `False` | Chạy các góc nhìn đã lật và hợp nhất |
| `half` | | Được chấp nhận, cảnh báo rồi bỏ qua |
| `device` | `None` | Chuyển mô hình trước khi dự đoán |

`imgsz` cũng ảnh hưởng chi phí vì đặt độ phân giải chạy mô hình, nhưng trước hết
đây là đối số độ chính xác và thuộc về mô hình thay vì phần này.

## Batching

<code-tabs name="batch" />

`batch` áp dụng cho nguồn thư mục và danh sách. Với `batch=1`, mỗi ảnh chạy một
forward pass. Trên `1`, từng nhóm được tiền xử lý, xếp thành một tensor, chạy một
lần rồi cắt lại để bước hậu xử lý ảnh đơn hiện có của từng họ nhận đúng dạng mong đợi.

Luồng xếp chồng chỉ được dùng khi thỏa tất cả điều kiện sau:

- `batch` lớn hơn `1`
- `tiling` tắt
- tăng cường dữ liệu khi kiểm thử không hoạt động
- họ đặt `SUPPORTS_BATCHED_PREDICT`
- mạng bên dưới không ở chế độ huấn luyện

Điều kiện cuối không chỉ là chi tiết kỹ thuật. Mạng ở chế độ huấn luyện sẽ chuẩn
hóa nhóm xếp chồng bằng thống kê batch xuyên ảnh, khiến các ảnh trong cùng nhóm
thay đổi dự đoán của nhau, nên các lượt đó vẫn chạy tuần tự.

`SUPPORTS_BATCHED_PREDICT` mặc định là true. Các họ sau không tham gia và chạy
một ảnh cho mỗi forward bất kể `batch`: Depth Anything V2, Depth Anything 3,
EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5,
Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, mọi detector
open-vocabulary và mọi mô hình thị giác-ngôn ngữ.

Còn một luồng dự phòng. Nếu tiền xử lý không trả về các tensor `(1, C, H, W)`
đồng nhất có shape, dtype và thiết bị khớp nhau trong nhóm, nhóm sẽ chạy tuần tự
thay vì xếp chồng, nên tính đúng đắn không phụ thuộc vào việc các ảnh tình cờ có
cùng kích thước.

Kết hợp `batch` với `stream=True` trên thư mục lớn để có forward theo batch mà không giữ mọi kết quả trong bộ nhớ.

## CUDA graph

<code-tabs name="graphs" />

CUDA graph ghi một forward pass một lần rồi phát lại dưới dạng một lần khởi chạy.
Detector nhỏ dành phần lớn thời gian batch-1 để khởi chạy kernel, nên gộp các lần
khởi chạy này giúp tăng thông lượng, còn đầu ra phát lại giống từng bit với thực
thi eager.

`cuda_graph` nhận ba giá trị. `False` là mặc định và không làm gì. `True` capture
ở lần dùng đầu tiên cho từng shape đầu vào. `"auto"` đợi đến khi shape lặp lại
trước khi capture, nên công việc chạy một lần và thay đổi shape không chịu chi phí capture.

`capture_graph(imgsz=None, batch=1, dtype=None)` chuyển chi phí đó ra khỏi yêu
cầu đầu tiên. Graph chỉ hợp lệ cho đúng shape đã capture, nên `batch` ở đây phải
khớp cách `predict` được gọi sau đó.

`graph_info()` báo cáo các graph đã capture, số lần phát lại và mọi lý do khiến
lượt chạy quay về eager. `release_graphs()` giải phóng graph cùng buffer tĩnh.

Capture cần CUDA và một họ đã bật qua `SUPPORTS_CUDA_GRAPH` vì cần forward không
có công việc hiển thị phía host, được xác minh theo từng họ. Yêu cầu trên họ chưa
bật sẽ phát sinh `NotImplementedError` thay vì âm thầm chạy eager.

Graph ghi địa chỉ bộ nhớ chứ không phải giá trị, nên mọi thao tác di chuyển tham
số đều loại graph. Thay đổi thiết bị qua `predict(device=...)`, lượng tử hóa và
giải lượng tử hóa đều làm các graph đã capture mất hiệu lực.

Ma trận hỗ trợ đầy đủ theo họ, các điểm tách đường nối và giao diện số học nằm trong [CUDA graph](/docs/reference/cuda-graphs).

## Độ chính xác số

<code-tabs name="precision" />

`half=True` khi dự đoán không làm gì. Tùy chọn được chấp nhận để tương thích dòng
lệnh, phát cảnh báo rằng đây là no-op và bị loại trước khi đến bất kỳ họ nào. Cờ
`--half` của CLI in cùng cảnh báo cho mô hình `.pt`.

Có hai cách thực sự để giảm độ chính xác số.

Với artifact đã xuất, độ chính xác số được chọn khi xuất bằng
`export(format=..., half=True)`, và tệp kết quả được tải lại không thay đổi qua
`LibreYOLO()`.

Để thực thi PyTorch, `model.quantize(recipe="fp16")` chuyển mô hình sang float16
và cài hook giữ float32 ở đầu vào cùng đầu ra mô hình. `"bf16"` làm tương tự với
bfloat16. Cả hai phép chuyển kiểu đều không đọc dữ liệu hiệu chuẩn, nên `calib`
bị bỏ qua. Lượng tử hóa hiện bao quát bốn họ: YOLOv9, RF-DETR, BiRefNet và
FeyNobg. Chuyển kiểu trên CPU ghi cảnh báo rằng thao tác sẽ chậm, nên các công
thức này dành cho GPU.

Cả hai cách đều thay đổi số học. Không cách nào bảo đảm thay thế trực tiếp mà vẫn có cùng phát hiện, vì vậy hãy đánh giá trước khi triển khai.

## Suy luận chia tile

<code-tabs name="tiling" />

Chia tile cắt ảnh lớn thành các tile vuông chồng lấn, dự đoán trên từng tile rồi
hợp nhất kết quả. Đây là tùy chọn cho đối tượng nhỏ trong ảnh độ phân giải cao,
nơi đổi kích thước toàn ảnh làm mục tiêu nhỏ hơn mức mô hình có thể phân giải.

Kích thước tile là kích thước đầu vào mô hình hoặc `imgsz` khi được cung cấp và
phải là hình vuông. `overlap_ratio` mặc định là `0.2`. Các tile chồng lấn được
điều hòa bằng non-maximum suppression theo lớp ở ngưỡng `iou`, rồi danh sách hợp
nhất được cắt còn `max_det`. Điều này nghĩa là `iou` tác động đến dự đoán chia
tile ngay cả với các họ không tự chạy NMS.

Chia tile bị bỏ qua hoàn toàn, không chỉ có chi phí thấp, khi ảnh đã vừa: nếu cả
hai chiều bằng hoặc nhỏ hơn kích thước đầu vào, một forward thông thường sẽ chạy
thay thế. Tính năng cũng bị bỏ qua cho phân loại, phân đoạn ngữ nghĩa và tác vụ
`embed`, các tác vụ này quay về một lượt vì chia tile không có ý nghĩa.

Tính năng phát sinh lỗi cho tác vụ có payload không thể ghép lại: mask phân đoạn
instance, box định hướng, điểm, độ sâu, cạnh và pháp tuyến. Không thể kết hợp với
`augment`.

Kết quả chứa `result.tiled` và `result.num_tiles`. Với `save=True`, lượt chạy chia
tile ghi một thư mục trong `runs/tiled_detections` chứa mọi tile, ảnh có chú
thích, trực quan hóa lưới và `metadata.json` ghi kích thước tile, mức chồng lấn
cùng các ngưỡng; `result.tiles_path` và `result.grid_path` trỏ đến các tệp đó.

## Tăng cường dữ liệu khi kiểm thử

<code-tabs name="tta" />

`augment=True` chạy ảnh nhiều lần và hợp nhất các phát hiện bằng non-maximum
suppression theo lớp ở ngưỡng `iou`. Giống chia tile, tính năng này khiến `iou`
có tác dụng với các họ vốn bỏ qua nó.

Trong thực tế, đây là lật ngang. Danh sách thang `TTA_SCALES` mặc định có một
thang `1.0` và không họ nào được phân phối ghi đè, nên mỗi họ chạy hai lượt: ảnh
gốc và ảnh phản chiếu. Các họ được đánh dấu `TTA_FIXED_SIZE` đổi kích thước về
hình vuông cố định, khiến multi-scale không có tác dụng trong mọi trường hợp.

Phân đoạn ngữ nghĩa và panoptic dùng cách hợp nhất khác. Góc nhìn đã lật được lật
lại, rồi hai phân phối softmax được lấy trung bình trước argmax thay vì hợp nhất
như box.

Tăng cường dữ liệu khi kiểm thử không có cho mọi tác vụ. Tính năng phát sinh lỗi
cho box định hướng, tư thế, điểm, độ sâu, pháp tuyến, cạnh, khôi phục, OCR và mô
hình embedding, đồng thời không thể kết hợp với chia tile.

Các họ sau tắt hoàn toàn tính năng, nên `augment=True` chạy một lượt thông thường:
BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS, NAFNet,
PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED, mọi biến
thể SAM, mọi detector open-vocabulary và mọi mô hình thị giác-ngôn ngữ.

## Đo lường

Trang này không đưa ra số liệu độ trễ vì một mili giây không kèm phần cứng,
runtime, độ chính xác số và kích thước batch không phải dữ kiện có ý nghĩa. Số
liệu đo trên nhiều phần cứng và runtime được công bố tại
[visionanalysis.org](https://www.visionanalysis.org), còn `libreyolo profile`
đo một mô hình cụ thể trên máy của bạn.

