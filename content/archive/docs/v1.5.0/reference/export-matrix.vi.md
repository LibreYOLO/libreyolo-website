---
title: Ma trận xuất đầy đủ
seo_title: Ma trận hỗ trợ xuất LibreYOLO và các quy tắc
description: >-
  Cách LibreYOLO quyết định một tổ hợp họ, tác vụ và định dạng có xuất được
  không: mười hai định dạng, ba tầng, quy tắc dự phòng và ngưỡng tương đương.
lead: >-
  Hỗ trợ xuất là phép tra cứu trên bộ ba (họ, tác vụ, định dạng). Trang này mô
  tả hình dạng ma trận, các quy tắc điền những ô không có mục rõ ràng và cách
  truy vấn tổ hợp bạn quan tâm.
keywords:
  - hỗ trợ xuất libreyolo
  - ma trận xuất
  - onnx tensorrt openvino tflite
  - libreyolo formats command
  - ngưỡng tương đương khi xuất
  - lỗi NotImplementedError khi xuất
last_verified: 1.5.0
verification: >-
  Định dạng, tầng, thứ tự dự phòng, block tác vụ, họ và NCNN được đọc từ
  libreyolo/export/support.py; bí danh và đối số dùng chung lấy từ
  libreyolo/export/exporter.py; định nghĩa tầng lấy từ
  docs/adr/0011-export-support-tiers.md; ngưỡng tương đương lấy từ
  docs/export_support.md, tất cả ở v1.5.0. Các ô theo tổ hợp không được chép lại
  tại đây; hãy truy vấn bằng snippet bên dưới.
snippets:
  usage:
    - label: 'Truy vấn ma trận, không cần mô hình'
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Xuất và đọc lý do từ chối
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # Kiểm tra trước khi gọi: tổ hợp bị chặn phát sinh lỗi trong preflight
        # và thông báo chứa lý do này.
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Hình dạng ma trận

Ma trận được lập khóa bằng `(family, task, format)`. Khóa họ là tên chuẩn từ
registry mô hình, khóa tác vụ lấy từ `libreyolo.tasks.TASKS`, và có mười hai định dạng:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` còn chấp nhận hai bí danh: `engine` cho `tensorrt` và
`litert` cho `tflite`, là tên hiện tại của TensorFlow Lite. Định dạng và hậu tố
`.tflite` không đổi.

<code-tabs name="usage" />

Vì một ô là hàm của ba khóa, lưới đầy đủ rất lớn và thay đổi theo mỗi bản phát
hành. Ma trận được tạo tự động thay vì viết thủ công và nằm trong
`docs/export_support.md` của repo thư viện. Hãy truy vấn ma trận từ Python hoặc
CLI thay vì đọc một bản sao.

## Ba tầng

| Tầng | Ý nghĩa |
|---|---|
| `validated` | Mức tương đương số học được bao quát trong CI hoặc lượt chạy hằng đêm có tài liệu |
| `available` | Đã triển khai chuyển đổi nhưng chưa ghi bằng chứng tương đương số học trong runtime |
| `blocked` | Preflight phát sinh `NotImplementedError` kèm lý do trước khi truy vết |

Cả tổ hợp validated và available đều tiếp tục mà không cần xác nhận hay cảnh báo
chung. Bằng chứng và ràng buộc đã ghi vẫn hiển thị trong tài liệu được tạo. Tổ
hợp blocked thất bại trước bước kiểm tra dependency, tải dữ liệu hiệu chuẩn,
truy vết hoặc tạo artifact.

Việc thêm mục validated cần phép kiểm thử tương đương và trường `since`.

`SupportEntry` chứa bốn trường: `tier`, chuỗi `reason`, bản phát hành `since` và
chuỗi `constraint`. Ràng buộc là phần quan trọng khi tích hợp: dấu kiểm chỉ áp
dụng trong các điều kiện được nêu, thường là canvas đầu vào cố định, batch 1,
FP32 và phiên bản runtime cụ thể.

## Cách quyết định một ô

`get_support(family, task, fmt)` phân giải theo thứ tự sau. Quy tắc khớp đầu tiên được áp dụng.

1. Tác vụ không xác định hoặc định dạng ngoài mười hai dạng trả về `blocked`.
2. Mục `(family, task, format)` rõ ràng trả về như đã ghi.
3. Block toàn họ trả về `blocked` cùng lý do của họ.
4. Block toàn tác vụ trả về `blocked` cùng lý do của tác vụ.
5. Với `ncnn`, họ trong danh sách chặn NCNN trả về `blocked`.
6. `mnn` trả về `blocked`: không có giao diện runtime cho họ và tác vụ này.
7. `rknn` trả về `blocked`. RKNN trong phiên bản này giới hạn ở đúng các biến thể phát hiện đã kiểm thử trên simulator: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s và PicoDet-s trên RK3588.
8. `tensorrt` và `openvino` trả về `available`: có luồng chuyển đổi nhưng chưa ghi mức tương đương runtime cho họ và tác vụ đó.
9. `tflite`, `paddle`, `coreai` và `coreml` trả về `blocked`, mỗi dạng có lý do riêng.
10. Mọi dạng khác trả về `available`: đã triển khai chuyển đổi nhưng chưa ghi mức tương đương số học trong runtime.

Sự bất đối xứng trong bước 8 đến 10 là có chủ ý. TensorRT và OpenVINO chuyển đổi
chung từ ONNX, nên tổ hợp chưa liệt kê vẫn đáng thử. TFLite, Paddle, Core AI và
CoreML đều cần luồng theo từng họ, nên tổ hợp chưa liệt kê bị từ chối thay vì
được khuyến khích thử.

## Tác vụ bị chặn

Các tác vụ sau bị chặn cho mọi họ không có mục rõ ràng.

| Tác vụ | Lý do |
|---|---|
| `ocr` | Hai mạng có crop động theo vùng không phù hợp giao diện xuất một đồ thị |
| `point` | Họ chưa được nối với heatmap điểm dùng chung và giao diện giải mã đỉnh backend |
| `semantic` | Họ chưa được nối với dense-logits dùng chung và giao diện argmax backend |
| `mesh` | Đầu ra đồ thị mesh cơ thể, metadata và giao diện runtime chưa được định nghĩa |
| `normal` | Họ chưa được nối với giao diện pháp tuyến đơn vị dày đặc canvas cố định và tái chuẩn hóa backend |
| `panoptic` | Xuất panoptic không có giao diện runtime backend |
| `gaze` | Họ chưa được nối với logits hai head dùng chung và giao diện giải mã kỳ vọng backend |

Mục rõ ràng ghi đè các quy tắc này, nhờ đó chẳng hạn một họ semantic đã nối vẫn xuất được.

## Họ bị chặn

| Họ | Bị chặn đối với |
|---|---|
| `depth_anything3` | Mọi định dạng; đồ thị độ sâu không nằm trong giao diện runtime đã xuất |
| `domedetr` | Mọi định dạng. PAQI đặt số query theo từng ảnh, nên đồ thị đã truy vết chỉ hợp lệ cho ảnh dùng để truy vết. Hãy dùng D-FINE nếu cần DETR có thể xuất |
| `eomt` | Xuất instance và panoptic vì không có phân tích runtime |
| `l2cs` | Mọi dạng ngoài ONNX, TorchScript, ExecuTorch, TensorRT và OpenVINO |
| `hrnet` | Mọi dạng ngoài ONNX, TorchScript, OpenVINO và TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Mọi định dạng; xuất mô hình dùng prompt nằm ngoài giao diện runtime v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Mọi định dạng; xuất runtime open-vocabulary nằm ngoài phạm vi v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Mọi định dạng; xuất VLM sinh nằm ngoài phạm vi v1 |

PicoSAM3 là ngoại lệ trong tầng dùng prompt: mô hình xuất mạng ROI 96 pixel thô sang ONNX.

## Bị chặn cho NCNN

Decoder kiểu DETR cần các thao tác lấy mẫu mà NCNN không triển khai, nên các họ
sau bị chặn với `ncnn` trừ khi mục rõ ràng quy định khác: Deformable DETR, DETR,
DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4,
RF-DETR và EC. Thông báo từ chối nêu ONNX, OpenVINO, TorchScript và TensorRT làm
phương án thay thế.

## Ngưỡng tương đương

Ô validated nghĩa là artifact đã xuất tái tạo mô hình native trong các giới hạn sau:

| Nhóm tác vụ | Ngưỡng |
|---|---|
| Phát hiện và OBB | IoU box đã ghép trên 0.95, MAE điểm dưới 0.01 |
| Phân đoạn và panoptic | IoU mask trên 0.95 |
| Tư thế | L2 keypoint dưới 2 pixel ở độ phân giải gốc |
| Phân loại | Cosine logits trên 0.999 và lớp top-1 giống nhau |
| Độ sâu và khôi phục | PSNR trên 40 dB so với đầu ra native |
| Pháp tuyến bề mặt | Sai số góc trung bình dưới 0.1 độ |
| Điểm | Vị trí đỉnh giống nhau trong phạm vi một ô đầu ra |

Các hàng query DETR là tập không có thứ tự, nên phép tương đương họ DETR căn chỉnh hàng query theo tập thay vì theo vị trí.

## Xuất

<code-tabs name="export" />

Tổ hợp blocked phát sinh `NotImplementedError` trong preflight và thông báo chứa
lý do đã ghi. `validated_alternatives(family, task)` trả về các định dạng
validated cho cặp đó, hữu ích để in bên cạnh thông báo từ chối.

Các đối số mọi trình xuất dùng chung được liệt kê trên
[trang API mô hình](/docs/reference/model-api). Đối số riêng theo định dạng nằm
trên trang của từng định dạng.

## Đọc ràng buộc

Ô validated là tuyên bố về một cấu hình đã đo, không phải về định dạng nói chung.
Chuỗi ràng buộc như `FP32, batch 1, fixed 520x520 input` nghĩa là mức tương đương
được ghi ở shape và độ chính xác số đó. Xuất ở độ phân giải hoặc kích thước batch
khác vẫn tạo artifact; đó chỉ không phải cấu hình tạo ra số liệu.

