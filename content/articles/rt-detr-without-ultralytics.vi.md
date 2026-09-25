---
title: "RT-DETR không cần Ultralytics: v1, v2 và v4 với Apache-2.0"
description: "Chạy và tinh chỉnh RT-DETR, RT-DETRv2 và RT-DETRv4 với trọng số Apache-2.0 trong thư viện MIT: một API cho predict, train, val và export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR có miễn phí cho mục đích thương mại không?"
    a: "Có, nếu bạn dùng một bản triển khai có giấy phép dễ dãi. Code gốc của RT-DETR và RT-DETRv2 (lyuwenyu/RT-DETR) cùng RT-DETRv4 (RT-DETRs/RT-DETRv4) dùng Apache-2.0, còn LibreYOLO chạy cả ba phiên bản với trọng số Apache-2.0 trong thư viện MIT. Bản triển khai RT-DETR bên trong package ultralytics dùng AGPL-3.0, với Enterprise License có phí là lựa chọn thay thế cho việc sử dụng mã nguồn đóng. Đây là thông tin chung, không phải tư vấn pháp lý."
  - q: "RT-DETRv4 dùng giấy phép nào?"
    a: "Apache-2.0. Tệp LICENSE tại github.com/RT-DETRs/RT-DETRv4 dùng Apache-2.0. RT-DETRv4 chỉ dùng teacher DINOv3 trong quá trình huấn luyện; trọng số student được phát hành không chứa tham số DINOv3, vì vậy giấy phép DINOv3 của Meta không áp dụng cho chúng."
  - q: "Tôi có thể tinh chỉnh RT-DETR mà không cần Ultralytics không?"
    a: "Có. LibreYOLO tinh chỉnh các checkpoint phát hiện đối tượng của RT-DETR, RT-DETRv2 và RT-DETRv4 bằng model.train(), hoặc dùng libreyolo train trên dòng lệnh, bắt đầu từ trọng số COCO đã phát hành. Các mô hình hộp xoay trên RT-DETRv2 chỉ dùng để suy luận."
  - q: "LibreYOLO có những trọng số RT-DETR nào?"
    a: "RT-DETR gồm r18, r34, r50, r50m, r101, l và x; RT-DETRv2 gồm r18, r34, r50, r50m và r101; RT-DETRv4 gồm s, m, l và x, tất cả đều phát hiện đối tượng trên COCO ở 640 px. RT-DETRv2 còn có các mô hình hộp xoay n, s, m, l và x được huấn luyện trên DOTA v1.0 ở 1024 px. Tất cả tệp đều dùng Apache-2.0."
---

RT-DETR là transformer phát hiện đối tượng thời gian thực của Baidu. Mô hình giải mã một tập truy vấn cố định thay vì một lưới dày đặc, vì vậy không có bước NMS cần tinh chỉnh. Khi tìm kiếm về mô hình này, một trong những bản triển khai đầu tiên bạn thấy thường nằm trong package Python `ultralytics`. Điều đó đặt ra câu hỏi về giấy phép mà bản thân mô hình chưa từng có.

## Giấy phép RT-DETR phụ thuộc vào repo

Giấy phép thuộc về repo, không phải mô hình. RT-DETR có một số repo và chúng không dùng cùng giấy phép:

| Repo | Phạm vi | Giấy phép |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR và RT-DETRv2, PyTorch và Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR bên trong package `ultralytics` | AGPL-3.0 hoặc Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 và RT-DETRv4 | Code MIT, trọng số Apache-2.0 |

Bản triển khai của Ultralytics có chất lượng tốt. [Trang RT-DETR](https://docs.ultralytics.com/models/rtdetr/) liệt kê `rtdetr-l.pt` và `rtdetr-x.pt` được huấn luyện sẵn, cùng các tính năng huấn luyện, validation, inference và export. Bản này được phân phối bên trong package dùng giấy phép AGPL-3.0, và Ultralytics bán Enterprise License cho các nhóm không muốn mở mã nguồn toàn bộ dự án. Nếu cả hai lựa chọn đều không phù hợp, kiến trúc này có sẵn theo Apache-2.0 từ chính các tác giả. [Bài giải thích giấy phép YOLO](/articles/yolo-licenses-explained) trình bày cùng mô hình này trong toàn bộ họ YOLO, còn [hướng dẫn giấy phép thương mại](/articles/yolo-commercial-license) giải thích AGPL-3.0 có ý nghĩa gì với sản phẩm mã nguồn đóng.

## Chạy RT-DETR với LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Có thể gọi tương tự từ dòng lệnh:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` và `max_det` lọc kết quả giải mã top-k trên các truy vấn và lớp đối tượng. `iou` được chấp nhận nhưng không được dùng vì không có NMS. Đối tượng `Results` trả về giống với đối tượng mà mọi mô hình LibreYOLO đều trả về, nên việc đổi detector chỉ cần sửa một dòng.

## RT-DETR, RT-DETRv2 và RT-DETRv4 trong cùng một loader

Phiên bản nằm trong tên tệp, còn `LibreYOLO()` tự chọn theo checkpoint, vì vậy cả ba phiên bản đều được nạp theo cùng một cách:

* **RT-DETR:** `LibreRTDETR` gồm r18, r34, r50, r50m, r101 (backbone ResNet) và l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` gồm r18, r34, r50, r50m và r101. Mô hình giữ nguyên kiến trúc của phiên bản 1 và thay đổi cách deformable attention lấy mẫu.
* **RT-DETRv4:** `LibreRTDETRv4` gồm s, m, l và x. Mô hình dùng lại kiến trúc của D-FINE, còn trọng số được tạo bằng cách chưng cất từ teacher DINOv3 sang student HGNetv2.

Tất cả trọng số phát hiện đối tượng đều được huấn luyện trên COCO và chạy ở 640 px. Để tham khảo, các README upstream báo cáo 46.5 AP cho RT-DETR-R18 và 53.0 AP cho RT-DETR-L trên COCO, cùng mức từ 49.8 AP cho RT-DETRv4-S đến 57.0 AP cho RT-DETRv4-X. [Trang tài liệu RT-DETR](/docs/models/rt-detr) có bảng benchmark riêng của LibreYOLO cho từng checkpoint.

RT-DETRv2 cũng hỗ trợ hộp xoay. `LibreRTDETRv2n-obb.pt` đến `LibreRTDETRv2x-obb.pt` là các checkpoint DOTA v1.0 chính thức, gồm 15 lớp đối tượng trên ảnh hàng không ở 1024 px, đã được chuyển đổi sang định dạng của LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[Phát hiện đối tượng hộp xoay](/docs/tasks/oriented-detection) trình bày định dạng nhãn và các metric.

## Tinh chỉnh RT-DETR trên dữ liệu của bạn

Huấn luyện bắt đầu từ một checkpoint đã phát hành:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Để chạy thực tế, hãy trỏ `data` đến YAML của dataset riêng. Mỗi phiên bản có learning rate mặc định riêng, còn phiên bản 1 và 2 đặt learning rate của backbone bằng một phần hai mươi `lr0`, theo recipe gốc. Lệnh CLI cho nhiều GPU là `device=0,1`, còn tinh chỉnh adapter [LoRA](/docs/train/lora) dùng `lora=True` sau khi cài extra `lora`. Các mô hình hộp xoay trên RT-DETRv2 chỉ dùng để suy luận. Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu và logger.

## Validation và export

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` trả về một dictionary thông thường. Các định dạng export gồm ONNX, TorchScript, ExecuTorch, TensorRT và OpenVINO; bảng export trên [trang mô hình](/docs/models/rt-detr) cho biết định dạng nào đã được kiểm chứng cho từng tác vụ. Tệp `.onnx` hoặc `.engine` đã export có thể được nạp lại qua `LibreYOLO()` và trả về cùng đối tượng `Results`. Xem [export](/docs/export) để đọc hướng dẫn cho từng định dạng.

## Khi nào nên dùng các repo gốc

Nếu bạn cần tái tạo kết quả của một bài báo bằng đúng cấu hình của tác giả, cần phiên bản Paddle, hoặc muốn tự chạy quá trình chưng cất teacher DINOv3 của RT-DETRv4, hãy dùng các repo upstream. LibreYOLO tinh chỉnh student v4 đã phát hành, chứ không chạy teacher. Nếu dự án của bạn đã dùng AGPL-3.0 hoặc được Enterprise License của Ultralytics bao phủ thì không có lý do về giấy phép để chuyển đổi.

## Lưu ý về giấy phép

Code của LibreYOLO dùng MIT. Mọi tệp trọng số RT-DETR đều dùng Apache-2.0, và mỗi repo trọng số trên Hugging Face đều có LICENSE và NOTICE upstream, những tệp mà Apache-2.0 yêu cầu giữ lại khi bạn phân phối lại trọng số. Trọng số được huấn luyện trên dữ liệu của riêng bạn thuộc về bạn. RT-DETRv4 chỉ dùng DINOv3 làm teacher trong quá trình huấn luyện, còn các student được phát hành không chứa tham số DINOv3. Đây là thông tin chung, không phải tư vấn pháp lý. Hãy kiểm tra các tệp LICENSE hiện hành trước khi phát hành.

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần thay đổi code.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu RT-DETR](/docs/models/rt-detr)
