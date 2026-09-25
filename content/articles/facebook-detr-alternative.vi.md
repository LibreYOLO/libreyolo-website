---
title: "Giải pháp thay thế Facebook DETR năm 2026: chạy bằng LibreYOLO"
description: "Repo DETR gốc của Facebook Research đã được lưu trữ. Chạy bốn biến thể DETR chính thức bằng LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR đã bị bỏ chưa?"
    a: "Repo facebookresearch/detr gốc đã được lưu trữ vào ngày 12 tháng 3 năm 2024."
  - q: "LibreYOLO có thể huấn luyện DETR gốc không?"
    a: "Không. LibreYOLO hỗ trợ dự đoán, validation và xuất DETR, nhưng không hỗ trợ recipe huấn luyện 500 epoch của mô hình này."
---

Repo [Facebook Research DETR](https://github.com/facebookresearch/detr) gốc đã được lưu trữ vào ngày 12 tháng 3 năm 2024. LibreYOLO vẫn cho phép sử dụng detector gốc thông qua cùng API với các dòng mô hình khác.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Có bốn biến thể ResNet-50 và ResNet-101 để dự đoán, validation và xuất sang ONNX và TorchScript. Tính năng huấn luyện chưa được triển khai. LibreYOLO cũng dùng đầu vào cố định 800 x 800, thay vì tái tạo pipeline đánh giá gốc có giữ nguyên tỷ lệ khung hình.

Hãy thử sau khi chạy `pip install libreyolo`. [Tài liệu DETR](https://www.libreyolo.com/docs/models/detr) liệt kê cả bốn checkpoint. Để so sánh với CNN, xem [giải pháp thay thế EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
