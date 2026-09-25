---
title: "EfficientDet alternative năm 2026: chạy D0-D4 với LibreYOLO"
description: "Dùng LibreYOLO để dự đoán, validation và xuất EfficientDet D0-D4 qua API phát hiện đối tượng."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDet có bị bỏ rơi không?"
    a: "Repo PyTorch của rwightman chưa chính thức bị lưu trữ. Mô tả chính xác hơn là repo đã yên ắng, chứ không bị bỏ rơi."
  - q: "LibreYOLO có thể huấn luyện EfficientDet không?"
    a: "Không. LibreYOLO hỗ trợ inference, validation và xuất EfficientDet, nhưng không hỗ trợ huấn luyện."
---

Repo [EfficientDet PyTorch được sử dụng rộng rãi](https://github.com/rwightman/efficientdet-pytorch) chưa bị lưu trữ, vì vậy gọi repo này là bị bỏ rơi sẽ không chính xác. Repo đã yên ắng. LibreYOLO cung cấp một hướng triển khai khác.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO hỗ trợ dự đoán và validation với EfficientDet D0 đến D4. Các bản xuất ONNX, TorchScript, OpenVINO và TensorRT đều có kiểm thử tính tương đương. Tính năng huấn luyện chưa được triển khai, và mỗi kích thước giữ nguyên độ phân giải đầu vào gốc cố định.

Cài đặt bằng `pip install libreyolo`. Xem [tài liệu EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) để biết thông tin từng kích thước, hoặc so sánh với [lựa chọn thay thế Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
