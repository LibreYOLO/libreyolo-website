---
title: "YOLOX Alternative 2026: Chạy YOLOX với LibreYOLO"
description: "Bản phát hành YOLOX mới nhất là từ năm 2022. Chạy, huấn luyện, đánh giá và xuất YOLOX qua API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX đã bị bỏ rơi chưa?"
    a: "Repo chính thức chưa được lưu trữ, nhưng bản phát hành mới nhất là YOLOX 0.3.0 từ tháng 4 năm 2022. Nói rằng dự án ít hoạt động sẽ chính xác hơn là nói dự án đã chính thức bị bỏ rơi."
  - q: "LibreYOLO có thể huấn luyện YOLOX không?"
    a: "Có. LibreYOLO hỗ trợ dự đoán, huấn luyện, đánh giá và xuất YOLOX."
---

YOLOX vẫn là một mô hình phát hiện đối tượng hữu ích theo giấy phép Apache-2.0. [Repo chính thức](https://github.com/Megvii-BaseDetection/YOLOX) chưa được lưu trữ, nhưng bản phát hành mới nhất, 0.3.0, ra mắt vào tháng 4 năm 2022. Vấn đề bảo trì là có thật. Mô hình vẫn hoạt động.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO hỗ trợ sáu kích thước YOLOX, từ Nano đến X, với chức năng dự đoán, huấn luyện, đánh giá và xuất trong cùng một API. Hãy dùng dự án gốc nếu bạn cần workflow cấu hình `Exp` chính xác như ban đầu.

Bắt đầu bằng lệnh `pip install libreyolo`. [Tài liệu YOLOX](https://www.libreyolo.com/docs/models/yolox) trình bày đầy đủ API. Ngoài ra còn có ghi chú ngắn về việc [YOLO-NAS vẫn được bảo trì](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
