---
title: "Giải pháp thay thế MMYOLO năm 2026: chạy RTMDet với LibreYOLO"
description: "Bản phát hành mới nhất của MMYOLO là từ năm 2023. Chạy các bộ phát hiện được hỗ trợ như RTMDet thông qua API trực tiếp của LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO đã bị bỏ dở chưa?"
    a: "MMYOLO chưa chính thức bị lưu trữ, nhưng bản phát hành GitHub mới nhất là phiên bản 0.6.0 từ tháng 8 năm 2023."
  - q: "LibreYOLO có thể tải mọi checkpoint MMYOLO không?"
    a: "Không. LibreYOLO hỗ trợ các dòng mô hình được chỉ định và không cam kết tải trực tiếp mọi checkpoint MMYOLO tùy ý."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) chưa bị lưu trữ, nhưng bản phát hành mới nhất, phiên bản 0.6.0, đã có từ tháng 8 năm 2023. Nếu bạn chỉ cần một bộ phát hiện được hỗ trợ như RTMDet, có thể bạn không cần toàn bộ stack OpenMMLab.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO không phải là bản thay thế hoàn chỉnh cho MMYOLO. Thư viện không tái tạo mọi config, recipe hay checkpoint. Thay vào đó, LibreYOLO cung cấp cho các dòng được hỗ trợ như RTMDet và YOLOX một API trực tiếp để dự đoán, huấn luyện, đánh giá và xuất mô hình.

Bắt đầu bằng lệnh `pip install libreyolo`. [Tài liệu RTMDet](https://www.libreyolo.com/docs/models/rtmdet) liệt kê các thao tác được hỗ trợ. [Hướng dẫn RTMDet ngắn](/articles/rtmdet-without-mmdetection) trình bày cách chuyển đổi.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
