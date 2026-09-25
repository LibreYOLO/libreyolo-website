---
title: "MiDaS Alternative năm 2026: ước lượng độ sâu với LibreYOLO"
description: "Repo MiDaS chính thức đã được lưu trữ. Chạy các mô hình ước lượng độ sâu Small và DPT-Large bằng LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaS đã bị ngừng phát triển chưa?"
    a: "Repo isl-org/MiDaS chính thức đã được lưu trữ vào August 25, 2025."
  - q: "MiDaS có trả về độ sâu theo đơn vị thực không?"
    a: "Không. MiDaS trả về độ sâu nghịch đảo tương đối, không có đơn vị đo thực hoặc tỷ lệ cố định giữa các ảnh."
---

[Repo MiDaS chính thức](https://github.com/isl-org/MiDaS) đã được lưu trữ vào August 25, 2025. LibreYOLO cung cấp cách thức được duy trì để chạy các checkpoint Small và DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Có thể dùng để dự đoán, kiểm định zero-shot và xuất ở độ phân giải cố định. Không hỗ trợ huấn luyện. Đầu ra là độ sâu nghịch đảo tương đối: giá trị cao hơn nghĩa là gần hơn, nhưng không có đơn vị đo thực hoặc tỷ lệ cố định giữa các ảnh.

Chỉ cần cài đặt cơ bản: `pip install libreyolo`. Tiếp tục với [tài liệu MiDaS](https://www.libreyolo.com/docs/models/midas) hoặc bài viết về [lựa chọn thay thế Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
