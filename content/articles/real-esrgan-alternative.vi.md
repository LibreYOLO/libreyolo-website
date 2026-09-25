---
title: "Real-ESRGAN Alternative năm 2026: phóng to ảnh với LibreYOLO"
description: "Chạy tính năng phóng to ảnh 2x và 4x của Real-ESRGAN bằng LibreYOLO, bao gồm suy luận chia tile cho ảnh lớn."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN đã bị bỏ không chưa?"
    a: "Repo chưa được lưu trữ, nhưng bản phát hành GitHub mới nhất là phiên bản 0.3.0, từ tháng 9 năm 2022."
  - q: "LibreYOLO có thể huấn luyện Real-ESRGAN không?"
    a: "Không. LibreYOLO hỗ trợ dự đoán, validation và xuất Real-ESRGAN, nhưng không hỗ trợ huấn luyện."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) chưa chính thức bị lưu trữ, nhưng bản phát hành GitHub mới nhất của dự án, phiên bản 0.3.0, có từ tháng 9 năm 2022. LibreYOLO chạy các checkpoint 2x và 4x của mô hình này thông qua API dự đoán thông thường.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO hỗ trợ checkpoint 2x, 4x và 4x tốc độ cao, cùng tính năng dự đoán chia tile, validation PSNR/SSIM và xuất mô hình. Tính năng huấn luyện chưa được triển khai. Cách blend điều chỉnh cường độ khử nhiễu của upstream cũng chưa có trong bản chuyển này.

Dùng `pip install libreyolo` để trải nghiệm. Xem các tùy chọn phục hồi đầy đủ trong [tài liệu Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Để tìm hiểu về độ sâu, xem [giải pháp thay thế MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
