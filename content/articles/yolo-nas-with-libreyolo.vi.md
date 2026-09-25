---
title: "YOLO-NAS vẫn được duy trì: dự đoán, huấn luyện và xuất với LibreYOLO"
description: "Chạy, huấn luyện, kiểm định và xuất YOLO-NAS trong LibreYOLO. SuperGradients phát hành phiên bản cuối vào tháng 4 năm 2024. Chúng tôi dự định huấn luyện trọng số mới từ đầu để không còn phụ thuộc vào giấy phép của Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS có bị bỏ rơi không?"
    a: "SuperGradients phát hành phiên bản 3.7.1 lần cuối vào tháng 4 năm 2024. LibreYOLO hỗ trợ dự đoán, huấn luyện, kiểm định và xuất YOLO-NAS trên các phiên bản PyTorch hiện tại."
  - q: "Ultralytics có duy trì YOLO-NAS không?"
    a: "Ultralytics hỗ trợ suy luận (inference), kiểm định và xuất. Thư viện này không hỗ trợ huấn luyện."
  - q: "Có thể sử dụng trọng số YOLO-NAS được huấn luyện sẵn (pretrained) cho mục đích thương mại không?"
    a: "Trọng số được huấn luyện sẵn của Deci vẫn chịu các điều khoản phi thương mại từ dự án gốc, bất kể thư viện nào tải chúng. Huấn luyện từ khởi tạo ngẫu nhiên sẽ tránh dùng các checkpoint đó. Chúng tôi cũng dự định phát hành trọng số COCO được huấn luyện từ đầu."
---

YOLO-NAS vẫn là một mô hình phát hiện hữu ích. Nơi phát triển ban đầu của mô hình là [SuperGradients](https://github.com/Deci-AI/super-gradients), dự án này phát hành phiên bản 3.7.1 lần cuối vào tháng 4 năm 2024. LibreYOLO duy trì các tính năng dự đoán, huấn luyện, kiểm định và xuất cho mô hình này.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO hỗ trợ các cỡ S, M và L cho tác vụ phát hiện, đồng thời hỗ trợ cả tác vụ pose. Các checkpoint được tải từ CDN công khai của Deci với tên như `LibreYOLONASs.pt`. LibreYOLO không lưu trữ các checkpoint này, và các điều khoản phi thương mại của Deci vẫn được áp dụng. SuperGradients ghim phiên bản PyTorch trong khoảng từ 1.9 đến 1.13; LibreYOLO yêu cầu phiên bản 2.4 trở lên. Tính năng xuất cho phát hiện hỗ trợ ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch và Core AI. CoreML không được hỗ trợ.

Để huấn luyện mà không dùng checkpoint của Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Chúng tôi cũng dự định huấn luyện YOLO-NAS từ đầu trên COCO và phát hành các trọng số đó.

Cài đặt bằng lệnh `pip install libreyolo`. [Tài liệu YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) có thông tin chi tiết về mô hình. Bài viết [Lựa chọn thay thế SuperGradients](/articles/supergradients-alternative) cung cấp phần giải thích dài hơn.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
