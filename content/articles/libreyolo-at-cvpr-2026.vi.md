---
title: "LibreYOLO tại CVPR 2026: Edge AI trên Hailo-8L và Snapdragon"
description: "Tại CVPR 2026 ở Denver, nhóm Jabra và IT University of Copenhagen đã dùng LibreYOLOXs làm mô hình ví dụ trong tutorial \"Edge AI in Action\", chạy trên Hailo-8L và Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "LibreYOLOXs chạy nhanh đến mức nào trên phần cứng biên?"
    a: "Trong tutorial CVPR 2026, LibreYOLOXs chạy ở mức 19.0 ms mỗi frame (52.6 FPS) trên Raspberry Pi 5 với Hailo-8L, và 6.69 ms trên HTP/DSP của Qualcomm QCS6490 sau khi lượng tử hóa INT8."
  - q: "Làm cách nào để triển khai mô hình LibreYOLO lên bộ tăng tốc biên?"
    a: "Xuất sang ONNX bằng lệnh export của LibreYOLO, sau đó biên dịch cho thiết bị đích: Hailo Dataflow Compiler tạo tệp HEF cho chip Hailo, còn bộ công cụ SNPE / QAIRT / AI Hub hỗ trợ Qualcomm Snapdragon. Đây chính xác là pipeline mà tutorial CVPR đã trình bày từ đầu đến cuối."
---

![CVPR 2026, Denver, Colorado, June 3-7](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO đã xuất hiện tại CVPR 2026.

CVPR được công nhận rộng rãi là hội nghị thị giác máy tính danh giá nhất thế giới. Năm nay, hội nghị diễn ra ở Denver, nơi một nhóm từ Jabra và IT University of Copenhagen tổ chức tutorial có tên "Edge AI in Action: Mastering On-Device Inference". Họ chọn LibreYOLOXs làm mô hình ví dụ để đưa khả năng phát hiện đối tượng lên các chip biên.

## Họ đã xây dựng những gì

Tutorial trình bày toàn bộ pipeline biên từ đầu đến cuối. Bắt đầu với mô hình LibreYOLOXs. Xuất mô hình sang ONNX. Sau đó biên dịch ONNX cho hai loại phần cứng rất khác nhau: bộ tăng tốc Hailo-8L và Qualcomm Snapdragon.

## Thời gian thực trên Hailo-8L

Họ biên dịch ONNX sang HEF bằng Hailo Dataflow Compiler, sau đó chạy trên Raspberry Pi 5 với Hailo-8L AI HAT.

![LibreYOLOXs chạy trên Raspberry Pi 5 với Hailo-8L, phát hiện người và túi xách trên một con phố đông đúc](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Mô hình chạy ở mức 19.0 ms mỗi frame, 52.6 FPS, trên Raspberry Pi.

## Thời gian thực trên Snapdragon

Ở phía Qualcomm, họ lượng tử hóa LibreYOLOXs xuống INT8 và chạy mô hình qua bộ công cụ SNPE, QAIRT và AI Hub, với benchmark trên QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs phát hiện trực tiếp TV, laptop, bàn phím, chuột và điện thoại di động trong ứng dụng EdgeVision AI trên điện thoại Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Kết quả là 6.69 ms trên HTP/DSP.

## Cảm ơn

Xin gửi lời cảm ơn chân thành đến Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera và Shan Ahmed Shaffi đã giới thiệu dự án.

- Tutorial và slide: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Các mô hình LibreYOLOXs đã lượng tử hóa cho Qualcomm: [trên Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần sửa code. Một API hỗ trợ YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, phân đoạn, tư thế, độ sâu và nhiều tính năng khác.

Đánh sao dự án trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
