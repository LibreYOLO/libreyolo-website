---
title: Cách chạy Depth Anything V2 đơn giản nhất với LibreYOLO
description: Depth Anything V2 tạo ra các bản đồ độ sâu đơn ảnh hàng đầu hiện nay. Sau đây là cách chạy mô hình chỉ với hai dòng lệnh bằng LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Cách đơn giản nhất để chạy Depth Anything V2 là gì?"
    a: "Chỉ cần hai dòng với LibreYOLO: tải LibreDepthAnythingV2l-depth.pt theo tên rồi gọi predict với save=True. Trọng số tự tải về trong lần dùng đầu tiên, còn chuẩn hóa, colormap và thiết lập thiết bị được thư viện xử lý cho bạn trên CUDA, Apple Silicon hoặc CPU thông thường."
  - q: "Tôi có thể dùng Depth Anything V2 cho mục đích thương mại không?"
    a: "Chỉ encoder Small dùng Apache 2.0. Base, Large và Giant dùng CC-BY-NC-4.0, vì vậy các checkpoint mạnh nhất chỉ dành cho mục đích phi thương mại. Giấy phép upstream vẫn áp dụng bất kể thư viện nào tải trọng số."
  - q: "LibreYOLO có hỗ trợ huấn luyện Depth Anything V2 không?"
    a: "Không. Việc huấn luyện vẫn diễn ra trong repo gốc; LibreYOLO hỗ trợ suy luận (inference), video và đánh giá (validation) cho tác vụ độ sâu."
---

![Guggenheim Bilbao bên cạnh bản đồ độ sâu của công trình](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 hiện tạo ra một số bản đồ độ sâu đơn ảnh tốt nhất.

Nếu bạn quen với YOLO, repo chính thức không phải là lựa chọn dễ bắt đầu nhất. Để tạo một bản đồ độ sâu, bạn cần thực hiện một số bước thủ công:

* tự tải checkpoint phù hợp rồi đặt vào đúng thư mục,

* chọn cấu hình khớp với encoder (`encoder="vitl"`, `features=256`, `out_channels=...`),

* tự viết bước chuẩn hóa và tô màu,

* xử lý việc thiết lập thiết bị để chạy trên Mac hoặc CPU, thay vì chỉ trên CUDA,

* và làm quen với API suy luận (inference) khác hẳn phần còn lại trong stack của bạn.

Không bước nào khó. Chúng chỉ tạo thêm trở ngại, và bạn có thể bỏ qua tất cả.

LibreYOLO tải cùng bộ trọng số Depth Anything V2 và cho phép bạn dùng tác vụ độ sâu chỉ với một lệnh gọi `predict()`. Chỉ định tên mô hình, thư viện sẽ tự tải ở lần dùng đầu tiên:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Vậy là xong. Nếu đã dùng API YOLO tiêu chuẩn, bạn sẽ thấy cách dùng quen thuộc: tải mô hình theo tên, gọi `predict`, rồi để `save=True` ghi hình ảnh trực quan hóa ra tệp. Đây chính xác là lệnh gọi bạn dùng cho phát hiện đối tượng, chỉ khác ở chỗ kết quả có bản đồ độ sâu thay vì các bounding box. Colormap, bước chuẩn hóa và việc thiết lập thiết bị đều được thư viện xử lý. Cách chạy giống nhau trên Linux với CUDA, trên Mac dùng Apple Silicon hoặc trên CPU thông thường. Không cần sửa code.

Từ đó, bạn có thể dùng chính lệnh gọi này để lấy các giá trị độ sâu thô và xử lý trực tiếp. Việc huấn luyện Depth Anything V2 vẫn diễn ra trong repo gốc; LibreYOLO hỗ trợ inference, video và validation.

Cùng một lệnh gọi trên những khung cảnh rất khác nhau:

![Ảnh mẫu parkour bên cạnh bản đồ độ sâu: những người đang nhảy ở phía trước nổi bật trước các bức tường bê tông phía sau.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Ảnh chụp từ trên cao vịnh La Concha ở Donostia bên cạnh bản đồ độ sâu: thuyền và bờ biển có vẻ ở gần, vùng nước rộng lùi xa.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![Sân trong có hàng cột của Casa de Juntas de Gernika bên cạnh bản đồ độ sâu, cho thấy kiến trúc lùi dần về phía xa.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Đám đông tại lễ hội ban đêm ở Plaza de la Constitucion, Donostia, bên cạnh bản đồ độ sâu. Những hàng người ở gần nổi bật trước mặt tiền được chiếu sáng phía sau.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Lưu ý về trọng số: LibreYOLO lưu trữ các checkpoint Depth Anything V2 đã chuyển đổi và tự tải chúng ở lần dùng đầu tiên, nên bạn không cần tự tải về. Giấy phép upstream vẫn áp dụng: encoder Small dùng Apache-2.0, còn Base, Large và Giant dùng CC-BY-NC-4.0 (phi thương mại), vì vậy các checkpoint mạnh chỉ dành cho mục đích phi thương mại. Bạn muốn làm việc offline hoặc tự chuyển đổi checkpoint? Script chuyển đổi chạy một lần vẫn có sẵn:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Dùng thử

```bash
pip install libreyolo
```

LibreYOLO là thư viện thị giác máy tính đầy đủ tính năng nhất mà bạn có thể cài bằng pip. Một API quen thuộc hỗ trợ danh sách mô hình hàng đầu ngày càng mở rộng: phát hiện đối tượng (RF-DETR, D-FINE, DEIM), phân đoạn, tư thế, bounding box xoay, và giờ là độ sâu đơn ảnh với Depth Anything V2, đồng thời hỗ trợ huấn luyện và validation cho các tác vụ tương ứng. Thư viện chạy trên mọi hệ điều hành lớn, bằng GPU hoặc CPU, và dùng giấy phép MIT hoàn toàn, cho phép bạn phát hành sản phẩm thương mại mà không kèm điều kiện ràng buộc.

Đánh sao cho dự án trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
