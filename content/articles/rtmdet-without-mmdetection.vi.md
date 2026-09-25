---
title: "Chạy RTMDet không cần mmdetection"
description: "RTMDet là một trong những bộ phát hiện vừa nhanh vừa chính xác nhất. Cách cài đặt chính thức bị lỗi với mọi phiên bản PyTorch phát hành trong hai năm qua. Đây là giải pháp thay thế."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Vì sao mmcv không cài được với các phiên bản PyTorch mới?"
    a: "Wheel mmcv mới nhất, phiên bản 2.2.0 phát hành vào tháng 4 năm 2024, chỉ có binary dựng sẵn đến torch 2.4 / CUDA 12.1. Với mọi phiên bản PyTorch mới hơn, mmcv phải biên dịch các ops C++/CUDA từ mã nguồn, mất từ 10 đến 30 phút và cần CUDA toolkit, nvcc tương ứng cùng trình biên dịch C++ tương thích. Đây là nguyên nhân gây ra các lỗi như thiếu module mmcv._ext."
  - q: "Tôi có thể chạy RTMDet mà không cài mmdetection không?"
    a: "Có. LibreYOLO tải trọng số RTMDet đã chuyển đổi từ các checkpoint OpenMMLab gốc, và kết quả suy luận (inference) tương đương từng bit với mmdetection khi dùng cùng checkpoint. Tải LibreRTMDets.pt theo tên, thư viện sẽ tự động tải về; cả năm kích thước, từ Tiny đến X, đều chạy ở 640 px."
  - q: "RTMDet có miễn phí cho mục đích thương mại không?"
    a: "Có. MMDetection và RTMDet dùng giấy phép Apache 2.0, trọng số đã chuyển đổi giữ nguyên điều khoản Apache 2.0, còn mã của LibreYOLO dùng giấy phép MIT. Không có hạn chế thương mại."
  - q: "Khi nào mmdetection vẫn là lựa chọn phù hợp?"
    a: "Nếu bạn cần huấn luyện hoặc tinh chỉnh (fine-tuning) RTMDet bằng pipeline tăng cường dữ liệu (data augmentation) đầy đủ của MMDetection, hoặc đã làm việc sâu trong hệ sinh thái OpenMMLab và mmcv đang hoạt động ổn định. Với inference và triển khai, LibreYOLO là lựa chọn tốt hơn."
---

RTMDet là bộ phát hiện thời gian thực của OpenMMLab, đạt độ chính xác COCO cao trong khi vẫn đủ nhanh để triển khai. Kiến trúc gọn gàng. Cài đặt thì không.

Để chạy RTMDet từ nguồn chính thức, bạn phải lắp ghép stack OpenMMLab gồm bốn lớp:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Các phiên bản được ghim khá chặt. mmdet 3.3.0 yêu cầu `mmcv < 2.2.0`, nhưng `mim install mmcv>=2.0.0` lại cài phiên bản 2.2.0 mà không gặp trở ngại nào, sau đó lỗi ở bước kiểm tra assertion khi chạy. Bạn phải tự ghim phiên bản.

Sau đó là vấn đề lớn hơn. Wheel mmcv mới nhất là phiên bản 2.2.0, phát hành vào tháng 4 năm 2024 và không được cập nhật kể từ đó. Phiên bản này chỉ có binary dựng sẵn đến **torch 2.4 / CUDA 12.1**. Lệnh `pip install torch` mới hiện nay cài PyTorch 2.12. Khoảng cách này khiến mmcv phải biên dịch các ops C++/CUDA từ mã nguồn: mất từ 10 đến 30 phút, cần CUDA toolkit tương ứng, `nvcc` và trình biên dịch C++ tương thích. Đây là những lỗi được ghi nhận thường xuyên:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- các ops đã biên dịch không được tạo hoặc không tương thích,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` trên mọi card RTX 30-series,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- stack buộc bạn phải quay lại Python 3.8,

* lỗi segmentation fault khi import do phiên bản GCC không tương thích.

FAQ của chính OpenMMLab hướng dẫn cài mmcv bằng pip thay vì mim để tránh bẫy phiên bản. Trang Get Started của họ lại hướng dẫn dùng mim. Cả hai tài liệu đều là tài liệu chính thức. Chúng mâu thuẫn với nhau.

Khi stack đã chạy, để inference, bạn vẫn phải chọn tệp config có tên mã hóa recipe huấn luyện, rồi tải riêng checkpoint có tên chứa hash và kết nối chúng qua registry mà trước đó bạn phải tìm hiểu.

LibreYOLO thay thế toàn bộ quy trình đó:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # tự động tải về ở lần chạy đầu tiên
results = model("image.jpg", save=True)
```

Không cần `mim`, không cần ma trận phiên bản gồm bốn package, không cần biên dịch từ mã nguồn, không cần tệp config, không cần tìm checkpoint theo hash, không cần ghim Python 3.8. Trọng số được chuyển đổi từ các checkpoint OpenMMLab gốc và kết quả suy luận tương đương từng bit với mmdetection khi dùng cùng checkpoint.

Có năm kích thước: Tiny, Small, Medium, Large và X. Tất cả chạy ở 640 px.

```python
print(results[0].boxes.xyxy)  # tọa độ xyxy
print(results[0].boxes.conf)  # điểm độ tin cậy
```

## Khi nào nên chọn bản gốc

Nếu bạn cần huấn luyện hoặc fine-tuning RTMDet bằng pipeline data augmentation đầy đủ của MMDetection, hoặc đã làm việc sâu trong hệ sinh thái OpenMMLab và mmcv đang hoạt động ổn định, hãy tiếp tục dùng nó. LibreYOLO phù hợp hơn cho inference và triển khai.

## Lưu ý về giấy phép

MMDetection và RTMDet dùng giấy phép Apache 2.0. Mã của LibreYOLO dùng giấy phép MIT. Trọng số giữ nguyên điều khoản Apache 2.0 từ dự án gốc. Không có hạn chế thương mại.

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần sửa code. Một API hỗ trợ RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, phân đoạn, tư thế, độ sâu và nhiều tính năng khác.

Đánh sao trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
