---
title: Câu hỏi thường gặp
seo_title: Câu hỏi thường gặp về LibreYOLO
description: >-
  Câu trả lời ngắn cho những câu hỏi áp dụng trên mọi mô hình LibreYOLO: phần
  cứng, giấy phép, trọng số, thiết bị, huấn luyện, phạm vi xuất và CLI.
lead: >-
  Câu trả lời cho những vấn đề không riêng một họ mô hình. Nội dung dành riêng
  cho từng họ nằm trên trang của họ đó.
keywords:
  - câu hỏi thường gặp libreyolo
  - libreyolo có cần gpu không
  - giấy phép libreyolo
  - trọng số libreyolo ở đâu
  - cli libreyolo
  - dùng libreyolo offline
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Tôi nên bắt đầu với mô hình nào?

YOLOv9 nếu bạn cần detector CNN và RF-DETR nếu cần detector transformer. Cả
hai thuộc cấp flagship, nghĩa là các tính năng được thiết kế và xác thực trên
GPU với chúng trước mọi mô hình khác. Xem [YOLOv9](/docs/models/yolov9) và
[RF-DETR](/docs/models/rf-detr), hoặc [tất cả mô hình](/docs/models) để xem các
lựa chọn còn lại.

## Tôi có cần GPU không?

Không. Mọi mô hình đều chạy trên CPU và mọi nội dung trong
[hướng dẫn bắt đầu nhanh](/docs/quickstart) đều được viết để chạy ở đó. GPU chỉ
thay đổi thời gian huấn luyện và inference video, không quyết định chúng có
hoạt động hay không.

## LibreYOLO chọn thiết bị như thế nào?

Giá trị mặc định là `device="auto"`: dùng CUDA khi PyTorch báo khả dụng, sau
đó dùng Metal Performance Shaders nếu có, còn lại dùng CPU. Để chỉ định thiết
bị, truyền `device` cho mô hình hoặc cho `predict`, `train`, `val` và `export`.
Tham số này nhận `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, một số nguyên như
`0` hoặc chuỗi chữ số; hai dạng cuối được mở rộng thành `cuda:<n>`.

`libreyolo checks` in ra bản build Torch, phiên bản CUDA và cuDNN, cùng mọi GPU
mà nó nhận diện được. Nếu lệnh này không hiển thị CUDA, wheel PyTorch là bản
CPU; phần [cài đặt](/docs/install) hướng dẫn cách thay thế.

## Trọng số tải xuống được lưu ở đâu?

Trong `weights/` tương đối với thư mục làm việc. Một tham chiếu mô hình không
có thành phần thư mục sẽ được phân giải tại đó và tải xuống ở lần dùng đầu
tiên; tham chiếu có thư mục được dùng chính xác như đã viết và không bao giờ
được tải về. Xem [checkpoint và trọng số](/docs/weights).

## Tôi có thể chạy khi không có mạng không?

Có. Tải checkpoint một lần trên máy có kết nối, sao chép thư mục `weights/`
sang máy đích và sẽ không còn truy cập mạng. Một đường dẫn dùng chung chỉ đọc
cũng hoạt động, vì tham chiếu có thư mục được hiểu theo đúng nghĩa đen. Dataset
được phân giải dưới `~/datasets` hoặc `LIBREYOLO_DATASETS_DIR`.

## Tôi có thể dùng LibreYOLO cho mục đích thương mại không?

Mã nguồn dùng giấy phép MIT. Trọng số huấn luyện sẵn là vấn đề riêng: chúng có
thể kế thừa điều khoản từ dự án hoặc dataset nguồn, và các điều khoản này
không đồng nhất ngay cả trong cùng một họ. Giấy phép trên repo Hugging Face cụ
thể là nguồn có thẩm quyền, và mỗi trang mô hình đều có phần giấy phép chép lại
nội dung đó. Khi trọng số bị hạn chế, LibreYOLO in thông báo hạn chế trước khi
bắt đầu tải.

## Tôi có thể nạp checkpoint từ dự án khác không?

Thông thường là có, bằng cách truyền đường dẫn cho `LibreYOLO()`. Các layout
upstream được nhận diện sẽ được chuyển đổi lúc nạp, giữ nguyên số lớp và tên,
đồng thời một checkpoint LibreYOLO được ghi cạnh file nguồn. Phần [nhập trọng
số có sẵn](/docs/migrate) nêu rõ định dạng được nhận diện và trường hợp cần
script chuyển đổi.

## Vì sao train báo NotImplementedError?

Vì họ đó chỉ cung cấp inference và ngoại lệ nêu rõ lý do. Predict, validate và
export ở nơi được hỗ trợ vẫn hoạt động; LibreYOLO không có vòng lặp huấn luyện
cho kiến trúc đó. Cấp hỗ trợ ở đầu trang mô hình cho bạn biết điều này trước
khi thử. Xem [khái niệm cốt lõi](/docs/concepts).

## val trả về gì?

Một dictionary thuần túy, không phải object. Các khóa detection gồm
`metrics/precision`, `metrics/recall`, `metrics/mAP50` và
`metrics/mAP50-95`. Những tác vụ khác trả về các khóa phù hợp, chẳng hạn
`metrics/accuracy_top1` cho classification hoặc `metrics/PQ`, `metrics/SQ` và
`metrics/RQ` cho panoptic segmentation.

## Làm sao chạy trên thư mục, video hoặc webcam?

Truyền đối tượng đó làm source. Đường dẫn file là một ảnh, thư mục là mọi ảnh
bên trong, đường dẫn video là video, số nguyên là chỉ mục webcam, còn URL RTSP,
RTMP, TCP, UDP hoặc HLS là stream trực tiếp. File `.streams` liệt kê nhiều
source cùng lúc. Source trực tiếp cần `stream=True`, trả về một `Results` cho
mỗi frame thay vì tạo danh sách; cờ này cũng đáng dùng cho video dài và thư
mục lớn. Chỉ URL trang YouTube cần gói bổ sung `libreyolo[stream]`.

## Làm sao chỉ giữ một số lớp?

Truyền `classes` cho `predict` cùng các chỉ mục lớp mong muốn, ví dụ
`classes=[0, 2]`. `conf` đặt ngưỡng độ tin cậy, mặc định `0.25`, còn `max_det`
giới hạn số detection mỗi ảnh, mặc định `300`.

## CLI dùng cờ hay cặp key=value?

Key và value nối bằng dấu bằng cho mọi lệnh:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` nhận đường dẫn hoặc tên ngắn dạng `family-size`, có thể kèm hậu tố tác
vụ, và `libreyolo models` liệt kê mọi giá trị hợp lệ. Các lệnh chẩn đoán và kiểm
kê cũng nhận `--json`, in cùng dữ liệu dưới dạng object máy đọc được ra stdout.

## Mọi mô hình đều có thể xuất sang mọi định dạng không?

Không. Phạm vi hỗ trợ tùy theo họ và tác vụ, không đồng nhất, và mỗi định dạng
có gói bổ sung riêng cần cài đặt. Mỗi trang mô hình có ma trận xuất của họ đó;
[phần xuất](/docs/export) trình bày các định dạng.

## segment, semantic và panoptic khác nhau thế nào?

Đây là ba tác vụ riêng biệt. `segment` tạo một mask cho mỗi object được phát
hiện. `semantic` gán lớp cho mọi pixel mà không tách thành instance. `panoptic`
gán chính xác một nhãn cho mỗi pixel, kết hợp các vật thể đếm được với vùng
không có hình dạng rõ ràng. Chúng dùng ground truth, trường kết quả và metric
khác nhau, và một họ hỗ trợ tác vụ nào xuất hiện trong danh sách tác vụ của họ.

## Làm sao huấn luyện trên các lớp của riêng tôi?

Viết YAML dataset với `train`, `val` và `names`. Nhãn nằm cạnh ảnh trong cây
`labels/` song song, mỗi ảnh có một file `.txt`, với tọa độ chuẩn hóa. `nc` là
tùy chọn và phải khớp với `names` nếu được cung cấp. Trước tiên hãy chạy
`libreyolo doctor <data.yaml>`: lệnh kiểm tra lỗi dataset và thoát với mã khác
0 khi tìm thấy lỗi, nên có thể dùng làm cổng CI.

## Vì sao lúc nạp lại xuất hiện cảnh báo metadata?

Vì checkpoint không mang đầy đủ metadata v1.0. Quá trình nạp tiếp tục qua
đường tương thích và cảnh báo nêu chính xác các khóa còn thiếu. Chạy
`libreyolo metadata path=<file>` để xem nội dung hiện có, và xem [checkpoint và
trọng số](/docs/weights) để biết schema yêu cầu gì.

## Một import ngừng hoạt động sau khi nâng cấp. Điều gì đã thay đổi?

Hai tên lớp được đổi để nhất quán: `LibreYOLORTDETR` thành `LibreRTDETR` và
`LibreYOLORFDETR` thành `LibreRFDETR`. Tên cũ vẫn phân giải được và phát
`DeprecationWarning` trỏ đến tên mới, nên mã hiện có vẫn chạy trong lúc bạn cập
nhật.
