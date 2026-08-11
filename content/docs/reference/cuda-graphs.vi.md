---
title: CUDA graph
seo_title: Ma trận hỗ trợ CUDA graph của LibreYOLO
description: >-
  Những họ capture forward khi dự đoán và forward cùng backward khi huấn luyện,
  các số liệu được bảo đảm thế nào, nơi capture được tách và lý do họ không hỗ
  trợ phát sinh lỗi.
lead: >-
  CUDA graph ghi một lần thực thi chuỗi kernel cố định và phát lại dưới dạng một
  lần khởi chạy. LibreYOLO capture suy luận trên 39 họ đã xác minh và huấn luyện
  trên 24 họ, luôn theo từng họ, luôn sau phép kiểm tra tương đương từng bit và
  không bao giờ âm thầm dùng luồng dự phòng.
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - ma trận hỗ trợ cuda graph
  - huấn luyện torch cuda graph
  - capture_error_mode thread_local
  - cuda graph giống từng bit
last_verified: 1.5.0
verification: >-
  Danh sách họ suy luận được suy ra từ ma trận CAPTURABLE trong
  tests/e2e/test_cuda_graph_families.py ở v1.5.0. Danh sách họ huấn luyện, lớp
  tương đương và thời gian lấy từ docs/training_cuda_graphs.md. API và
  NotImplementedError lấy từ BaseModel._require_cuda_graph_support,
  cuda_graph_scope và capture_graph trong libreyolo/models/base/model.py, cùng
  biến lớp SUPPORTS_CUDA_GRAPH. Các điểm tách đường nối được đọc từ phần ghi đè
  _get_graph_runner trong các họ depth_anything3, birefnet, ppocr, sam và
  sensenova cùng libreyolo/models/base/detr_cuda_graph.py. capture_error_mode
  lấy từ libreyolo/models/base/cuda_graph.py và
  libreyolo/training/cuda_graph.py. Luồng dự phòng huấn luyện lấy từ
  libreyolo/training/trainer.py và cờ --cuda-graph từ
  libreyolo/cli/commands/train.py.
meta:
  - label: Họ suy luận
    value: '39'
  - label: Họ huấn luyện
    value: '24'
  - label: Cờ suy luận
    value: predict(cuda_graph=True)
    mono: true
  - label: Cờ huấn luyện
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Dự đoán
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True capture ở lần dùng đầu tiên cho mỗi shape đầu vào.
        # "auto" đợi shape lặp lại trước khi chịu chi phí capture.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Huấn luyện
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Huấn luyện từ CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Nội dung được capture

Graph ghi chuỗi kernel cố định và địa chỉ bộ nhớ được đọc, ghi. Nó không ghi giá
trị, shape hay luồng điều khiển. Phát lại là một lần khởi chạy thay vì hàng trăm,
nên mức tăng lớn nhất trên mạng nhỏ với kích thước batch nhỏ, nơi mỗi bước bị chi
phối bởi chi phí khởi chạy thay vì phép toán.

Hai điểm vào capture lượng công việc khác nhau.

| | Bên trong graph | Eager |
|---|---|---|
| Suy luận | Forward của mạng, `model._forward(x)` | Tiền xử lý, NMS, toàn bộ hậu xử lý |
| Huấn luyện | Forward và backward của mạng | Loss, bước optimizer, cắt gradient, EMA, lịch LR |

Cả NMS lẫn loss phát hiện đều không phải ứng viên. Cả hai chọn bằng mask boolean,
chạy ghép Hungarian hoặc assigner rồi rẽ nhánh theo kết quả, đúng là những gì
graph không thể ghi. Giữ chúng bên ngoài giúp capture an toàn chứ không phải hạn
chế cần khắc phục.

<code-tabs name="usage" />

`cuda_graph` chấp nhận ba giá trị khi dự đoán. `False` là mặc định. `True` capture
ở lần đầu gặp mỗi shape đầu vào. `"auto"` đợi shape lặp lại, nên công việc chạy
một lần và thay đổi shape không chịu chi phí capture không được dùng lại.
`capture_graph(imgsz=None, batch=1, dtype=None)` chuyển chi phí ra khỏi yêu cầu
đầu, `graph_info()` báo cáo graph đã capture cùng số lần phát lại, còn
`release_graphs()` giải phóng chúng.

Khi huấn luyện, cờ là boolean thông thường, `--cuda-graph` trên CLI. Xem
[hiệu năng dự đoán](/docs/predict/performance) và
[hiệu năng huấn luyện](/docs/train/performance) để biết các điều khiển liên quan.

## Hỗ trợ suy luận

Hỗ trợ theo từng họ, được khai báo qua biến lớp `SUPPORTS_CUDA_GRAPH`; họ chỉ
được đánh dấu sau khi capture và phát lại giống từng bit trên hai đầu vào thử lấy
từ các phân phối khác nhau. Ma trận tương đương dùng chung đó bao quát 39 họ trên
chín tác vụ.

| Tác vụ | Các họ |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Một số họ xuất hiện dưới nhiều tác vụ, nên ma trận chạy nhiều hàng hơn số họ riêng
biệt. Ba họ khác capture qua luồng mã riêng theo họ với phép kiểm thử chuyên dụng
thay vì ma trận dùng chung và không nằm trong 39 họ: PP-OCR, SAM và SenseNova.

Quá trình xác minh theo từng bit, không phải xấp xỉ. Phiên bản giao thức trước
đánh giá mức tương đương theo độ lớn tương đối và hạ nhầm ba họ hoạt động tốt là
YOLOX, EfficientNetV2 và YOLOv7, có chênh lệch eager so với graph khoảng 1e-7
nhưng vẫn giống từng bit trên đầu vào thử quan trọng.

## Hỗ trợ huấn luyện

Capture huấn luyện tăng từ hai lên 24 họ trong bản phát hành này, trên năm tác vụ.

| Task | Families |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Mọi trường hợp khác huấn luyện eager: tác vụ khác trên cùng các họ đó, họ không
được liệt kê, lượt chạy phân tán và chưng cất. Capture cũng bị bỏ qua khi shape
còn mới vì luồng huấn luyện đợi shape đầu vào lặp lại ba lần trước khi capture,
nghĩa là `multi_scale=True` có thể không bao giờ capture.

## Hai cách xử lý khác nhau cho họ không được hỗ trợ

Luồng suy luận phát sinh lỗi. `predict(cuda_graph=True)` trên họ chưa bật sẽ phát
sinh `NotImplementedError` nêu tên họ, thay vì chạy eager và khiến bạn tin rằng
đã tăng tốc dù không có. Lý do là capture lỗi không thất bại rõ ràng: phát lại
forward thực hiện thao tác không thể capture sẽ âm thầm trả về số sai, nên hỗ trợ
phải là khẳng định rõ theo từng họ thay vì thử rồi dùng luồng dự phòng.

Luồng huấn luyện ghi log. Luôn an toàn khi truyền `train(cuda_graph=True)`; họ,
tác vụ hoặc cấu hình không thể capture sẽ ghi một dòng và huấn luyện eager không
thay đổi. Capture thất bại giữa lượt chạy cũng chuyển phần còn lại sang eager
thay vì hủy. Sự bất đối xứng là có chủ ý: dự đoán là lời gọi có thể sửa tại chỗ
gọi, còn lượt huấn luyện không nên chết ở giờ thứ sáu vì tối ưu tùy chọn.

## Tách tại đường nối

Một số họ không thể capture toàn bộ vì một giai đoạn thực sự làm điều graph không
thể ghi. Thay vì loại họ, capture được tách tại đường nối đã xác minh: phần có thể
capture được phát lại, phần còn lại chạy eager, và đầu ra kết hợp giống với chạy
tất cả ở eager.

| Họ | Được capture | Eager và lý do |
|---|---|---|
| Depth Anything 3 | Mạng | Bước sky là công việc hiển thị phía host sau forward |
| BiRefNet | Encoder, `forward_enc` | Decoder có `deform_conv2d` phát lại thành kết quả khác khi capture |
| PP-OCR | Giai đoạn phát hiện, `forward_det` | Nhận dạng vì độ rộng crop thay đổi theo dòng |
| SAM | Bộ mã hóa ảnh | Luồng prompt chạy nhiều lần cho mỗi lần mã hóa |
| SenseNova | Vision tower | Sinh tự hồi quy với KV cache tăng mỗi bước |
| Detector encoder-decoder | Backbone và encoder | Decoder và tiêu chí Hungarian |

Điểm tách BiRefNet đáng đọc kỹ: hành vi sai của `deform_conv2d` khi capture tái
hiện trên lời gọi trần ngoài mọi mô hình. Việc thay bằng dạng PyTorch thuần bị từ
chối vì cũng làm dịch chuyển dự đoán eager, mà số liệu eager là giao diện.

Trường hợp encoder-decoder bao quát D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 và EC. Decoder tạo query khử nhiễu tương phản từ ground truth, và số
query lấy từ số ground-truth lớn nhất trong batch, nên số token của decoder thay
đổi giữa các batch. Đây là điều graph không thể chấp nhận. Backbone cộng encoder
chiếm khoảng một phần năm đến một phần tư bước của các họ này, vì vậy chúng nằm
cuối bảng tăng tốc.

PP-OCR capture một graph cho mỗi shape đầu vào phát hiện, bị giới hạn bởi mức tối
đa bộ nhớ đệm của runner, và trả về kết quả eager khi không có phạm vi capture hoạt động.

## Số học

Hầu hết các họ giống từng bit; khi không giống, lý do được nêu rõ. Ở bước 0 của
huấn luyện, loss giống từng bit cho cả 24 họ và không buffer BatchNorm nào khác;
so sánh gradient là phần phân tách các loại.

| Loại | Các họ | Ý nghĩa |
|---|---|---|
| Chính xác | Phần lớn 24 họ | Mọi gradient giống từng bit |
| 1 ULP | fomo, lingbotvision | Bit cuối của float32, tương đối khoảng 1e-7, do thứ tự tính tổng khác |
| Nhiễu eager | Dòng DETR | Graph khác eager không nhiều hơn hai lượt eager khác nhau |
| Làm tròn số thực | rtmdet | 137 trong 139 gradient giống từng bit, hai gradient khác khoảng 3e-4 |
| Luồng RNG riêng | segformer | Stochastic depth nằm trong vùng đã capture |

Cần hiểu đúng loại nhiễu eager. Với các họ đó, hai lượt eager có cùng seed đã
khác nhau, nên giống từng bit không phải tiêu chuẩn lượt graph không vượt qua mà
là tiêu chuẩn không lượt nào vượt qua. Điều này phổ biến hơn ở `amp=False`, nơi
mức không xác định tương đối 3.2e-7 đo được trong gradient trọng số fp32 tích lũy:
hai lượt eager YOLOv9-t cùng seed phân kỳ 36 phần trăm sau 20 bước và tắt TF32
không khắc phục.

## Pin memory

Capture chạy với `capture_error_mode="thread_local"`. Trong chế độ `"global"`
mặc định của PyTorch, thread pin-memory DataLoader chuẩn bị batch tiếp theo gọi
`cudaHostAlloc`, vừa làm capture đang chạy mất hiệu lực vừa bị capture làm hỏng,
nên lượt chạy chết khi lấy batch kế tiếp với lỗi phát sinh từ trong thread
pin-memory. Cặp hành vi đó đã xuất hiện hai lần trong chiến dịch huấn luyện thực
trước khi được chẩn đoán.

Chế độ thread-local chỉ giới hạn thread capture. Thread pin không bao giờ chạm
luồng capture, nên ngay từ đầu không thao tác nào của nó thuộc graph. Huấn luyện
còn tạm thời thay bằng lớp con `torch.cuda.CUDAGraph` buộc chế độ này vì
`make_graphed_callables` không cung cấp đối số cho nó, dưới một lock để hai
capture đồng thời không thể làm bản thay thế bị giữ lại.

## Mức tăng thực tế

Đo trên RTX 5070 Ti dưới AMP, mỗi nhánh một tiến trình, phát lại một batch thực
để loại dataloader khỏi vòng lặp, lấy bước nhanh nhất trong 24 bước sau warm-up.
Phát hiện ở 640 px, phân loại ở 224 px.

| Họ | Batch | Mức tăng tốc |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Tất cả trường hợp khác | thay đổi | 1.04x đến 1.26x |

Toàn bộ lượt chạy tăng ít hơn vì graph không thể tăng tốc dataloader hoặc đánh
giá. Lượt tinh chỉnh YOLOv9-t 20 epoch trên 406 ảnh giảm từ 428.4 giây xuống
367.7 giây, tăng đầu cuối 1.16 lần, với mAP50-95 giống nhau là 0.6394 ở cả hai
nhánh và loss theo epoch giống nhau.

Mức trần được quyết định bởi phần thời gian mạng chiếm trong một bước. Trên cùng
phần cứng ở 640 px và batch 8, tỉ lệ là 84 phần trăm cho YOLOv9-t nhưng chỉ 26
phần trăm cho RTMDet-t, vốn dành phần lớn bước trong label assigner. Chi phí khởi
chạy cao nhất trên Windows, nên mức tăng trên Linux khoảng một phần ba đến một
nửa bảng này; lượt chạy bị giới hạn bởi dataloader không thay đổi thời gian thực.
Bộ nhớ đỉnh dao động từ thấp hơn 5 phần trăm đến cao hơn 19 phần trăm.

## Lưu ý

Graph ghi địa chỉ chứ không phải giá trị, nên mọi thao tác di chuyển tham số đều
loại graph. Thay đổi thiết bị qua `predict(device=...)`, lượng tử hóa và giải
lượng tử hóa đều làm graph đã capture mất hiệu lực.

Kích thước batch quan trọng hơn họ: RT-DETR-r18 tăng 1.19 lần ở batch 2 và 1.04
lần ở batch 8 vì batch lớn bị giới hạn bởi tính toán và có ít chi phí khởi chạy
để loại bỏ hơn.

Bộ kiểm thử tương đương suy luận chạy khi chưa cài gói `kernels` tùy chọn, nên
không bao quát độ an toàn capture khi kernel Hub đã biên dịch hoạt động. Đặt
`LIBREYOLO_HUB_KERNELS=0` để loại chúng khi cô lập vấn đề capture. Xem
[kernel](/docs/reference/kernels).

