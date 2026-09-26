---
title: Hiệu năng huấn luyện
seo_title: 'Huấn luyện nhanh hơn: CUDA graph, AMP, profiler'
description: >-
  Tăng tốc một lượt huấn luyện: capture bước chạy vào CUDA graph, chọn kiểu dữ
  liệu AMP và dùng profiler tích hợp để tìm nơi thực sự chiếm thời gian.
lead: >-
  Ba đòn bẩy thay đổi tốc độ của một bước huấn luyện: mixed precision, capture
  forward và backward của mạng bằng CUDA graph, cùng phương án xử lý điểm nghẽn
  mà profiler thực sự tìm thấy.
keywords:
  - cuda graph khi huấn luyện
  - tăng tốc huấn luyện
  - huấn luyện mixed precision
  - huấn luyện bfloat16
  - pytorch profiler
  - dataloader bị nghẽn
  - kernel launch overhead
  - mức sử dụng gpu
last_verified: 1.5.0
snippets:
  profile:
    - label: Profile rồi tiếp tục huấn luyện
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Profile một cửa sổ ngắn gồm các bước thực, in kết luận, rồi
        # tiếp tục lượt chạy sau khi gỡ các hook.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: Chỉ đo rồi dừng
      language: bash
      code: |
        # Đặt no_aug_epochs=0 và chỉ chạy đủ số epoch để lấp đầy cửa sổ.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Phân tích sâu kết quả
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Đo trước khi thay đổi bất cứ thứ gì

Ba đòn bẩy bên dưới giải quyết các vấn đề khác nhau, và áp dụng sai đòn bẩy sẽ
không thay đổi gì. Profiler cho biết bạn đang gặp vấn đề nào.

<code-tabs name="profile" />

`profile=True` đo một cửa sổ gồm các bước huấn luyện thực, mặc định bỏ năm bước
rồi đo hai mươi bước, in báo cáo, ghi các artifact và tiếp tục huấn luyện sau
khi gỡ hook. Tùy chọn này không tốn gì khi tắt và bị bỏ qua trong huấn luyện
phân tán.

Báo cáo kết thúc bằng một trong bốn kết luận:

| Kết luận | Ý nghĩa | Đòn bẩy |
|---|---|---|
| `dataloader` | GPU chờ dữ liệu đầu vào | tăng `workers`, dùng `cache="ram"` hoặc `"disk"`, augmentation nhẹ hơn, batch lớn hơn |
| `host / launch` | GPU được cấp việc quá chậm, nhiều kernel nhỏ | batch lớn hơn, CUDA graph, ít đồng bộ host theo bước hơn |
| `compute` | GPU đã bão hòa | AMP hoặc bfloat16, hoặc chấp nhận hiện trạng |
| `memory-pressure` | allocator hoạt động dồn dập, VRAM sát giới hạn | giảm batch; số liệu utilization ở đây không đáng tin cậy |

Số utilization là thời gian kernel bận chia cho thời gian bước không đồng bộ.
Cửa sổ được chủ ý chia đôi: nửa đầu chạy không có đồng bộ bổ sung để kết luận
phản ánh overlap thực, và chỉ nửa sau bao từng phase bằng một lần sync để quy
thời gian GPU về đúng phần. Đồng bộ mọi phase sẽ tạo khoảng trống cho worker
dataloader và che giấu tình trạng thiếu dữ liệu, vì vậy các số liệu thành phần
không bao giờ được dùng để chọn kết luận.

Bốn tệp được ghi vào thư mục lượt chạy: `timeline.html`, tự mở trong trình
duyệt, `profile_trace.json` cho Perfetto hoặc Nsight, `profile_summary.json` và
`profile.json`, tệp độc lập để sao chép và truyền lại cho các lệnh con
`libreyolo profile`.

Cần biết hai điều về `profile run`. Nó đặt `no_aug_epochs=0` vì profiler đo epoch
0, còn lượt chạy ngắn với `no_aug_epochs` mặc định sẽ profile dataloader không
có augmentation nhẹ hơn thay vì dataloader thực sự được dùng khi huấn luyện.
Đồng thời, `--repeat N` báo cáo trung bình và độ lệch chuẩn, điều quan trọng vì
một bước bị giới hạn bởi launch có độ nhiễu đủ cao để một lượt chạy duy nhất gây
hiểu nhầm; lệnh ghi các thư mục theo thử nghiệm `prof_1`, `prof_2` và tiếp tục,
cùng tệp tổng hợp `profile_repeat.json`.

## Mixed precision

`amp=True` là mặc định cho phần lớn family và chạy forward pass trong CUDA
autocast. `amp_dtype` chọn `float16` hoặc `bfloat16`.

<code-tabs name="amp" />

Float16 cần dynamic loss scaling và nhận gradient scaler hoạt động; phạm vi số
mũ rộng hơn của bfloat16 không cần, vì vậy scaler của nó bị tắt. Bốn family được
phân phối với `amp=False` là D-FINE, DEIM, YOLO-NAS và FOMO, còn cài đặt DEIM
được RT-DETRv4 kế thừa. D-FINE nêu rõ lý do: decoder của nó clamp activation ở
65504, giá trị float16 hữu hạn lớn nhất.

Ngữ nghĩa đối số, gồm hành vi của yêu cầu bfloat16 trên phần cứng không hỗ trợ
bfloat16, nằm trong [Siêu tham số](/docs/train/hyperparameters).

## CUDA graph

`cuda_graph=True` capture forward và backward khi huấn luyện của mạng vào CUDA
graph, loại bỏ overhead khởi chạy kernel theo từng bước.

<code-tabs name="graph" />

Luôn an toàn khi truyền flag này. Family, tác vụ hoặc cấu hình không thể capture
sẽ ghi một dòng log và tiếp tục huấn luyện eager mà không thay đổi.

Chỉ mạng được capture. Loss chủ ý giữ ở chế độ eager vì các loss phát hiện chọn
bằng mask boolean, chạy Hungarian matching và rẽ nhánh theo kết quả gán, không
thao tác nào trong số đó có thể được graph ghi lại. Bước optimizer, gradient
clipping, cập nhật EMA và lịch learning rate cũng giữ ở chế độ eager.

Điều đó giới hạn mức cải thiện theo tỷ lệ của mạng trong một bước, và tỷ lệ này
rất khác nhau. Đo trên RTX 5070 Ti ở 640 px, batch 8: mạng chiếm 84 phần trăm
một bước YOLOv9-t, 44 phần trăm bước YOLOv7-b, 31 phần trăm bước YOLOX-t và 26
phần trăm bước RTMDet-t. Hai family cuối dành phần lớn một bước bên trong label
assigner, vì vậy việc capture mạng giúp chúng ít nhất.

### Giá trị thực tế

Điều kiện cho mọi số liệu bên dưới: RTX 5070 Ti, Windows, AMP, mỗi nhánh một
process từ cùng trạng thái đã lưu, phát lại một batch thực để loại dataloader
khỏi vòng lặp, lấy bước nhanh nhất trong 24 bước sau warm-up. Phát hiện ở 640 px,
phân loại ở 224 px. Kích thước batch theo từng dòng.

| Family | Kích thước | Batch | Eager | Graphed | Tăng tốc |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Các số liệu đó cô lập bước GPU. Một lượt tinh chỉnh hoàn chỉnh còn trả chi phí
cho dataloader và xác thực. YOLOv9-t trên tập phát hiện 406 ảnh, 20 epoch, batch
8, 640 px, 4 worker dataloader, trên cùng máy: thời gian thực 428.4 giây ở chế
độ eager so với 367.7 giây khi dùng graph, cải thiện 1.16x, với mAP50-95 bằng
0.6394 ở cả hai nhánh.

Ba yếu tố làm thay đổi các con số này. Batch nhỏ bị giới hạn bởi launch còn batch
lớn bị giới hạn bởi compute, vì vậy RT-DETR-r18 cải thiện 1.19x ở batch 2 và
1.04x ở batch 8. Launch overhead cao nhất trên Windows, còn mức cải thiện trên
Linux xấp xỉ một phần ba đến một nửa bảng. Lượt chạy bị giới hạn bởi dataloader
hoàn toàn không thay đổi thời gian thực, đó là lý do profiler phải chạy trước.

Capture được kích hoạt theo cùng cách ở `amp=False`, nhưng kernel fp32 chạy lâu
hơn nên một bước ít bị giới hạn bởi launch và phần lớn family cải thiện ít hơn.
Trên cùng phần cứng, MobileNetV4-s ở batch 16 thay đổi từ 2.74x dưới AMP thành
3.61x ở fp32, trong khi YOLOv9-t ở batch 8 thay đổi từ 1.99x thành 1.69x và
RT-DETR-r18 ở batch 4 từ 1.12x thành 0.99x.

### Phạm vi áp dụng capture

| Tác vụ | Family |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Mọi trường hợp khác quay về eager với một dòng log: các tác vụ khác trên những
family đó, family không được liệt kê, lượt chạy phân tán và lượt chạy chưng cất.
Lỗi capture ở runtime cũng chuyển phần còn lại của lượt chạy sang eager thay vì
làm lượt chạy thất bại.

Với các detector encoder-decoder gồm D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 và
v4, cùng EC, chỉ backbone và encoder được capture. Decoder của chúng đọc ground
truth để dựng query khử nhiễu tương phản, còn số lượng query đó theo số ground
truth lớn nhất trong batch, vì vậy số token thay đổi giữa các batch.

### Shape

Graph chỉ hợp lệ cho đúng shape đầu vào dùng để capture. Trainer đếm shape của
batch và capture sau khi một shape lặp lại ba lần. Batch có shape khác chạy ở
chế độ eager: batch đa tỷ lệ và batch cuối epoch không đủ kích thước.

Đây là điểm dễ nhầm với các family DETR, vốn mặc định đổi kích thước mọi batch.
Với `multi_scale=True`, lượt chạy ngắn có thể không bao giờ gặp một shape đủ số
lần để capture. Hãy truyền `multi_scale=False` khi mục tiêu là tăng tốc.

YOLOX thay đổi nội dung tính toán trong vùng được capture giữa lượt chạy, bật
nhánh hồi quy L1 khi mosaic đóng tại `no_aug_epochs`. Trainer vô hiệu hóa capture
tại đó và capture lại sau khi shape mới ổn định.

### Số học và bộ nhớ

Phần lớn family tái tạo quỹ đạo loss eager giống từng bit dưới AMP. FOMO và
LingBot-Vision khác ở bit cuối của float32 do thứ tự tính tổng khác. Các detector
deformable attention gồm D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR và EC cũng không
tái tạo được chính lượt chạy eager của mình vì backward đó tích lũy bằng atomic,
còn convolution TF32 chọn thứ tự reduce theo từng lần launch; lượt chạy graph
vẫn nằm trong độ phân tán đó. RTMDet khác khoảng 3e-4 tương đối trên hai trong
139 gradient vì dùng chung convolution của head giữa các cấp pyramid, còn hai
đường dẫn backward cộng ba phần đóng góp theo thứ tự khác. SegFormer có
stochastic depth bên trong vùng capture, vì vậy graph được phát lại lấy random
stream riêng và tương đương về mặt thống kê với eager thay vì giống hệt; manager
ghi điều đó một lần khi capture.

Ở `amp=False`, không có kết quả giống từng bit cho bất kỳ thứ gì trên phần cứng
này, dù có capture hay không. Hai lượt chạy eager YOLOv9-t có seed giống nhau
lệch 36 phần trăm tương đối sau 20 bước, còn YOLOX-t lệch 2.6 phần trăm, vì cuDNN
chọn thuật toán gradient trọng số không xác định cho một số shape convolution
fp32.

Graph đã capture giữ cố định buffer tĩnh cho đầu vào, đầu ra và workspace, vì
vậy VRAM đỉnh tăng xấp xỉ một tập activation bổ sung. Trên các family ở trên,
phân bổ đỉnh thay đổi từ -5 đến +19 phần trăm. Chi phí tương đối lớn nhất trên
các mô hình phân loại nhỏ, vốn có activation nhỏ: ResNet-18 ở 224 px, batch 16,
tăng từ 0.48 GB ở eager lên 0.57 GB khi dùng graph. Nếu điều này đẩy lượt chạy
vượt giới hạn, hãy giảm batch hoặc tắt flag.

## Nội dung liên quan

- Xem [Siêu tham số](/docs/train/hyperparameters) để biết về `batch`, `nbs`,
  `cache` và `workers`.
- Xem [Huấn luyện multi-GPU](/docs/train/multi-gpu), nơi CUDA graph và profiler
  đều không khả dụng.
- Xem [CUDA graph](/docs/reference/cuda-graphs) để biết ma trận hỗ trợ kết hợp
  cho inference và huấn luyện, các điểm tách seam và hợp đồng số học.
