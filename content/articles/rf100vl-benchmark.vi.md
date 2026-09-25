---
title: "Benchmark RF100-VL: khả năng tổng quát hóa của các mô hình LibreYOLO"
description: Kết quả RF100-VL đo lường khả năng tổng quát hóa của YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter và RF-DETR trên 100 tập dữ liệu thực tế sau khi tinh chỉnh.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL đặt câu hỏi một detector thích ứng tốt đến đâu ngoài COCO. Mỗi mô hình được tinh chỉnh riêng trên 100 tập dữ liệu rất khác nhau, bao gồm ảnh hồng ngoại, X-quang, hiển vi, thể thao, ảnh phổ vô tuyến, vật thể nhỏ và trò chơi điện tử. Chúng tôi benchmark 17 cấu hình LibreYOLO: **tổng cộng 1700 lần tinh chỉnh.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Kết quả

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L dẫn đầu phần so sánh này với **61.76**. Tiếp theo là M với 61.13, rồi S với 60.41. Bốn kết quả RF-DETR khá sát với [các con số Roboflow đã công bố](https://rfdetr.roboflow.com/latest/learn/benchmarks/), đây là một cách hữu ích để xác thực việc huấn luyện RF-DETR trong LibreYOLO.

Kích thước không dự đoán được mọi kết quả. YOLO-NAS-S và M gần như hòa nhau ở mức 58.00 và 57.99. EdgeCrafter-M đạt 57.93, cao hơn S ở mức 55.99 và L ở mức 56.11. Chúng tôi dự định tìm hiểu các kết quả giảm này. Đợt chạy này không phải thử nghiệm scaling có kiểm soát: độ phân giải, trọng số pretrained, precision và recipe khác nhau.

RF-DETR-S với LoRA trên backbone đạt 57.19, so với 60.41 khi tinh chỉnh toàn phần. Tinh chỉnh toàn phần thắng trên 95 tập dữ liệu. Thời gian trung vị được ghi nhận chỉ lâu hơn bảy phút, trong khi tổng thời gian các job gần như không đổi.

Các dòng YOLOv9 có trước những bản sửa quan trọng trong huấn luyện. Kết quả M bất thường giúp phát hiện PGI bị thiếu, quy ước letterbox khác biệt và giới hạn huấn luyện 100 nhãn. Các con số được lưu trữ mô tả đúng code đã chạy. Chúng không phải là kết luận về kiến trúc YOLOv9.

Chọn một mô hình bên dưới để xem điểm số của mô hình đó trên cả 100 tập dữ liệu.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Phương pháp

Với mỗi tập dữ liệu, chúng tôi huấn luyện trên `train`, chọn checkpoint AP50:95 validation tốt nhất và đánh giá một lần trên `test`. Điểm tổng quan là trung bình không trọng số của 100 kết quả test này.

| Thiết lập | Quy tắc chiến dịch |
|---|---|
| Huấn luyện | 100 epoch; không early stopping |
| Batch hiệu dụng | 16 |
| Seed | 0; một lần chạy hoàn tất cho mỗi tập dữ liệu |
| Checkpoint | AP50:95 validation tốt nhất, dùng trọng số EMA |
| Chấm điểm test | pycocotools với `maxDets=500` |
| Tuning | Một recipe được cố định cho mỗi họ; không tuning theo từng tập dữ liệu |

Mỗi họ dùng recipe huấn luyện riêng:

| Họ | Độ phân giải | Precision | Tóm tắt tăng cường dữ liệu |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; tắt các tăng cường mạnh trong 15 epoch cuối |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; 15 epoch cuối |
| YOLOX S/M | 640 | FP32 | Cùng recipe của họ, với learning rate riêng theo kích thước |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV và flip; tắt mosaic |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; thiết lập mặc định theo họ của LibreYOLO |
| RF-DETR N/S/M/L và S LoRA | 384/512/576/704; LoRA ở 512 | BF16 | Đa tỉ lệ, crop/resize và flip |

Đây là kết quả với một seed. Chưa thể khẳng định các chênh lệch nhỏ là cải thiện. RF-DETR M và L dùng gradient accumulation để vừa bộ nhớ khả dụng, và một số tập dữ liệu dày đặc cần batch vật lý nhỏ hơn. Các recipe RF-DETR công khai cũng bắt đầu từ checkpoint COCO, trong khi Roboflow dùng checkpoint Objects365 riêng tư cho bảng lịch sử của họ.

Thời gian huấn luyện là số liệu ghi nhận trong chiến dịch, không phải benchmark tốc độ có kiểm soát. Các job đôi khi dùng chung GPU, chạy lại hoặc tiếp tục sau gián đoạn. Chúng tôi không chạy quy trình độ trễ một artifact duy nhất T4 tùy chọn. YOLO-NAS dùng [trọng số pretrained phi thương mại được cấp phép riêng](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) của Deci.

## Cải thiện workload

Các bài kiểm tra nhỏ có thể chứng minh rằng việc huấn luyện khởi chạy được. Chúng không tái hiện nhiều tuần tinh chỉnh liên tục trên nhiều kiến trúc và tập dữ liệu. Ở quy mô này, các đường chạy chậm, áp lực bộ nhớ và vấn đề về tính đúng đắn trở nên khó bỏ sót.

- **Nạp ảnh và validation.** Chúng tôi cache bước resize xác định trước augmentation, tái sử dụng worker và buffer của validator, đồng thời lập biểu đồ các lượt forward validation khi được hỗ trợ. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graph trong huấn luyện.** Hỗ trợ capture được mở rộng từ YOLOv9 và RF-DETR lên 24 họ. Chúng tôi cũng sửa race trong pin-memory của DataLoader và lỗi vô hiệu hóa graph khi chuyển trạng thái huấn luyện. Một bài kiểm tra YOLOv9-T giảm từ 428.4 xuống 367.7 giây với AP được báo cáo không đổi. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Chấm điểm COCO.** Việc đánh giá trên tập dữ liệu dày đặc đôi khi mất nhiều thời gian hơn huấn luyện. Tích hợp faster-coco-eval chấm điểm các dự đoán đã lưu từ toàn bộ 100 phần test trong 8.4 giây thay vì 131.4 giây. Trong 1,400 giá trị metric, 1,381 giá trị giống hệt từng bit; chênh lệch lớn nhất là 2.22e-16 và AP tổng quan không đổi. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR và các đường huấn luyện dùng chung.** Ghép cặp L1 nhanh hơn, ít đồng bộ hóa thiết bị hơn, AdamW hợp nhất và bộ nhớ matcher có giới hạn đã giảm một bước RF-DETR-S từ 266 xuống 234 ms trong bài kiểm tra được ghi nhận. Cùng quá trình tìm hiểu này cũng cải thiện năm matcher khác, logging YOLOv9 và tái sử dụng tensor EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Chúng tôi chuyển các attention phù hợp sang PyTorch SDPA, tích hợp một triển khai CUDA theo giấy phép Apache-2.0 và sửa việc thực thi mixed precision. Chúng tôi cũng viết một kernel Triton deformable-attention ngay trong repo để inference. Trong các bài kiểm tra được ghi lại, kernel nhanh hơn khoảng bốn lần khi chạy riêng và nhanh hơn khoảng 7% từ đầu đến cuối với RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Tính đúng đắn của huấn luyện.** Chúng tôi sửa việc tái tạo BatchNorm của YOLOX và khôi phục PGI, metadata letterbox, sức chứa nhãn và warmup momentum của YOLOv9. Benchmark cung cấp các checkpoint và mẫu lỗi giúp phát hiện cả hai vấn đề. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Các con số này đến từ những bài kiểm tra riêng biệt trên các workload khác nhau. Không nên nhân chúng với nhau để tạo thành một con số tăng tốc tổng quát. LibreYOLO hiện nhanh hơn và đáng tin cậy hơn cho workload huấn luyện thực tế so với trước chiến dịch này.

## Harness và artifact

[Harness huấn luyện công khai](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) lên lịch các tập dữ liệu trên nhiều GPU, tiếp tục các job bị gián đoạn, xử lý khôi phục sau OOM và ghi lại recipe, phiên bản tập dữ liệu, log, checkpoint và dự đoán.

[Kỹ năng run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) cung cấp cho AI coding agent quy trình và hướng dẫn vận hành Vast.ai. [Kho lưu trữ kết quả](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) chứa mọi lần chạy đã công bố.

## Cảm ơn Roboflow

Joseph Nelson hỗ trợ GPU sau khi tôi viết rằng mình muốn benchmark ngoài COCO nhưng không đủ khả năng chi trả cho các lần chạy. Matvei Popov chia sẻ các thiết lập tham chiếu, giải thích cấu hình đánh giá và theo dõi kết quả khi chúng được gửi về.

Sự hỗ trợ đó giúp chúng tôi xác thực việc huấn luyện LibreYOLO trên 17 cấu hình, công bố bằng chứng hữu ích cho việc chọn mô hình, phát hành harness và thử tải thư viện cho đến khi các điểm yếu lộ rõ.

Cảm ơn Joseph, Matvei và đội ngũ Roboflow vì tài nguyên tính toán, thời gian và hướng dẫn.
