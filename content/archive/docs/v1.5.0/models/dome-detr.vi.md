---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: phát hiện vật thể siêu nhỏ trong LibreYOLO'
description: >-
  Dùng Dome-DETR trong LibreYOLO để phát hiện vật thể siêu nhỏ trên ảnh hàng
  không và ảnh drone. Chuyển đổi trọng số upstream, dự đoán, tinh chỉnh và xác
  thực với mã nguồn dùng giấy phép MIT.
lead: >-
  Một mô hình chuyên xử lý vật thể siêu nhỏ được xây dựng trên D-FINE: density
  head xác định vị trí vật thể, attention của encoder bị giới hạn trong các cửa
  sổ chứa chúng, còn số lượng query được tính từ mật độ đó thay vì cố định.
  LibreYOLO hỗ trợ mô hình này cho tác vụ phát hiện.
keywords:
  - Dome-DETR
  - phát hiện vật thể siêu nhỏ
  - phát hiện vật thể nhỏ
  - ảnh hàng không
  - phát hiện bằng drone
  - viễn thám
  - VisDrone
  - AI-TOD
  - DETR
  - query thích ứng theo mật độ
last_verified: 1.5.0
snippets:
  predict:
    - label: Chuyển đổi rồi dự đoán
      language: bash
      code: |
        # LibreYOLO không lưu trữ trọng số Dome-DETR, vì vậy checkpoint được
        # tải từ repo upstream rồi chuyển đổi một lần.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Đây là đường dẫn cục bộ, không phải tên rút gọn: family này không tải
        gì về.

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        result = model("drone-frame.jpg", save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Tên lớp đối tượng
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Không có checkpoint COCO, vì vậy các lớp đến từ dataset dùng để
        # huấn luyện trọng số và được đọc từ metadata của checkpoint.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 lớp AI-TOD-V2

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 lớp VisDrone
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 381f01d769e7c420
---

## Cài đặt

Dome-DETR không cần thành phần tùy chọn nào. Mọi nội dung mà mô hình import đều
có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Không có gì để tự động tải xuống. LibreYOLO không lưu trữ các trọng số này, vì
vậy quy trình là: tải checkpoint upstream, chuyển đổi một lần, sau đó nạp tệp
đã chuyển đổi bằng đường dẫn. Phần [Giấy phép](#licensing) giải thích lý do.

<code-tabs name="predict" />

Đối tượng `Results` trả về giống với đối tượng của mọi family, vì vậy việc thay
sang detector khác chỉ cần đổi một dòng. `conf` và `max_det` lọc quá trình chọn
query; `iou` được chấp nhận để giữ tính tương thích của API nhưng không có tác
dụng, vì decoder là một bộ dự đoán theo tập hợp không có bước NMS. Xem
[dự đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

Hai khả năng bị tắt cho family này. Việc capture CUDA graph bị tắt vì số lượng
query của PAQI phụ thuộc vào dữ liệu, nên forward pass thay đổi shape giữa các
ảnh, đúng vào trường hợp graph capture không thể xử lý. Tăng cường tại thời điểm
kiểm thử (test-time augmentation) chạy ở một kích thước vuông cố định, vì vậy
yêu cầu TTA đa tỷ lệ sẽ không tạo ra thay đổi nào.

## Biến thể

Ba kích thước s, m và l đều chạy ở 800 x 800. Kích thước chọn backbone, còn
dataset tạo ra trọng số sẽ chọn độ sâu decoder và ngân sách query, vì vậy chỉ
mã kích thước không đủ để xác định một graph. Trọng số AI-TOD-V2 chọn từ 300 đến
1500 query cho mỗi ảnh, trọng số VisDrone chọn từ 250 đến 500, và mô hình lớn
chạy bốn lớp decoder trên AI-TOD-V2 so với sáu lớp trên VisDrone.

Dome-DETR là D-FINE với ba phần bổ sung. DeFE dự đoán bản đồ mật độ. MWAS dùng
bản đồ đó để giới hạn attention của encoder vào những cửa sổ thực sự chứa vật
thể, thay vì attention ở mọi nơi. PAQI xác định kích thước tập query từ cùng mật
độ đó thay vì decode một tập 300 query cố định. Mức cải thiện tập trung ở nơi
vật thể nhỏ nhất và thu hẹp khi vật thể lớn hơn: ablation của chính upstream đưa
AP trên vật thể cực nhỏ từ 14.0 lên 17.8, trong khi AP trên vật thể trung bình
chỉ tăng từ 45.4 lên 46.4. Hãy dùng mô hình này như phần bổ trợ cho
[D-FINE](/docs/models/d-fine) trên ảnh hàng không, drone và viễn thám, không phải
để thay thế D-FINE.

LibreYOLO không công bố dòng benchmark nào cho family này vì không công bố
checkpoint để benchmark.

## Huấn luyện

Dome-DETR có thể huấn luyện. Quá trình huấn luyện chạy toàn bộ objective của
upstream: các loss D-FINE cùng supervision về mật độ và số lượng của DeFE, với
các query được padding bị loại khỏi thành phần phân loại và attention mask khử
nhiễu theo từng ảnh để phần padding của ảnh này không thể rò rỉ sang ảnh khác.

<code-tabs name="train" />

Cấu hình kế thừa recipe của D-FINE và thay đổi những gì MWAS yêu cầu. `imgsz` là
800, `lr0` là `2e-4`, nhóm tham số backbone được nhân tỷ lệ bằng
`backbone_lr_mult=0.1`, còn `multi_scale` bị buộc tắt vì các cửa sổ MWAS cần đầu
vào luôn chia hết cho stride 8. `batch` mặc định là 4 thay vì 16 như D-FINE:
PAQI padding mỗi batch theo phần tử rộng nhất, nên bộ nhớ phụ thuộc vào ảnh có
nhiều đối tượng nhất trong batch thay vì ảnh trung bình.

Có một lưu ý thẳng thắn về độ chính xác. Upstream huấn luyện trong 160 epoch với
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, trong khi các giá trị mặc định
này chạy lịch flat-cosine của D-FINE trong cùng 160 epoch. Lịch đó chưa được tái
lập tại đây và các số liệu AP trong bài báo cũng chưa được tái lập, vì vậy hãy
đọc chúng như kết quả của tác giả upstream chứ không phải cam kết rằng recipe
này sẽ đạt được. Hãy cung cấp lịch upstream nếu mục tiêu là khớp với bài báo.

Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data
augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về một dictionary được lập chỉ mục bằng tên metric và in kết quả
theo từng lớp đối tượng khi vẫn bật `verbose`.

<code-tabs name="val" />

Quá trình xác thực chạy trên dataset của bạn ở định dạng đã dùng để huấn luyện.
Cổng xác thực COCO của thư viện không áp dụng ở đây vì family này không có
checkpoint COCO để đo.

## Xuất

Không hỗ trợ xuất sang bất kỳ định dạng nào, và yêu cầu xuất sẽ phát sinh lỗi
thay vì tạo tệp.

Lý do là PAQI. Nó quyết định số lượng query cho mỗi ảnh từ các proposal đã lọc
theo mật độ và một vòng lặp suppression thích ứng theo mật độ theo kiểu greedy,
vì vậy độ dài đầu ra của decoder là thuộc tính của đầu vào chứ không phải của
graph. Tracing sẽ cố định số lượng mà ảnh dùng để trace tình cờ tạo ra, dẫn đến
artifact âm thầm trả về kết quả sai cho mọi ảnh khác. Một cách biểu diễn tĩnh sẽ
phải trải vòng suppression đó trên toàn bộ 250 đến 1500 candidate, còn việc thu
gọn về top-k cố định sẽ loại bỏ đúng phần recall vật thể siêu nhỏ vốn là lý do
family này tồn tại. Nếu cần detection transformer có thể xuất, hãy dùng
[D-FINE](/docs/models/d-fine).

## Checkpoint

Không có checkpoint nào để liệt kê. LibreYOLO không công bố trọng số Dome-DETR,
và không tên nào có dạng `LibreDOMEDETR<size>-<dataset>.pt` được phân giải thành
một lượt tải xuống.

Upstream công bố sáu checkpoint với kích thước s, m và l cho mỗi một trong hai
dataset: AI-TOD-V2 có 9 lớp đối tượng và VisDrone có 12. Không có checkpoint
COCO, vì vậy tên tệp chuẩn luôn có hậu tố dataset, còn tên lớp đối tượng được
lưu trong metadata của checkpoint thay vì lấy từ hằng số family. Yêu cầu tên
rút gọn `LibreDOMEDETRs.pt` sẽ phát sinh lỗi ngay với thông báo nêu tên hai tệp
thực và lệnh chuyển đổi, thay vì cố tải một địa chỉ sẽ trả về 404.

`weights/convert_domedetr_weights.py` thực hiện việc chuyển đổi. Script dựng lại
graph LibreYOLO, nạp các tensor upstream vào đó và từ chối ghi bất kỳ thứ gì nếu
có dù chỉ một key bị thiếu, ngoài dự kiến hoặc sai shape, vì vậy tệp đã chuyển
đổi hoặc khớp chính xác, hoặc không tồn tại. Trỏ script tới một tệp `.pth`
upstream rồi truyền kích thước và biến thể:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Về độ tương đồng số học, `weights/parity_domedetr.py` so sánh bản port này với
implementation upstream trên cả sáu checkpoint và báo cáo `max_abs_diff ==
0.0` cho cả `pred_logits` lẫn `pred_boxes`, sau khi kiểm tra từng bit của window
mask MWAS, đồng thời so sánh riêng từng thành phần loss với criterion của
upstream. Cần hiểu rõ đây là gì: một script thủ công cần checkout upstream và
các checkpoint đã công bố trên ổ đĩa, được chạy bằng tay. Nó không thuộc quy
trình tích hợp liên tục và không có job CI nào tái lập kiểm tra này.

## Giấy phép

<provenance-box>

Trọng số là lý do family này không được mirror. Model card upstream không có
trường giấy phép trong metadata, còn phần văn bản nói rằng dự án dùng Apache-2.0
nhưng đồng thời giới hạn tài liệu chỉ cho mục đích nghiên cứu học thuật. Hai
cách hiểu đó không thống nhất, và cách hiểu nghiêm ngặt hơn không cấp quyền phân
phối lại, vì vậy LibreYOLO liên kết tới repo upstream thay vì sao chép các tệp
trong khi chờ làm rõ. Cùng lập luận này cũng được áp dụng cho
[YOLO-NAS](/docs/models/yolo-nas) tại đây.

Mã nguồn là một vấn đề riêng và rõ ràng hơn. Repo upstream dùng Apache-2.0, bản
port của LibreYOLO dùng MIT, còn trọng số bạn tự huấn luyện trên dữ liệu của
mình thuộc về bạn.

</provenance-box>

## Trích dẫn

Dome-DETR được công bố tại ACM Multimedia 2025 với tên "Dome-DETR: DETR with
Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection". Bản preprint có tại
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Các tác giả không
công bố khối BibTeX trong repo, vì vậy tài liệu này không dựng lại một khối thủ
công để đưa vào đây.

<citation-block />
