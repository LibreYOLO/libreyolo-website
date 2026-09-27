---
title: Khắc phục sự cố
seo_title: Sửa các lỗi LibreYOLO thường gặp
description: >-
  Những lỗi LibreYOLO thường phát nhất, ý nghĩa và cách sửa. Bao gồm hai lỗi tạo
  đầu ra sai thay vì phát ngoại lệ.
lead: >-
  Các lỗi được nhóm theo thông báo bạn nhìn thấy. Hai mục cuối đề cập vấn đề
  ngược lại: mã chạy, trả về kết quả có vẻ hợp lý nhưng lại sai.
keywords:
  - lỗi libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda hết bộ nhớ
  - libreyolo notimplementederror
  - khắc phục sự cố libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Các lỗi được nhóm theo nội dung bạn nhìn thấy. Nếu thông báo của bạn không có ở
đây, [FAQ](/docs/faq) trả lời những câu hỏi không phải lỗi, còn `libreyolo
models` báo những gì bản cài thực sự có thể nạp.

## ModuleNotFoundError nêu tên package bạn chưa từng import

Một số họ cần gói bổ sung tùy chọn. Thông báo nêu tên package còn thiếu thay vì
tên gói bổ sung, nên cách sửa không phải lúc nào cũng rõ từ traceback.

Chạy `libreyolo models`. Mọi họ còn thiếu dependency được in kèm lệnh pip chính
xác để kích hoạt, nên bạn không phải tự ánh xạ package về gói bổ sung.
`libreyolo models --json` in cùng dữ liệu dưới dạng object.

[Trang cài đặt](/docs/install) liệt kê mọi gói bổ sung và phạm vi của chúng.

## Inference ONNX cần onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Package cơ sở không phụ thuộc vào runtime vì lựa chọn phù hợp tùy phần cứng.
Cài `onnxruntime` cho CPU hoặc `onnxruntime-gpu` cho CUDA. Cả hai cung cấp cùng
module `onnxruntime`, vì vậy chỉ cài một, không cài cả hai.

## Không tìm thấy mô hình ONNX

```
FileNotFoundError: ONNX model not found: <path>
```

Đường dẫn được phân giải tương đối với thư mục làm việc, không phải script. Lỗi
này cũng xuất hiện khi quá trình xuất âm thầm ghi ở nơi khác: `export()` trả về
đường dẫn đã ghi, vì vậy hãy lấy giá trị trả về thay vì giả định tên.

## NotImplementedError từ train()

Không phải họ nào cũng huấn luyện được. Một số chỉ được port cho dự đoán, xác
thực và xuất, và `train()` của chúng sẽ báo lỗi thay vì giả vờ chạy.

[Mục FAQ](/docs/faq) giải thích lý do. Để kiểm tra một họ cụ thể trước khi viết
script huấn luyện, trang mô hình của họ cho biết khả năng huấn luyện.

## NotImplementedError từ export()

Một họ có thể hỗ trợ tác vụ nhưng vẫn không xuất được tác vụ đó. EoMT là trường
hợp thường gặp: `export()` nhận tác vụ semantic và báo lỗi với `segment` cùng
`panoptic`, vì runtime contract query-mask mà chúng cần chưa được định nghĩa.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Mỗi trang họ có ma trận xuất cho biết tổ hợp tác vụ và định dạng nào đã được
xác thực.

## CUDA hết bộ nhớ

Trước tiên giảm `batch`, sau đó giảm `imgsz`. Cả hai làm thay đổi bộ nhớ gần
tỷ lệ với kích thước, nhưng batch có thể giảm mà không thay đổi nội dung mô
hình nhìn thấy.

Nếu lỗi xảy ra khi xác thực thay vì huấn luyện, quá trình xác thực chạy batch
size riêng, vì vậy cũng hãy giảm giá trị đó.

Trên Windows, GPU dùng cho màn hình có một kiểu lỗi thứ hai trông giống lỗi
CUDA ngẫu nhiên thay vì hết bộ nhớ: driver reset GPU nếu GPU ngừng phản hồi lâu
hơn timeout, làm tiến trình đang chạy bị dừng. Kernel dài trên card đang điều
khiển màn hình có thể gây ra lỗi này.

## Không tải được trọng số

Trọng số được lấy từ Hugging Face ở lần dùng đầu tiên và cache cục bộ.
[FAQ](/docs/faq) nêu vị trí cache và cách chạy hoàn toàn offline.

Nếu tải xuống trả về 404, hãy kiểm tra tên file đã truyền. URL được suy ra từ
tên đó, gồm cả hậu tố tác vụ, nên tên không khớp checkpoint đã công bố sẽ tạo
URL không tồn tại. Bảng checkpoint trên mỗi trang mô hình liệt kê chính xác các
tên file đã công bố.

## Huấn luyện bị treo hoặc khởi động lại trên Windows

Windows không có `fork`, nên worker dataloader bắt đầu bằng cách import lại
script của bạn. Nếu thiếu guard `if __name__ == "__main__":`, mỗi worker chạy
lại lệnh huấn luyện, dẫn đến deadlock hoặc sinh tiến trình vô hạn.

```python
def main():
    ...  # dựng mô hình và gọi train()

if __name__ == "__main__":
    main()
```

Đặt `workers=0` cũng tránh được lỗi nhưng làm giảm throughput. Guard là cách
sửa tốt hơn.

## Hai lỗi không phát ngoại lệ

Phần còn lại của trang nói về lỗi. Hai lỗi này tệ hơn vì mã vẫn chạy và trả về
thứ trông có vẻ đúng.

### Lập chỉ mục một kết quả đơn

`predict()` trả về một `Results` cho một ảnh và danh sách cho nhiều ảnh. Lập
chỉ mục giá trị trả về của một ảnh sẽ chọn một *detection*, không phải ảnh:

```python
result = model.predict("image.jpg")   # một Results
result.boxes                          # mọi detection, chính xác
result[0].boxes                       # MỘT detection, âm thầm
```

Không có ngoại lệ vì lập chỉ mục `Results` là thao tác hợp lệ trả về tập con.
Mã viết cho dạng danh sách âm thầm báo một box cho mỗi ảnh. Chỉ lập chỉ mục khi
bạn biết chắc đó là danh sách.

### Đọc metric như thuộc tính

`val()` trả về dictionary thuần túy với khóa là tên metric, không phải object
có truy cập thuộc tính:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # chính xác
metrics.box.map               # AttributeError
```

Các khóa có namespace `metrics/` và `speed/`. Hãy in dictionary một lần để xem
tác vụ tạo ra gì, vì tập khóa khác nhau theo tác vụ.

## Kiểm tra dataset trước khi huấn luyện

Hầu hết lỗi huấn luyện là vấn đề dataset. `libreyolo doctor data.yaml` chạy các
kiểm tra tình trạng trên dataset detection và báo phát hiện theo mức độ nghiêm
trọng, nhanh hơn đọc traceback từ epoch đầu tiên.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Xem [lệnh doctor](/docs/cli/doctor) để biết danh mục kiểm tra.
