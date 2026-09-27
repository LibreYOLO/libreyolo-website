---
title: Cấp ổn định
seo_title: Ý nghĩa từng cấp hỗ trợ LibreYOLO
description: >-
  Từ vựng cấp của LibreYOLO: ba cấp hỗ trợ xuất, bốn cấp API, sáu nhóm phạm vi
  và những gì chúng không cam kết.
lead: >-
  LibreYOLO dùng từ cấp cho ba khái niệm riêng: bằng chứng đứng sau pipeline
  xuất, call contract mà họ mô hình đáp ứng và nhóm phạm vi mà họ tham gia.
  Trang này định nghĩa từng khái niệm và nêu điều nó không hàm ý.
keywords:
  - cấp hỗ trợ libreyolo
  - validated available blocked
  - cấp hỗ trợ xuất
  - nhóm phạm vi libreyolo
  - g0 g1 g2 g3 g4
  - cấp mô hình
last_verified: 1.5.0
verification: >-
  Cấp xuất lấy từ docs/adr/0011-export-support-tiers.md và
  libreyolo/export/support.py; nhóm phạm vi và số lượng theo họ từ MODEL_GROUPS
  trong libreyolo/models/registry.py; cổng huấn luyện từ đầu từ
  libreyolo/models/base/model.py và libreyolo/cli/commands/train.py; danh mục
  CLI từ libreyolo/models/inventory.py; cấp API từ docstring package và contract
  base.py trong libreyolo/models/sam/, openvocab/ và vlm/, tất cả ở v1.5.0. Nhãn
  nhóm cho người đọc (Flagship, Core, Supported, Chỉ inference, Museum, Cấp
  sibling) là từ vựng riêng của website cho cùng các nhóm, từ
  src/data/docs/registry.json.
snippets:
  usage:
    - label: Đọc cả hai phân loại của một họ
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Cấp hỗ trợ xuất

Cấp quyết định lệnh gọi có thành công hay không. Nó áp dụng cho bộ ba `(family,
task, format)`, và mỗi tổ hợp có đúng một cấp.

| Cấp | Ý nghĩa | Điều xảy ra khi gọi `export()` |
|---|---|---|
| `validated` | Parity số học được kiểm tra trong CI hoặc lần chạy nightly có tài liệu | Chạy |
| `available` | Đã triển khai chuyển đổi nhưng chưa ghi nhận bằng chứng parity số học ở runtime | Chạy |
| `blocked` | Không có pipeline được hỗ trợ | Phát `NotImplementedError` trong preflight kèm lý do |

Cả validated và available đều tiếp tục mà không cần xác nhận hoặc cảnh báo
chung. Khác biệt nằm ở bằng chứng chứ không phải quyền: mục validated có parity
test và bản phát hành `since`, còn mục available thì chưa. Ví dụ, chuyển đổi
CoreML không có lần chạy dự đoán macOS là available chứ không phải validated.

Tổ hợp blocked thất bại trước khi kiểm tra dependency, nạp dữ liệu hiệu chuẩn,
tracing hoặc tạo artifact, nên không có gì dở dang được ghi.

Mỗi ô validated có ràng buộc mô tả cấu hình tạo ra con số parity, thường là
canvas đầu vào cố định, batch 1, FP32 và phiên bản runtime cụ thể. Hãy hiểu đó
là cam kết về cấu hình ấy, không phải định dạng nói chung. Quy tắc điền các ô
không có mục tường minh nằm trên trang [ma trận
xuất](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Cấp API

Cấp quyết định hình dạng lệnh gọi. Mỗi họ nằm trong đúng một cấp, được chọn
theo call contract chứ không theo kiến trúc.

| Cấp | Factory | Contract |
|---|---|---|
| Detector factory | `LibreYOLO` | Một forward không có prompt trả về mọi object tìm thấy với score đã hiệu chuẩn. Thành viên tự đăng ký bằng cách nhận diện checkpoint |
| Promptable segmentation | `LibreSAM` | Forward vô nghĩa nếu thiếu prompt không gian hoặc khái niệm theo ảnh tại lúc gọi. Tương tác và có trạng thái: encode một lần, prompt nhiều lần |
| Open-vocabulary detection | `LibreOpenVocab` | Detector phân biệt có điều kiện văn bản. Danh sách lớp là prompt được đặt bằng `set_classes` |
| Vision-language | `LibreVLM` | Mô hình sinh được điều khiển như detector. Danh sách lớp là prompt và độ tin cậy là placeholder |

Ba cấp sibling cố ý không đăng ký vào detector factory, vì vậy
`LibreYOLO("some-alias")` không truy cập chúng. Chúng nạp theo alias kích thước
và tự động tải thay vì dò checkpoint.

Cả bốn trả về cùng `Results`, nên mã downstream không đổi giữa chúng. Khác biệt
là phương thức nào hoạt động: cấp sibling phát `NotImplementedError` với
`train()`, `val()` và `export()`, còn cấp SAM cùng open-vocabulary cũng phát lỗi
với `track()`. Mỗi trang cấp liệt kê các phần loại trừ riêng.

## Nhóm phạm vi

Phân loại này quyết định lần chạy kiểm thử đa họ bao gồm những họ nào và là
phân loại người đọc thường gặp nhất trên trang mô hình. Mỗi họ đã đăng ký nằm
trong đúng một nhóm, và kiểm thử thất bại nếu họ đã đăng ký thiếu enrollment.
`GROUPS` trong `libreyolo/models/registry.py` là nguồn của cột Ý nghĩa bên dưới;
`MODEL_GROUPS` trong cùng file gán mọi họ, còn cột Họ đếm trực tiếp phép gán đó.
Cột Nhãn là tên ngắn mà website dùng cho cùng nhóm ở đầu trang mô hình.

| Nhóm | Nhãn | Số họ | Ý nghĩa |
|---|---|---|---|
| `g0` | Flagship | 2 | Mỏ neo flagship bắt buộc trong phạm vi tính năng dùng chung |
| `g1` | Core | 10 | Tập phạm vi detector có thể huấn luyện |
| `g2` | Supported | 14 | Tập phạm vi họ có thể huấn luyện bổ sung |
| `g3` | Chỉ inference | 35 | Các họ không có bản triển khai huấn luyện |
| `g4` | Museum | 5 | Họ lịch sử có phạm vi inference |
| `s` | Cấp sibling | 21 | API sibling (SAM, open-vocab, VLM, zero-shot) được kiểm tra riêng |

Tổng cộng 87 họ trong sáu nhóm. Chỉ `g3` đã chứa nhiều họ hơn tất cả nhóm còn
lại cộng lại, vì phần lớn registry là dòng chỉ inference và phạm vi museum thay
vì detector đang được huấn luyện tích cực.

Với người đọc chọn mô hình, nhóm cho biết nơi kỳ vọng ưu tiên kỹ thuật, không
phải độ chính xác của họ. `g0` và `g1` là nơi tính năng mới được thiết kế và
đưa vào trước; `g2` được duy trì ổn định trong CI nhưng tính năng xuất hiện tùy
cơ hội thay vì cùng đợt phát hành. `g3` nêu sự thiếu vắng chứ không phải giới
hạn: predict, validate và export ở nơi họ hỗ trợ vẫn hoạt động, còn `train()`
trên họ `g3` hoặc `g4` phát `NotImplementedError` nêu lý do thay vì âm thầm làm
một phần. Họ `s` hoàn toàn không nằm trong đánh đổi này vì nạp qua factory riêng
thay vì `LibreYOLO()`. Xem [khái niệm cốt lõi](/docs/concepts) để biết nhóm kết
hợp với tác vụ, họ và kích thước thế nào khi đọc tên file checkpoint.

Nhóm không tự cấp hoặc hạn chế khả năng hướng đến người dùng. Hỗ trợ đến từ API
đã triển khai của họ và phép kiểm tra khả năng theo định dạng, không bao giờ chỉ
từ thành viên nhóm. Nhóm phân loại họ chứ không phân loại tác vụ, nên lần chạy
phạm vi theo tác vụ sẽ nêu tác vụ tường minh, như "g1 detect".

Hai nơi đọc nhóm ở runtime thay vì chỉ trong kiểm thử.
`collect_model_inventory()` trong `libreyolo/models/inventory.py` gắn nhóm vào
mọi mục danh mục CLI in ra, còn `pretrained=False` chỉ kích hoạt pipeline khởi
tạo lại đặc biệt từ đầu cho họ trong `g0` và `g1`. Ngoài hai nhóm này, phép kiểm
tra trong `libreyolo/models/base/model.py` bị bỏ qua hoàn toàn, nên
`pretrained=False` đến `train()` riêng của họ như keyword thông thường.

## Huấn luyện

Họ trong `g3` hoặc `g4` không có bản triển khai huấn luyện và gọi `train()` sẽ
báo lỗi. Đây là thuộc tính mã của họ, không phải do nhóm gây ra; nhóm ghi nhận
sự thật đó.

Với họ có thể huấn luyện, việc một nút augmentation có đến pipeline hay không
là câu hỏi riêng với từ vựng ba giá trị `used`, `gated_by_mosaic` và `ignored`.
Xem [ma trận augmentation](/docs/reference/augmentation-matrix).

## Những gì một cấp không cho bạn biết

Cấp không phải cam kết độ chính xác. Bản xuất validated cho biết artifact tái
tạo mô hình native trong ngưỡng đã nêu; nó không nói mô hình native đạt điểm
thế nào trên dataset. Con số benchmark nằm trên trang mô hình.

Cấp cũng không phải tuyên bố giấy phép. Giấy phép trọng số khác nhau trong một
họ và repo lưu checkpoint cụ thể là nguồn có thẩm quyền. Việc một họ nằm trong
detector factory không nói gì về khả năng dùng trọng số công bố cho thương mại.
