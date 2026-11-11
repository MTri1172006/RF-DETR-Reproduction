# Đồ án CS431: Tái hiện và Mô phỏng RF-DETR (Neural Architecture Search for Real-Time Detection Transformers)

Tài liệu này hướng dẫn chi tiết về nội dung và cách sử dụng file notebook **`RF_DETR_Reproduction_CS431.ipynb`** — một pipeline thực nghiệm toàn diện bằng PyTorch nhằm mô phỏng cơ chế cốt lõi, tái hiện các kết quả đánh giá và thực thi mô hình pre-trained từ bài báo nghiên cứu **RF-DETR**.

---

## 1. Giới thiệu chung

Bài báo **RF-DETR** kết hợp sức mạnh của nền tảng Vision Foundation Model (**DINOv2**) với kiến trúc **Real-Time DETR** thông qua cơ chế **Tìm kiếm Kiến trúc Chia sẻ Trọng số (End-to-End Weight-Sharing NAS)**. Thay vì phải huấn luyện lại từ đầu cho từng kích cỡ mô hình (Nano, Small, Medium, Large) hay từng thiết bị phần cứng, RF-DETR huấn luyện một mạng gốc duy nhất và cho phép tinh chỉnh linh hoạt **5 "núm vặn" (tunable knobs)** ngay lúc suy luận (Zero-shot Inference).

Notebook `RF_DETR_Reproduction_CS431.ipynb` được thiết kế thành **2 nhánh rõ ràng**:
1. **Nhánh Mô phỏng Kiến trúc & Cơ chế (`RFDETRSimulator` - Mục 2 đến Mục 7.1):** Xây dựng từ đầu bằng PyTorch để minh họa trực quan toán học và cấu trúc bên trong của 5 núm vặn NAS, nhánh phân đoạn thực thể (`RF-DETR-Seg`), giao thức đo kiểm chống quá nhiệt GPU và quy trình xuất ONNX.
2. **Nhánh Thực thi Mô hình Thực tế (`RFDETRNano` - Mục 7.2):** Tích hợp trực tiếp thư viện chính thức `rfdetr` của Roboflow, tự động tải trọng số đã huấn luyện sẵn (pre-trained trên Objects-365 & COCO) để phát hiện vật thể trên ảnh thực tế.

---

## 2. Cấu trúc chi tiết của Notebook

File `RF_DETR_Reproduction_CS431.ipynb` bao gồm **7 phần thực nghiệm** nối tiếp nhau:

| Mục | Tên phần trong Notebook | Nội dung kỹ thuật cốt lõi (Đối chiếu Paper) |
| :--- | :--- | :--- |
| **1** | **Cài đặt môi trường & Kiểm tra phần cứng** | Cài đặt `rfdetr`, `supervision`, `onnx`, `onnxscript`, `pycocotools` và tự động nhận diện thiết bị (`cuda` trên GPU T4 hoặc fallback về `cpu`). |
| **2** | **Mô phỏng kiến trúc RF-DETR & RF-DETR-Seg (`RFDETRSimulator`)** | Cài đặt lớp mô phỏng PyTorch hỗ trợ 5 núm vặn NAS:<br>• **Patch Size ($a$):** Nội suy song tuyến tính bộ lọc tích chập theo phong cách *FlexiViT*.<br>• **Decoder Layers ($b$):** Cắt tỉa (prune) từ $6 \rightarrow 0$ lớp decoder lúc inference.<br>• **Query Dropping ($c$):** Lọc giữ lại top-$k$ queries có độ tự tin cao nhất từ Encoder.<br>• **Resolution ($d$):** Co giãn ảnh đầu vào và nội suy *Positional Embedding* ($H' \times W'$).<br>• **Window Attention ($e$):** Tùy chỉnh số cửa sổ chú ý.<br>• **Nhánh RF-DETR-Seg:** Nội suy bilinear đặc trưng encoder kết hợp phép nhân tensor `einsum` với query embeddings để sinh mặt nạ phân đoạn. |
| **3** | **Mô phỏng Vòng lặp Huấn luyện NAS & Suy luận Zero-shot** | Tái hiện kỹ thuật *Architecture Augmentation* (lấy mẫu đồng nhất cấu hình con từ không gian tìm kiếm ở mỗi bước train - *Appendix A*) và suy luận tức thì trên các cấu hình Pareto-Optimal (*Table 7*). |
| **4** | **Trực quan hóa Nội suy Bộ lọc Patch Embedding** | Trích xuất trọng số `patch_embed.weight` và vẽ đồ thị minh họa quá trình nội suy bộ lọc từ kích thước gốc $16 \times 16$ sang $8 \times 8$ và $32 \times 32$ (*Figure 3a*). |
| **5** | **Quét lưới NAS & Vẽ đường cong Pareto Frontier** | • Vẽ biểu đồ đối chiếu **mAP vs. Latency** giữa RF-DETR và YOLOv8, YOLOv11, D-FINE, LW-DETR (*Table 2, Figure 4*).<br>• Chạy thực nghiệm quét lưới các mức `num_decoder_layers` $\in \{0, 1, 2, 4, 6\}$ và `num_queries` $\in \{50, 100, 200, 300\}$ để đo tốc độ thực tế. |
| **6** | **Kiểm chứng Giao thức Đo độ trễ Chuẩn hóa (200ms Buffer)** | Thực nghiệm đối chiếu độ ổn định Latency giữa việc chạy liên tục (`0ms buffer` - dễ bị GPU power throttling/quá nhiệt) và chèn thời gian nghỉ `200ms buffer` (*Table 1 & Table 20*). |
| **7** | **Xuất ONNX (Opset 17) & Chạy Pre-trained `rfdetr` thực tế** | • **7.1:** Đóng gói mô hình qua `RFDETROnnxWrapper` và xuất sang định dạng `.onnx` với **Opset 17** (*Table 1 & Appendix A*).<br>• **7.2:** Tải trọng số pre-trained chính thức `RFDETRNano` và vẽ Bounding Box phát hiện vật thể trên ảnh mẫu COCO. |

---

## 3. Hướng dẫn cài đặt và chạy trên Google Colab

### Bước 1: Mở Notebook và thiết lập GPU
1. Tải file `RF_DETR_Reproduction_CS431.ipynb` lên [Google Colab](https://colab.research.google.com/).
2. Trên thanh menu, chọn **Runtime (Thời gian chạy)** $\rightarrow$ **Change runtime type (Thay đổi loại thời gian chạy)**.
3. Tại mục **Hardware accelerator**, chọn **T4 GPU** rồi nhấn **Save**. *(Lưu ý: Code có cơ chế tự động chuyển về CPU nếu không bật GPU, nhưng để đo kiểm chính xác hiện tượng GPU Power Throttling ở Mục 6, bạn nên bật T4 GPU).*

### Bước 2: Chạy tuần tự các Cell
Nhấn **Runtime $\rightarrow$ Run all** (hoặc chạy từng cell từ Mục 1 đến Mục 7).

### Lưu ý xử lý lỗi xuất ONNX trên Colab (Mục 7.1)
Trên môi trường Google Colab mới sử dụng Python 3.13 và PyTorch 2.x, hàm `torch.onnx.export` yêu cầu thêm gói `onnxscript` hoặc tắt trình xuất Dynamo. Nếu gặp lỗi `ModuleNotFoundError: No module named 'onnxscript'`, bạn chỉ cần:
* Thêm dòng `!pip install -q onnxscript` vào Cell 1 (hoặc đầu Cell 7.1), **HOẶC**
* Thêm tham số `dynamo=False` vào hàm `torch.onnx.export(...)`.

---

## 4. Hướng dẫn thử nghiệm với ảnh cá nhân (Mục 7.2)

Ở Mục **7.2**, mô hình sử dụng trọng số pre-trained thật của **RF-DETR Nano**. Để chạy nhận diện vật thể trên ảnh riêng của bạn:
1. Tải file ảnh của bạn (ví dụ `my_photo.jpg`) lên thư mục gốc của Google Colab.
2. Thay đổi đường dẫn trong Cell 7.2:
   ```python
   image = Image.open("my_photo.jpg").convert("RGB")
   detections = real_model.predict(image, threshold=0.5)
   ```
3. Chạy lại cell để hiển thị ảnh kèm khung giới hạn (Bounding Box) và nhãn phân lớp.

---

## 5. Tài liệu tham khảo (Citation)

Nội dung và mã nguồn trong repository này được xây dựng dựa trên bài báo nghiên cứu gốc của nhóm tác giả tại Roboflow & Carnegie Mellon University:
* **Bài báo:** *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*
* **Thư viện chính thức:** [https://github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr)