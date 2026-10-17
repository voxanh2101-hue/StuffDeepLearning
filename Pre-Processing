# 📘  Tiền Xử Lý Dữ Liệu 

---

## 1. Tensor & Chuẩn Hóa Dữ Liệu (Normalization)

### ❓ Tensor là gì và tại sao phải chuyển ảnh sang Tensor?
Máy tính không thể "nhìn" và hiểu hình ảnh trực tiếp như con người. Để mô hình học sâu (Deep Learning) có thể xử lý được, hình ảnh cần được chuyển đổi thành **Tensor** (mảng đa chiều chứa các con số).

* **Mục tiêu:**
  1. **Đồng nhất định dạng:** Đưa toàn bộ dữ liệu đầu vào về cấu trúc toán học mà framework (PyTorch/TensorFlow) có thể tính toán trên ma trận.
  2. **Tối ưu việc học:** Đưa giá trị pixel từ khoảng `[0, 255]` về khoảng `[0, 1]`, sau đó chuẩn hóa (Normalize) về phân phối chuẩn. 


### 📊 Thông số chuẩn hóa trên tập CIFAR-10
Sau khi chuyển ảnh sang Tensor `[0, 1]`, từng kênh màu (R, G, B) sẽ được chuẩn hóa theo công thức: `output = (input - mean) / std`. 

Đối với tập dữ liệu **CIFAR-10**, các giá trị trung bình (`Mean`) và độ lệch chuẩn (`Std`) đã được tính toán sẵn trên toàn bộ **50.000 ảnh** của tập huấn luyện:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
