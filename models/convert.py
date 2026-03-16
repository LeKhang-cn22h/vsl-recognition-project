import os
import torch
import ai_edge_torch

current_dir = os.path.dirname(os.path.abspath(__file__))
pt_path = os.path.join(current_dir, 'best.pt')
tflite_path = os.path.join(current_dir, 'vsl_bilstm.tflite')

# ---------------------------------------------------------
# QUAN TRỌNG: KHỞI TẠO MÔ HÌNH TỪ CLASS CỦA BẠN
# Thay thế 'YourBiLSTMClass' bằng tên class mạng của bạn
# Ví dụ: from model_def import VSL_BiLSTM
# model = VSL_BiLSTM(input_size=..., hidden_size=..., num_classes=...)
# ---------------------------------------------------------

# Tạm gọi biến model (Bạn cần thay thế bằng code khởi tạo thực tế của bạn)
# model = YourBiLSTMClass() 

# Load trọng số vào mô hình
print("Đang load trọng số...")
model.load_state_dict(torch.load(pt_path, weights_only=True))
model.eval()

# ---------------------------------------------------------
# TẠO DUMMY INPUT (Đầu vào giả lập)
# Với BiLSTM, input thường là 3 chiều: (Batch_Size, Sequence_Length, Features_Size)
# Ví dụ: Bạn nhận diện 30 khung hình, mỗi khung hình trích xuất 63 điểm landmark từ tay/mặt
# Thì input sẽ là: (1, 30, 63)
# ---------------------------------------------------------
batch_size = 1
sequence_length = 30 # Sửa lại cho đúng project của bạn
features_size = 63   # Sửa lại cho đúng project của bạn
dummy_input = torch.randn(batch_size, sequence_length, features_size)

# Chuyển đổi bằng ai-edge-torch
print("Đang chuyển đổi PyTorch BiLSTM -> TFLite...")
try:
    edge_model = ai_edge_torch.convert(model, (dummy_input,))
    edge_model.export(tflite_path)
    print(f"✅ Hoàn tất! Đã lưu TFLite tại: {tflite_path}")
except Exception as e:
    print(f"❌ Lỗi trong quá trình chuyển đổi: {e}")