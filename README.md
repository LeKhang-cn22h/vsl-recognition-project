# vsl-recognition-project

#  Tạo virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

#  Upgrade pip
python -m pip install --upgrade pip

# Install dependencies 
pip install -r requirements.txt

có thể thêm dataset bằng python src/collect_data.py
huấn luận bằng python src/train_simple.py
chạy thử bằng python src/test_realtime.py
lưu ý: chỉ mới có 3 sign thôi muốn thêm phải vè trong collect_data.py bổ sung thêm
