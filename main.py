import subprocess
import sys
import os

def run_script(script_path):
    print(f"▶️ BẮT ĐẦU CHẠY: {script_path}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Đảm bảo chạy từ thư mục gốc của project
        )
        print(f"\n✅ HOÀN THÀNH: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ LỖI KHI CHẠY: {script_path}")
        print(f"Mã lỗi (Exit code): {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ KHÔNG TÌM THẤY FILE: {script_path}")
        return False

def main():
    print("🚀 KHỞI ĐỘNG PIPELINE MÔ HÌNH DỰ ĐOÁN BURNOUT!")
    
    # Định nghĩa thứ tự các script cần chạy
    pipeline_scripts = [
        "src/data_loader.py",
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py"
    ]
    
    # Kiểm tra xem thư mục src có tồn tại không
    if not os.path.exists("src"):
        print("❌ Lỗi: Không tìm thấy thư mục 'src'. Vui lòng chạy script từ thư mục gốc của project.")
        sys.exit(1)

    # Chạy lần lượt từng script trong pipeline
    for script in pipeline_scripts:
        success = run_script(script)
        if not success:
            print("\n🛑 Pipeline đã dừng lại do có lỗi xảy ra.")
            sys.exit(1)

    print("\n" + "="*60)
    print("🎉 PIPELINE ĐÃ CHẠY XONG THÀNH CÔNG TỪ A -> Z!")
    print("📂 Bạn có thể kiểm tra kết quả trong thư mục 'data/processed', 'models/' và 'reports/figures/'.")
    print("="*60)

if __name__ == "__main__":
    main()
