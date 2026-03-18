import pandas as pd

def convert_excel_to_csv(excel_path, csv_path, sheet_name=0):
    """
    Hàm chuyển đổi file Excel sang CSV.
    
    Tham số:
    - excel_path: Đường dẫn tới file Excel cần chuyển.
    - csv_path: Đường dẫn lưu file CSV đầu ra.
    - sheet_name: Tên hoặc chỉ số của sheet cần đọc (mặc định là 0 - sheet đầu tiên).
    """
    try:
        # Đọc dữ liệu từ file Excel
        # Lưu ý: Cần cài đặt thêm thư viện 'openpyxl' để đọc file .xlsx
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Lưu dữ liệu ra file CSV
        # index=False giúp loại bỏ cột số thứ tự mặc định của pandas
        # encoding='utf-8' đảm bảo không bị lỗi font nếu có tiếng Việt
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Chuyển đổi thành công!\nFile đã được lưu tại: {csv_path}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    
    duong_dan_excel = r"D:\burnout-model\data\external\Data_Questionnaire_Mapping.xlsx" 
    duong_dan_csv = r"D:\burnout-model\data\csv\Data_Questionnaire_Mapping.csv"

    # Gọi hàm
    convert_excel_to_csv(duong_dan_excel, duong_dan_csv)