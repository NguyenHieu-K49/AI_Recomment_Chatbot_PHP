import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI", "mysql+mysqlconnector://root:jgbXYBOIBYIcPPdJNlatuufCfsEjUfVT@switchback.proxy.rlwy.net:53764/railway")

try:
    # pool_recycle giúp giữ kết nối không bị timeout
    engine = create_engine(DB_URI, pool_recycle=3600)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with engine.connect() as conn:
        print(" Đã kết nối thành công đến MySQL!")
except Exception as e:
    print(f" Lỗi kết nối Database: {e}")
    sys.exit(1)

def get_db_connection():
    """Hàm lấy connection raw dùng cho Pandas đọc dữ liệu nhanh"""
    return engine.connect()