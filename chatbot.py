import os
import google.generativeai as genai
import pandas as pd
from database import get_db_connection
from recommender import recommender
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ================= 1. CÔNG CỤ TRA CỨU SẢN PHẨM =================
def search_products(query: str):
    """
    Tìm sản phẩm theo tên/hãng/loại và KIỂM TRA TỒN KHO theo size.
    """
    print(f"--> [BOT] Tìm sản phẩm: {query}")
    conn = get_db_connection()
    try:
        sql = f"""
            SELECT p.product_name, p.base_price, b.brand_name, 
                   GROUP_CONCAT(CONCAT(i.size, '(', i.quantity, ')') SEPARATOR ', ') as stock_info
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN inventory i ON p.product_id = i.product_id
            WHERE (p.product_name LIKE '%%{query}%%' 
               OR b.brand_name LIKE '%%{query}%%'
               OR c.category_name LIKE '%%{query}%%')
               AND p.is_active = 1
            GROUP BY p.product_id
            LIMIT 3
        """
        df = pd.read_sql(sql, conn)
        if df.empty: return "Kho không tìm thấy sản phẩm nào khớp yêu cầu."
        
        res = "Kết quả tìm kiếm:\n"
        for _, row in df.iterrows():
            res += f"👟 {row['product_name']} ({row['brand_name']})\n"
            res += f"   💰 Giá: ${row['base_price']}\n"
            res += f"   📦 Size tồn kho: {row['stock_info']}\n"
            res += "----------------\n"
        return res
    except Exception as e: return f"Lỗi DB: {str(e)}"
    finally: conn.close()

# ================= 2. CÔNG CỤ TRA CỨU ĐƠN HÀNG CỤ THỂ (Theo ID) =================
def track_order(order_id: str):
    """Tra cứu chi tiết tình trạng đơn hàng khi biết mã đơn."""
    conn = get_db_connection()
    try:
        sql_order = f"SELECT * FROM orders WHERE order_id = {order_id}"
        order = pd.read_sql(sql_order, conn)
        
        if order.empty: return "Không tìm thấy mã đơn hàng này."
        
        sql_items = f"""
            SELECT p.product_name, oi.quantity, oi.size, oi.subtotal
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            WHERE oi.order_id = {order_id}
        """
        items = pd.read_sql(sql_items, conn)
        
        item_str = "\n".join([f"   + {r['quantity']}x {r['product_name']} (Size {r['size']})" for _, r in items.iterrows()])
        
        o = order.iloc[0]
        return f"""
        📦 Đơn hàng #{o['order_id']}
        📅 Ngày đặt: {o['order_date']}
        🚚 Trạng thái: {o['status'].upper()}
        💵 Tổng tiền: ${o['total_amount']}
        🛒 Chi tiết:
        {item_str}
        """
    except Exception as e: return "Lỗi tra cứu: " + str(e)
    finally: conn.close()

# ================= 3. (MỚI) CÔNG CỤ LẤY DANH SÁCH ĐƠN HÀNG CỦA TÔI =================
def get_my_orders(user_id: str):
    """
    Tra cứu danh sách đơn hàng của người dùng hiện tại.
    Chỉ cần user_id, không cần mã đơn.
    """
    print(f"--> [BOT] Check đơn hàng cho User ID: {user_id}")
    conn = get_db_connection()
    try:
        # Join User -> Customer -> Orders
        # Lấy 5 đơn gần nhất
        sql = f"""
            SELECT o.order_id, o.order_date, o.status, o.total_amount, COUNT(oi.product_id) as item_count
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN users u ON c.user_id = u.id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            WHERE u.id = {user_id}
            GROUP BY o.order_id
            ORDER BY o.order_date DESC
            LIMIT 5
        """
        df = pd.read_sql(sql, conn)
        
        if df.empty:
            return "Bạn hiện tại chưa có đơn hàng nào trong lịch sử."
            
        res = f"📋 Danh sách đơn hàng gần đây của bạn (User {user_id}):\n"
        for _, row in df.iterrows():
            status_icon = "✅" if row['status'] == 'delivered' else "🚚" if row['status'] == 'shipped' else "⏳"
            res += f"{status_icon} Đơn #{row['order_id']} ({row['order_date']}) - {row['status']}\n"
            res += f"   Tổng: ${row['total_amount']} ({row['item_count']} sản phẩm)\n"
        
        res += "\n💡 Bạn muốn xem chi tiết đơn nào thì nhắn 'Xem chi tiết đơn số X' nhé!"
        return res
    except Exception as e: return f"Lỗi lấy danh sách đơn: {str(e)}"
    finally: conn.close()

# ================= 4. CÔNG CỤ TRA CỨU VOUCHER =================
def lookup_vouchers():
    """Tra cứu mã giảm giá."""
    conn = get_db_connection()
    try:
        sql = "SELECT coupon_code, description, discount_value, discount_type FROM coupons WHERE is_active = 1 AND end_date >= CURDATE() LIMIT 5"
        df = pd.read_sql(sql, conn)
        if df.empty: return "Hiện không có voucher nào."
        res = "🎟 Voucher HOT:\n"
        for _, r in df.iterrows():
            val = f"${r['discount_value']}" if r['discount_type'] == 'fixed_amount' else f"{r['discount_value']}%"
            res += f"- {r['coupon_code']}: {r['description']} (Giảm {val})\n"
        return res
    except: return "Lỗi voucher."
    finally: conn.close()

# ================= 5. GỢI Ý CÁ NHÂN =================
def get_personal_recommendations(user_id: str):
    """Gợi ý sản phẩm."""
    try:
        items = recommender.recommend(user_id, n_items=4)
        if not items: return "Shop có nhiều mẫu Nike, Adidas mới về, bạn xem thử nhé!"
        res = "🌟 Gợi ý riêng cho bạn:\n"
        for i in items: res += f"- {i['name']} (${i['price']})\n"
        return res
    except: return "Hệ thống bận."

# ================= SETUP MODEL =================
# Thêm tool mới get_my_orders vào danh sách
tools_list = [search_products, track_order, get_my_orders, lookup_vouchers, get_personal_recommendations]

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    tools=tools_list,
    system_instruction="""
    Bạn là Trợ lý Ảo TechShop.
    - Nhiệm vụ: Hỗ trợ tìm sản phẩm, tra cứu đơn hàng, voucher.
    - Phong cách: Thân thiện, ngắn gọn, dùng Emoji.
    
    QUY TẮC QUAN TRỌNG:
    1. Nếu khách hỏi "đơn hàng của tôi", "kiểm tra đơn hàng", "lịch sử mua"... -> GỌI TOOL `get_my_orders(user_id)`.
       (Tuyệt đối KHÔNG được hỏi lại mã đơn hàng, hãy dùng User ID có sẵn trong context).
    2. Nếu khách hỏi chi tiết 1 đơn cụ thể (VD: đơn số 10) -> GỌI TOOL `track_order(order_id)`.
    3. Nếu khách hỏi sản phẩm -> GỌI TOOL `search_products`.
    4. Nếu khách hỏi voucher -> GỌI TOOL `lookup_vouchers`.
    """
)

chat_sessions = {}

def get_user_name(user_id):
    """Hàm phụ: Lấy tên khách hàng để chào cho thân thiện"""
    conn = get_db_connection()
    try:
        sql = f"SELECT first_name FROM customers WHERE user_id = {user_id}"
        df = pd.read_sql(sql, conn)
        if not df.empty: return df.iloc[0]['first_name']
    except: pass
    finally: conn.close()
    return "Bạn"

def chat_process(user_id: str, message: str) -> str:
    global chat_sessions
    
    # Khởi tạo session chat nếu chưa có
    if user_id not in chat_sessions:
        # Lấy tên khách để Bot biết đường xưng hô
        customer_name = get_user_name(user_id)
        
        # Mẹo: Tạo lịch sử giả để "mớm" context cho Gemini biết user_id là bao nhiêu
        # Nhờ vậy, khi tool get_my_orders được gọi, Gemini sẽ tự điền user_id vào.
        history = [
            {
                "role": "user",
                "parts": [f"Xin chào, tôi là khách hàng có User ID: {user_id}. Tên tôi là {customer_name}."]
            },
            {
                "role": "model",
                "parts": [f"Chào {customer_name}! Mình đã ghi nhận User ID {user_id} của bạn. Mình có thể giúp gì cho bạn?"]
            }
        ]
        chat_sessions[user_id] = model.start_chat(history=history, enable_automatic_function_calling=True)
    
    try:
        response = chat_sessions[user_id].send_message(message)
        return response.text
    except Exception as e:
        # Nếu lỗi session (do để lâu quá), reset lại
        if "400" in str(e) or "session" in str(e).lower():
            del chat_sessions[user_id]
            return "Xin lỗi, phiên chat bị gián đoạn. Bạn nhắn lại giúp mình nhé!"
        return f"Hệ thống đang bận: {str(e)}"