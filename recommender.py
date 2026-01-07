import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import get_db_connection
import pickle
import os

class HybridRecommender:
    def __init__(self):
        self.svd_model = None
        self.user_item_matrix = None
        self.user_index_map = {}
        self.product_index_map = {}
        self.reverse_product_index_map = {}
        
        self.content_matrix = None
        self.product_map = {} 
        self.product_id_to_idx = {} 
        self.user_ids = []
        self.model_path = "trained_model_shoes.pkl"
        
    def prepare_data(self):
        print("--> ĐANG ĐỒNG BỘ DỮ LIỆU TỪ MYSQL...")
        conn = get_db_connection()
        
        # 1. Lấy Full thông tin sản phẩm
        query_products = """
            SELECT p.product_id, p.product_name, p.base_price, p.description, 
                   b.brand_name, c.category_name, p.is_active
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN categories c ON p.category_id = c.category_id
            WHERE p.is_active = 1
        """
        df_products = pd.read_sql(query_products, conn)
        
        self.product_map = {}
        for _, row in df_products.iterrows():
            pid = str(row['product_id'])
            self.product_map[pid] = {
                "id": row['product_id'],
                "name": row['product_name'],
                "price": float(row['base_price']),
                "desc": row['description'],
                "brand": row['brand_name'],
                "category": row['category_name']
            }
        
        product_ids = list(self.product_map.keys())
        print(f"   - Đã index {len(product_ids)} sản phẩm active.")

        # 2. Lấy Lịch sử Mua hàng
        query_orders = """
            SELECT u.id as user_id, oi.product_id
            FROM users u
            JOIN customers c ON u.id = c.user_id
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
        """
        df_orders = pd.read_sql(query_orders, conn)
        
        df_orders['rating'] = 5 
        df_orders['product_id'] = df_orders['product_id'].astype(str)
        df_orders['user_id'] = df_orders['user_id'].astype(str)
        
        print(f"   - Tìm thấy {len(df_orders)} lượt mua hàng.")

        # 3. Train Collaborative Filtering (Dùng Sklearn SVD thay cho Surprise)
        if not df_orders.empty:
            # Tạo ma trận User-Item (Pivot Table)
            self.user_item_matrix = df_orders.pivot_table(
                index='user_id', 
                columns='product_id', 
                values='rating', 
                fill_value=0
            )
            
            # Lưu lại index để map ngược sau này
            self.user_index_map = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
            self.product_index_map = {pid: i for i, pid in enumerate(self.user_item_matrix.columns)}
            self.reverse_product_index_map = {i: pid for pid, i in self.product_index_map.items()}
            self.user_ids = list(self.user_item_matrix.index)

            # Matrix Factorization bằng TruncatedSVD
            # n_components: số lượng đặc trưng ẩn (latent features)
            n_components = min(20, len(self.product_index_map) - 1) 
            if n_components > 0:
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                self.svd_matrix = self.svd_model.fit_transform(self.user_item_matrix)
                print("   - Đã train xong mô hình hành vi (SVD Sklearn).")
            else:
                 print("   - Dữ liệu quá ít để train SVD.")
        
        # 4. Train Content-Based (TF-IDF)
        product_ids.sort()
        self.product_id_to_idx = {pid: i for i, pid in enumerate(product_ids)}
        
        corpus = []
        for pid in product_ids:
            p = self.product_map[pid]
            text = f"{p['name']} {p['brand']} {p['category']} {p.get('desc', '')}"
            corpus.append(text)
            
        if corpus:
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(corpus)
            self.content_matrix = cosine_similarity(tfidf_matrix)
            print("   - Đã tính toán xong độ tương đồng nội dung.")
            
        conn.close()
        return True

    def train_model(self):
        if self.prepare_data():
            self.save_model()
            return True
        return False

    def save_model(self):
        data = {
            'svd_model': self.svd_model,
            'user_item_matrix': self.user_item_matrix,
            'svd_matrix': getattr(self, 'svd_matrix', None),
            'user_index_map': self.user_index_map,
            'product_index_map': self.product_index_map,
            'reverse_product_index_map': self.reverse_product_index_map,
            'product_map': self.product_map,
            'user_ids': self.user_ids,
            'content_matrix': self.content_matrix,
            'product_id_to_idx': self.product_id_to_idx
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self):
        if not os.path.exists(self.model_path): return self.train_model()
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.svd_model = data.get('svd_model')
            self.user_item_matrix = data.get('user_item_matrix')
            self.svd_matrix = data.get('svd_matrix')
            self.user_index_map = data.get('user_index_map', {})
            self.product_index_map = data.get('product_index_map', {})
            self.reverse_product_index_map = data.get('reverse_product_index_map', {})
            self.product_map = data.get('product_map', {})
            self.user_ids = data.get('user_ids', [])
            self.content_matrix = data.get('content_matrix')
            self.product_id_to_idx = data.get('product_id_to_idx', {})
            return True
        except: return self.train_model()

    def recommend(self, user_id: str, n_items: int = 5):
        if not self.product_map: self.load_model()
        
        # Lấy lịch sử mua từ DB
        conn = get_db_connection()
        query = f"""
            SELECT oi.product_id 
            FROM users u
            JOIN customers c ON u.id = c.user_id
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE u.id = {user_id}
        """
        try:
            purchased = pd.read_sql(query, conn)
            purchased_ids = set(purchased['product_id'].astype(str).tolist())
        except: purchased_ids = set()
        conn.close()

        scores = []
        all_pids = list(self.product_map.keys())
        is_new_user = str(user_id) not in self.user_ids
        
        # A. Tính điểm CF (Hành vi) dùng SVD tái tạo ma trận
        cf_predictions = {}
        if not is_new_user and self.svd_model is not None:
            try:
                user_idx = self.user_index_map[str(user_id)]
                # Tái tạo lại hàng rating của user này: User_vector * Components
                user_vector = self.svd_matrix[user_idx].reshape(1, -1)
                predicted_ratings = np.dot(user_vector, self.svd_model.components_)
                
                # Map lại thành {product_id: score}
                for idx, score in enumerate(predicted_ratings[0]):
                    if idx in self.reverse_product_index_map:
                        pid = self.reverse_product_index_map[idx]
                        cf_predictions[pid] = score
            except Exception as e:
                print(f"Lỗi dự đoán CF: {e}")

        for pid in all_pids:
            if pid in purchased_ids: continue 
            
            # Điểm CF
            cf_score = cf_predictions.get(pid, 0)
            
            # Điểm Content
            content_score = 0
            if self.content_matrix is not None and purchased_ids:
                if pid in self.product_id_to_idx:
                    idx = self.product_id_to_idx[pid]
                    sims = []
                    for bought in purchased_ids:
                        if bought in self.product_id_to_idx:
                            b_idx = self.product_id_to_idx[bought]
                            sims.append(self.content_matrix[idx][b_idx])
                    if sims: content_score = np.mean(sims)
            
            # Hybrid
            final_score = content_score if is_new_user else (0.6 * cf_score + 0.4 * content_score)
            scores.append((pid, final_score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for pid, score in scores[:n_items]:
            p = self.product_map[pid]
            results.append({
                "id": p["id"],     
                "name": p["name"],
                "price": p["price"],
                "brand": p["brand"],
                "score": round(score, 4)
            })
        return results

recommender = HybridRecommender()