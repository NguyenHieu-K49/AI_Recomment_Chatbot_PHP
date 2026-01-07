from fastapi import FastAPI
from pydantic import BaseModel
from recommender import recommender 
from chatbot import chat_process
from apscheduler.schedulers.asyncio import AsyncIOScheduler

app = FastAPI(title="Shoe Shop AI Service")
scheduler = AsyncIOScheduler()

class ChatRequest(BaseModel):
    user_id: str 
    message: str

@app.on_event("startup")
async def startup():
    print("🚀 AI Service Starting...")
    recommender.load_model()
    # Tự động học lại dữ liệu mới mỗi đêm lúc 3h sáng
    scheduler.add_job(recommender.train_model, "cron", hour=3)
    scheduler.start()

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """API Chatbot cho Laravel gọi"""
    reply = chat_process(req.user_id, req.message)
    return {"reply": reply}

@app.get("/api/recommend/{user_id}")
async def recommend_endpoint(user_id: str):
    """API Gợi ý sản phẩm"""
    items = recommender.recommend(user_id)
    return {"status": "success", "data": items}

@app.post("/api/train")
async def force_train():
    """API ép AI học lại ngay lập tức"""
    recommender.train_model()
    return {"status": "success", "message": "Model re-trained!"}