from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from getProducts import get_products
from main import chat

app = FastAPI()
#Cấp quyền cho mọi miền
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# Khởi tạo API
# Lấy nội dung tin nhắn từ người dùng thông qua trang web
class ChatInput(BaseModel):
    message: str
@app.get("/")
async def root():
    return {"message":"Success!"}
@app.post("/chat")
async def chat_api(input_data: ChatInput):
    user_input = input_data.message
    # response = chat(user_input)
    response = chat(user_input)

    return {"response": response}
