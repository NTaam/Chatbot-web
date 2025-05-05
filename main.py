import torch
from model import IntentClassifier
from preprocessing import load_data_and_build_vocab, preprocess_text, tokenize
from train import predict_intent, get_response
from getProducts import get_products
from get_info import get_in4
from get_code import get_cfb
from utils import extract_product_code, extract_brand_from_input
json_path = 'data.json'
data, vocab, intent_labels, label_to_intent, train_data = load_data_and_build_vocab(json_path)
input_size = len(vocab)  # Đảm bảo rằng vocab đã được định nghĩa đúng
hidden_size = 128
embedding_dim = 64
num_classes = len(intent_labels)  # Đảm bảo intent_labels đã được định nghĩa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = IntentClassifier(
    input_size=input_size,
    hidden_size=hidden_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes
).to(device)

# Tải lại trọng số đã lưu
model.load_state_dict(torch.load("Nino.pth"))
model.eval()

#Định nghĩa trường hợp chat, cái này của Quang
# def chat(msg):
#     intent_tag = predict_intent(msg)
#     if intent_tag == "Hỏi sản phẩm": #trường hợp 1 số lượng sản phẩm
#         return get_response(intent_tag) + get_products()
#     #Trường hợp thông tin 1 sản phẩm được hỏi
#     response = get_response(intent_tag)
#     return response


# Cái này của Tâm
# def chat1(msg):
#     intent_tag1 = predict_intent(msg)
#     if intent_tag1 == "Hỏi thông tin theo code":
#         # Trích xuất tất cả mã sản phẩm từ câu hỏi
#         product_codes = extract_product_code(msg)  # Trích xuất danh sách mã sản phẩm
#         if product_codes:
#             response = get_response(intent_tag1)
#             for code in product_codes:
#                 response += get_in4(code)  # Thêm thông tin từng mã sản phẩm vào phản hồi
#             return response
#     response = get_response(intent_tag1)
#     return response

# Test
def chat(msg):
    # Dự đoán intent từ câu hỏi của người dùng
    intent_tag = predict_intent(msg)

    if intent_tag == "Hỏi sản phẩm": 
        return get_response(intent_tag) + get_products()
    elif intent_tag == "Hỏi thông tin theo code": 
        product_codes = extract_product_code(msg)
        if product_codes:
            response = get_response(intent_tag)
            for code in product_codes:
                response += get_in4(code)
            return response
    elif intent_tag == "hỏi mã theo brand":
        # brand = msg.split("brand")[-1].strip()  # Lấy phần thương hiệu sau từ khóa "brand" (có thể cần điều chỉnh thêm)
        # response = get_cfb(brand)
        # return response
        brand = extract_brand_from_input(msg)  # Hàm xử lý lấy thương hiệu từ câu hỏi
        if brand:
            return get_cfb(brand)
        else: 
            return "Xin lỗi, không tìm thấy thương hiệu của bạn!"
    response = get_response(intent_tag)
    return response


# Đang sai
# def chat(msg):
#     intent_tag = predict_intent(msg)
#     if intent_tag == "Hỏi sản phẩm":
#         return get_response(intent_tag) + get_products()
#     elif intent_tag == "Hỏi theo thông tin code":
#         product_codes = extract_product_code(msg)  
#         if product_codes:
#             response = get_response(intent_tag)
#             for code in product_codes:
#                 response += get_in4(code)
#             return response
#     response = get_response(intent_tag)
#     return response