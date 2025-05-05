from connect import get_listProducts
def get_products():
    df = get_listProducts()
    response = ""
    for _, row in df.iterrows():
        # print(f"Kiểu quần áo: {row['Kiểu quần áo']}, Hãng: {row['Hãng sản phẩm']}, Số lượng: {row['Số lượng còn lại']}")
        response += f"Kiểu quần áo: {row['Kiểu quần áo']}<br>Hãng: {row['Hãng sản phẩm']}<br>Số lượng: {row['Số lượng còn lại']}<br><br>"
    return response