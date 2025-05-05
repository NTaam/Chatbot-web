from connect import get_brand_info
def get_cfb(brand):
    brand_data = get_brand_info(brand)

    if brand_data is None:
        return "Không tìm thấy thương hiệu này."

    # Format thông tin để hiển thị
    result = f"<b>Thương hiệu: {brand}</b><br>"
    for _, row in brand_data.iterrows():
        result += f"Mã sản phẩm: {row['ProductCode']}<br>"
    result += "<br>Đây là các mã sản phẩm mà shop em hiện đang có."

    return result