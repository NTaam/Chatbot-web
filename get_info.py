from connect import get_in4Product_byCode

def get_in4(product_code):
    product_info = get_in4Product_byCode(product_code)
    if product_info:
        response = f"""
        Tên sản phẩm: {product_info['Tên sản phẩm: ']}<br>
        Số lượng sản phẩm: {product_info['Tổng số lượng size: ']}<br>
        Giá: {product_info['Giá: ']}<br><br>
        """
        return response
    return "Không tìm thấy thông tin sản phẩm."
