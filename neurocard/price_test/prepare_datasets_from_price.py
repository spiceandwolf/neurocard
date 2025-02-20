import pandas as pd

def infer_value_type(value):
    """推断单个值的类型"""
    # 处理空值或空字符串
    if pd.isnull(value) or str(value).strip() == '':
        return '空值'
    value_str = str(value).strip()
    
    # 尝试判断是否为整数
    try:
        int(value_str)
        return '整数'
    except ValueError:
        pass
    
    # 尝试判断是否为浮点数
    try:
        float(value_str)
        return '浮点数'
    except ValueError:
        pass
    
    # 其他情况视为字符串
    print(f"{value_str}")
    return '字符串'

def get_unique_value_types(csv_file, column_name):
    """获取指定列中不同值的类型"""
    # 读取CSV，确保目标列作为字符串处理
    df = pd.read_csv(csv_file, dtype={column_name: str}, sep='|', keep_default_na=False, na_values=['NULL'],)
    column_data = df[column_name]
    
    # 收集所有类型
    types = set()
    for value in column_data:
        # 处理可能的NaN值（转换为空字符串）
        if pd.isnull(value):
            value = ''
        types.add(infer_value_type(value))
    
    return types

# 示例用法
csv_file = '/home/user/oblab/PRICE/datas/datasets/accidents/nesreca.csv'  # 替换为你的CSV文件路径
column_name = 'oznaka_odsek_ali_ulica'  # 替换为目标列名
unique_types = get_unique_value_types(csv_file, column_name)

print(f"列 '{column_name}' 中的不同类型值：{unique_types}")