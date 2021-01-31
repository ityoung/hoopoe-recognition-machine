import torch
import torchvision
# print(torchvision.__version__)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# todo: 训练模型
#   1. 读取训练集
#   2. 算法
#   3. 保存模型文件，输出验证集结果


# todo：接收图片，调用模型判断，返回结果
def is_dyson(image_path: str) -> bool:
    """接收图片，调用模型判断，返回结果"""
if __name__ == '__main__':
    print(unpickle("data_batch_1"))