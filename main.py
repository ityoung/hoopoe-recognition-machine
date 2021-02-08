import cv2
import torch as t
import numpy as np
from torch.tensor import Tensor
from torchvision.transforms import transforms
from torchvision.transforms import ToPILImage

to_pil_image = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Scale((32, 32)), # resize
    # transforms.CenterCrop((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])


# todo: 训练模型
#   1. 读取训练集
#   2. 算法（网络定义）
#   3. 保存模型文件，输出验证集结果

def read_image(image_path):
    """读出图片数据"""
    img = transform(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR))
    img = np.reshape(img, (3, 32, 32))
    # debug: 输出图片
    # img = to_pil_image(img)
    # img.show()
    return Tensor(img).unsqueeze(0)


def is_dyson(image_path: str) -> bool:
    """接收图片，调用模型判断，返回结果"""
    model_path = ''
    # debug: 模型文件路径
    # model_path = 'my_model.pth'
    image = read_image(image_path)
    net = t.load(model_path)
    outputs = net(image)
    _, predict = t.max(outputs, 1)

    print(predict)
    return predict == "dyson"


if __name__ == '__main__':
    # TODO: train and save model
    # is_dyson('download.jpeg')
    print(read_image('download.jpeg'))
