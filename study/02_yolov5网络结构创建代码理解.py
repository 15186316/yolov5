import torch
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.yolo import Model


print(sys.path)

def t0():
    cfg_path = "models\yolov5s_copy.yaml"
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目，不给定的时候直接使用cfg里面的nc
        anchors=None  # 给定初始的先验框大小（高度/宽度）
    )
    print(net)
    X = torch.rand(4, 3, 608, 608)
    r = net(X, augment=None)
    # print(r)
    print(type(r))
    print(r[0].shape)
    print(r[1].shape)
    print(r[2].shape)

    torch.onnx.export(
        model=net,
        args=(X,),
        f = 'yolov5s_copy_00.onnx',
        input_names=['image'],
        output_names=['labels'],
        opset_version=12
    )

def t1():
    cfg_path = "models\yolov5s_copy_02.yaml"
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目，不给定的时候直接使用cfg里面的nc
        anchors=None  # 给定初始的先验框大小（高度/宽度）
    )
    print(net)
    X = torch.rand(2, 3, 608, 608)
    r = net(X, augment=None)
    # print(r)
    print(type(r))
    # print(len(r))
    # for rr in r:
    #     print(rr.shape)

    # torch.onnx.export(
    #     model=net,
    #     args=(X,),
    #     f = 'yolov5s_copy_02.onnx',
    #     input_names=['image'],
    #     output_names=['labels'],
    #     opset_version=12
    # )

def t2():
    cfg_path = "models\yolov5s_copy_00.yaml"
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目，不给定的时候直接使用cfg里面的nc
        anchors=None  # 给定初始的先验框大小（高度/宽度）
    )
    print(net)
    X = torch.rand(2, 3, 608, 608)
    # 训练时候的返回值
    # na -->  number of anchor  每个锚点/grid 对应几个anchor box/预测边框
    # nc: class of number --> 类别数目
    # N: 批次样本大小； H:feature map的高度； W:feature map的宽度；
    # 训练时候返回各个分支/各层对应的预测值， [N, na, H, W, nc+1+4]
    net.train()
    r = net(X, augment=None)
    # print(r)
    print(type(r))
    for rr in r:
        print(rr.shape)

    # 推理时候的返回值
    # 推理预测时候，返回的是预测结果，是一个二元组
    # 二元组的第一个元素，就是模型推理预测结果： tensor对象， shape为：[N, na*H*W, nc+1+4]
        ### 每个样本、每个预测边框对应的置信度&回归系数
    # 二元组的第二个元素和训练时候返回的结果一样， list(tensor)的结构, tesor.shape为：[N, na, H, W, nc+1+4]
    with torch.no_grad():
        print("*" * 50)
        net.eval()
        r = net(X)
        print(type(r))


    # torch.onnx.export(
    #     model=net,
    #     args=(X,),
    #     f = 'yolov5s_copy_00.onnx',
    #     input_names=['image'],
    #     output_names=['labels'],
    #     opset_version=12
    # )

if __name__ =='__main__':
    t2()
