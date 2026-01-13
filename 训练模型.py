from ultralytics import YOLO

model = YOLO("yolo.pt")

if __name__ == '__main__':
    #训练这个模型，训练五十轮
    model.train(data = "./dataset/mydata.yaml",epochs = 50)
    #在验证集上评估这个模型的性能
    m = model.val()

    # 验证集有几个指标 => 精度,平均精度,精确率,召回率
    # 获取精度 => 你大致圈出来苹果的位置,虽然不够精确,但是框柱苹果
    print(f"精度:{m.box.map50}:.4f")
    # 平均精度 => 50:大致框柱的50%的苹果,75:大致框柱的75%的苹果,90%几乎完美的框柱的所有的苹果轮廓
    print(f"精度:{m.box.map}:.4f")
    # 精确率 => 默认认为评估的框柱为苹果的比例
    print(f"精确率:{m.box.map}")
    # 召回率 => 所有的真实苹果中,被检测出来的苹果的比例
    print(f"召回率:{m.box.mr}")

    # 用训练好的模型来识别一张图片

    results = model("./dataset/images/test/img56.jpg")
    #处理检测结果
    for result in results:
        #在屏幕上显示这个结果
        result.show()
