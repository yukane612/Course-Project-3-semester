from ultralytics import YOLO
model = YOLO("best.pt")
if __name__ == '__main__':
    # 模型预测的结果
    results = model("./dataset/images/test/img235.jpg",imgsz=640)
    # 处理检测出来的结果
    for result in results:
        # 如果有检测到物品,那么 result.boxes 的长度是大于0的
        if len(result.boxes) > 0:
            # 检测第一个物品的编号
            cls_index = int(result.boxes.cls[0])
            # 查看置信度
            conf = float(result.boxes.conf[0])
            print(f"置信度:{conf}")
            print(f"检测到的物品的编号是:{cls_index},物品是:{result.names[cls_index]}")
        else:
            print("没有检测到物品")
