import os
# 我们先看看原文件夹里面的数据
files = os.listdir('./data')
# print(files)

# 我们要打乱数据
# random 随机数
import random
random.shuffle(files)
# print(files)

# 数据乱了之后,我们需要存储到一个变量里面去
# 定义一个变量,是一个空的列表,用来存储的图片文件的
images = []
for file in files:
    # print(file)
    # 如果图片文件名的后三位是jpg,那么就往images变量盒子里面丢
    if file[-3:] == 'jpg':
        # print(file)
        # 将这个file图片往images里面丢
        images.append(file)
# print(images)
# 我们看看总共有多少张图片
images_count = len(images)
# print(images_count)

"""
计算各数据集里面的图片数量,按比例分配
    训练数据集 85%
    验证数据集 10%
    测试数据库 5%
"""
num_train = images_count * 0.85 # 训练数据集
num_valid = images_count * 0.1  # 验证数据集
num_test = images_count - num_train - num_valid # 剩下的就是测试数据集
print(num_train,num_valid,num_test)
# 将数据往dataset里面塞
import shutil # 文件操作的包,相当于文件搬运工
count = 0
for image in images:
    if count < num_train: # count 从0开始,到255都是拷贝到train文件夹里面
        # 拷贝的训练数据集
        shutil.copy("./data/"+image,"./dataset/images/train")
        # 拷贝标签,比如: 我们将这个1.jpg拷贝到dataset/images/train
        #              我们还要将1.txt拷贝到dataset/label/train
        #              我们只需要将jpg替换成txt就可以了
        shutil.copy("./data/"+image.replace("jpg","txt"),"./dataset/labels/train")
    elif count < num_valid + num_train:
        # 拷贝的是验证数据集
        shutil.copy("./data/"+image,"./dataset/images/valid")
        shutil.copy("./data/" + image.replace("jpg", "txt"), "./dataset/labels/valid")
    else:
        # 拷贝的是测试数据集
        shutil.copy("./data/" + image, "./dataset/images/test")
        shutil.copy("./data/" + image.replace("jpg", "txt"), "./dataset/labels/test")
    count += 1
