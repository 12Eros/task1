from ultralytics import YOLO
# import os
# os.environ['GITHUB_ASSETS'] = 'https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/'
# # 使用国内镜像源下载
model = YOLO("yolov8n.pt")
dir_list = [r'D:\conda pytorch\datasets_custom\images\train\22.png',
            r'D:\conda pytorch\datasets_custom\images\train\10(1).png',
            r'D:\conda pytorch\datasets_custom\images\train\4(1).png']
results = model(source = dir_list,show =True)
# results作为一个列表，其中的每一个元素代表着一个对输入的检测结果
for i in range(len(results)):
    results[i].save()
for result in results:
    print("----------------------------------------------------------")
# 获取这个检测结果
# result类有很多属性包括orig_shape,shape,speed,boxes(检测框信息)
    position = result.boxes.xyxy
    print("检测框的位置:",position)
    position_ = result.boxes.xywh
    print("检测框的大小:",position_)
    confidence = result.boxes.conf
    print("置信度:",confidence)
    print(result)

