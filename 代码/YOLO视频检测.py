from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model(source = r"F:\DownKyi-1.0.20-1.win-x64\B站下载视频\全网最帅的城市飘移大片！\2-全网最帅的城市飘移大片！-1080P 高清-AVC.mp4"
                         ,show =True)