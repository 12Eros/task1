from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"D:\anaconda\envs\pytorch_env\Lib\site-packages\ultralytics\cfg\models\11\yolo11withSPDConv.yaml")
    model.train(
    data=r"F:\Visdrone_datasets\Visdrone.yaml",
    batch=8,
    imgsz=640,
    epochs=100,
    save_period=20,
    name='yolov11(changed)',
    device='0',
  )