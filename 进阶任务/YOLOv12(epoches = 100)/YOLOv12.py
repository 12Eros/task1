from ultralytics import YOLO
model = YOLO("yolo12n.pt")


if __name__ == '__main__':
    model.train(
        data='D:/conda pytorch/datasets_custom/custom_datasets.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        name='yolov12',
        patience=100,
        erasing=0.4,
        device = 0,
        save_period = 10
    )
    model.val()