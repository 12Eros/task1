from ultralytics import YOLO


def main():
    # 加载模型
    model = YOLO('yolov8n.pt')

    # 训练模型
    results = model.train(
        data='D:/conda pytorch/datasets_custom/custom_datasets.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='custom_yolo',
        patience=100,
        erasing = 0.4,
        device= 0,
        save_period = 10
    )
    metrics = model.val()
    print(f"mAP50:{metrics.box.map50}")
    print(f"mAP50-95:{metrics.box.map}")

if __name__ == '__main__':
    # 在Windows上运行多进程时需要这个
    import multiprocessing
    multiprocessing.freeze_support()
    main()