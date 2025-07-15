def dataset_info(dataset_name='vehicledata'):
    if dataset_name == 'vehicledata':
        train_path = "/storage/sjpark/vehicle_data/Dataset3/train_image/"
        ann_path = "/storage/sjpark/vehicle_data/Dataset3/ann_train/"
        val_path = '/storage/sjpark/vehicle_data/Dataset3/val_image/'
        val_ann_path = '/storage/sjpark/vehicle_data/Dataset3/ann_val/'
        test_path = '/storage/sjpark/vehicle_data/Dataset3/test_image/'
        test_ann_path = '/storage/sjpark/vehicle_data/Dataset3/ann_test/'
        json_file = '/storage/sjpark/vehicle_data/Dataset3/json_file/'
        num_class = 21
    else:
        raise NotImplementedError("Not Implemented dataset name")

    return dataset_name, train_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, num_class


def get_config_dict():
    dataset_name = "vehicledata"
    name, img_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, num_class, = dataset_info(dataset_name)


    dataset = dict(
        name=name,
        img_path=img_path,
        ann_path=ann_path,
        val_path=val_path,
        val_ann_path=val_ann_path,
        test_path=test_path,
        test_ann_path=test_ann_path,
        num_class=num_class,
        image_size = 224,
        size= (224, 224)
    )
    args = dict(
        gpu_id='0',
        batch_size=8,
        epochs=65,
        num_workers=6,
        network_name='Swin_Transformer'
    )
    solver = dict(
        output_stride = 16,
        optimizer="adam",
        scheduler='cycliclr',
        step_size=5,
        gamma=0.95,
        loss="crossentropy",
        lr=1e-4,
        weight_decay=5e-4,
        print_freq=20,
        deploy=False
    )

    model = dict(
        resume='',  # weight_file
        mode='train',
        save_dir='/storage/sjpark/vehicle_data/runs/Swin_transformer2/train/256/Swin_transformer-S',
        checkpoint='/storage/sjpark/vehicle_data/checkpoints/night_dataloader/Swin_transformer2'  # checkpoint_path
    )
    config = dict(
        args=args,
        dataset=dataset,
        solver=solver,
        model=model
    )

    return config
