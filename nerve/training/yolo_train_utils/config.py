cfg = dict()

# General YOLO options
cfg['YOLO'] = dict()
cfg['YOLO']['CLASSES_NUM'] = 1
cfg['YOLO']['ANCHORS'] = '/mnt/space/tang42/sensor_fusion_demo_dev/datasets/demo_poc_train_dataset/2023-10-25_14-26-59_rgb_pc/anchors_resize_258_128.txt'
cfg['YOLO']['STRIDES'] = [32]
cfg['YOLO']['ANCHOR_PER_SCALE'] = 3
cfg['YOLO']['IOU_LOSS_THRESHOLD'] = 0.5
cfg['YOLO']['MIN_DIST'] = 0.5
cfg['YOLO']['MAX_DIST'] = 5.0

# Training options
cfg['TRAIN'] = dict()
cfg['TRAIN']['ANNOT_PATH'] = '/mnt/space/tang42/sensor_fusion_demo_dev/datasets/demo_poc_train_dataset/2023-10-25_14-26-59_rgb_pc/image_info_resize_256_128.txt'
cfg['TRAIN']['BATCH_SIZE'] = 32
cfg['TRAIN']['INPUT_SIZE'] = (128, 256)
cfg['TRAIN']['LR_INIT'] = 1e-3
cfg['TRAIN']['LR_END'] = 1e-6
cfg['TRAIN']['WARMUP_EPOCHS'] = 2
cfg['TRAIN']['EPOCHS'] = 30
cfg['TRAIN']['QUANT_EPOCHS'] = 5

# Test options
cfg['TEST'] = dict()
cfg['TEST']['SCORE_THRESHOLD'] = 0.5
cfg['TEST']['IOU_THRESHOLD'] = 0.45

# Validation options
cfg['VAL'] = dict() 
cfg['VAL']['ANNOT_PATH'] = '/mnt/space/tang42/sensor_fusion_demo_dev/datasets/demo_poc_train_dataset/2023-10-26_16-00-58_rgb_pc/image_info_resize_256_128.txt'
cfg['VAL']['BATCH_SIZE'] = 32
