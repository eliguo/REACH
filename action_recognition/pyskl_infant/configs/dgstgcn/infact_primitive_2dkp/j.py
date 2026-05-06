modality = 'j'
graph = 'coco'
work_dir = f'./work_dirs/dgstgcn/infact_primitive_2dkp/{modality}'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='DGSTGCN',
        num_person=1,
        gcn_ratio=0.125,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(
            layout=graph,
            mode='random',
            num_filter=8,
            init_off=.04,
            init_std=.02
        )
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=5,
        in_channels=256
    )
)

dataset_type = 'PoseDataset'
ann_file = '../Data/InfActPrimitive/2d/InfAct_plus_withZ_noscore.pkl'

train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=['j'], dataset='coco'),
    dict(type='UniformSample', clip_len=48),          # ← 改这里
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=['j'], dataset='coco'),
    dict(type='UniformSample', clip_len=48, num_clips=1),   # ← 改这里
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=['j'], dataset='coco'),
    dict(type='UniformSample', clip_len=48, num_clips=10),  # ← 改这里
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='val')
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 24

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

log_level = 'INFO'
find_unused_parameters = False
