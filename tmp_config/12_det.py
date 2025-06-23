train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=data_root + 'annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    num_workers=2,
    persistent_workers=True
)
