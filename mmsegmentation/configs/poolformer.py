_base_ = [
    './_base_/models/fpn_poolformer_s12.py', 
    './_base_/datasets/custom_dataset.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py',
    './logger.py'
    ]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth'  # noqa
# load_from = '/opt/ml/input/code/pretrained/fpn_poolformer.pth'

# model settings
model = dict(
    backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
            ),
    neck=dict(in_channels=[96, 192, 384, 768]))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()
# learning policy
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(
    _delete_=True,
    type='EpochBasedRunner', 
    max_epochs=100,
)

checkpoint_config = dict(
    _delete_=True,
    interval=-1,
)

evaluation = dict(
    _delete_=True,
    interval=1, 
    metric='mIoU', 
    save_best='mIoU'
)

data = dict(samples_per_gpu=4, workers_per_gpu=4)
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')