_base_ = [
    '_base_/models/fcn_hr18.py',
    '_base_/datasets/custom_dataset.py',
    './logger.py'
    # './default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),
        num_classes=11))

load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
cudnn_benchmark = True  # Whether use cudnn_benchmark to speed up, which is fast for fixed input size.

learning_rate = 1e-4
optimizer = dict(type='Adam', lr=learning_rate, weight_decay=learning_rate*0.1)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 20, 30])
runner = dict(type='EpochBasedRunner', max_epochs=40,)
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=-1)  # The save interval.

evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

