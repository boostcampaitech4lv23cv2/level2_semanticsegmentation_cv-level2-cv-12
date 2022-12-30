log_config = dict(  # config to register logger hook
    interval=654,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='MMSegWandbHook', by_epoch=True, # The Wandb logger is also supported, It requires `wandb` to be installed.
             init_kwargs={'entity': "cv12-semantic-seg", # The entity used to log on Wandb
                          'project': "hrnet", # Project name in WandB
                          'name': 'ocrnet_hr48'}), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        # MMSegWandbHook is mmseg implementation of WandbLoggerHook. ClearMLLoggerHook, DvcliveLoggerHook, MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook, SegmindLoggerHook are also supported based on MMCV implementation.
    ])

dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
