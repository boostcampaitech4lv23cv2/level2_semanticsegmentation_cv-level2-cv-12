from torch.optim import lr_scheduler
import json

_scheduler_entrypoints = {
    "step": lr_scheduler.StepLR,
    "multi_step": lr_scheduler.MultiStepLR,
    "cosine_annealing": lr_scheduler.CosineAnnealingLR
}

def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]

def is_criterion(scheduler_name):
    return scheduler_name in _scheduler_entrypoints

def create_scheduler(optimizer, scheduler_name):
    if is_criterion(scheduler_name):
        with open('./modules/configs/scheduler_conf.json') as f:
            scheduler_args = json.load(f)[scheduler_name]
        
        create_fn = scheduler_entrypoint(scheduler_name)
        scheduler = create_fn(optimizer, **scheduler_args)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    return scheduler