import wandb


class WandbLogger():
    def __init__(self, project, model):
        wandb.init(project=project)
        wandb.watch(model)

    def log_hyperparams(self, params):
        wandb.config.update(params)

    def log_metrics(self, metrics, step_num=0):
        wandb.log(metrics)

    def log(self, key, value):
        wandb.config[key] = value

    def save_file(self, path):
        wandb.save(path)
