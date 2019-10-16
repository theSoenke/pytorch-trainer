import torch


class Trainer():
    def __init__(self, seed=0, gpu_id=0):
        self.gpu_id = gpu_id

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu_id}" if self.use_gpu else "cpu")

    def fit(self, model):
        self.optimizer = model.configure_optimizers()
        model.to(self.device)
        dataloader = model.train_dataloader()
        for batch in dataloader:
            if self.use_gpu:
                batch = self.transfer_batch_to_gpu(batch, self.gpu_id)
            output = model.training_step(batch)
            if 'loss' in output:
                output['loss'].backward()
            model.optimizer_step(self.optimizer)

    def validate(self, model):
        pass

    def test(self, model):
        pass

    def transfer_batch_to_gpu(self, batch, gpu_id):
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id)
        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id))
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        return batch
