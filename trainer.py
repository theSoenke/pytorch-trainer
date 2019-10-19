from tqdm import tqdm

import torch


class Trainer():
    def __init__(self, seed=0, gpu_id=0, num_max_epochs=100, checkpoint_callback=None):
        self.gpu_id = gpu_id
        self.num_max_epochs = num_max_epochs
        self.checkpoint_callback = checkpoint_callback

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu_id}" if self.use_gpu else "cpu")
        self.current_epoch = 0

    def fit(self, model):
        self.model = model
        self.optimizer = self.model.configure_optimizers()
        self.model.to(self.device)
        self.model.train()
        dataloader = model.train_dataloader()
        outputs = []

        for epoch in range(self.num_max_epochs):
            with tqdm(total=len(dataloader)) as pbar:
                for batch in dataloader:
                    pbar.set_description(f"Epoch {epoch}")
                    if self.use_gpu:
                        batch = self.transfer_batch_to_gpu(batch, self.gpu_id)
                    output = model.training_step(batch)
                    outputs.append(output)
                    if 'loss' in output:
                        output['loss'].backward()
                    # pbar.set_postfix({"loss": 1.0})
                    model.optimizer_step(self.optimizer)
                    pbar.update(1)
            self.on_epoch_end(epoch, outputs)

    @torch.no_grad()
    def validate(self, model):
        dataloader = model.val_dataloader()
        model.eval()

        outputs = []
        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                pbar.set_description("Validation")
                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch, self.gpu_id)
                output = model.validation_step(batch)
                outputs.append(output)
                pbar.update(1)
        eval_results = self.model.validation_end(outputs)
        # self.logger.log_metrics(eval_results['log'])
        return eval_results

    def test(self, model):
        pass

    def __process_logs(self, logs):
        metrics = {}
        for key, value in logs.items():
            if key == 'log':
                continue
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value

        return metrics

    def on_epoch_end(self, epoch, outputs):
        logs = self.validate(self.model)
        processed_logs = self.__process_logs(logs)
        if self.checkpoint_callback != None:
            self.checkpoint_callback.on_epoch_end(epoch, save_func=self.save_checkpoint, logs=processed_logs)

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

    def save_checkpoint(self, filepath):
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
