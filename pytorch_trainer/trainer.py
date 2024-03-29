import random

import torch
from tqdm import tqdm

from .model_summary import ModelSummary

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class Trainer():
    def __init__(
        self,
        seed=0,
        gpu_id=0,
        epochs=100,
        checkpoint_callback=None,
        early_stop_callback=None,
        logger=None,
        use_amp=False,
        val_percent=1.0,
        test_percent=1.0,
    ):
        self.seed = seed
        self.gpu_id = gpu_id
        self.epochs = epochs
        self.checkpoint_callback = checkpoint_callback
        self.early_stop_callback = early_stop_callback
        self.logger = logger
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.current_epoch = 0
        self.scheduler = None

        self.use_amp = False
        if use_amp:
            if not APEX_AVAILABLE:
                self.use_amp = False
                print("apex is not installed")
            else:
                self.use_amp = True

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu_id}" if self.use_gpu else "cpu")

    def fit(self, model, checkpoint=None):
        print(ModelSummary(model, mode='top'))

        self.model = model
        self.model.trainer = self
        optimizer = self.model.configure_optimizers()
        if isinstance(optimizer, tuple):
            self.optimizer = optimizer[0]
            self.scheduler = optimizer[1]
        else:
            self.optimizer = optimizer
            self.scheduler = None

        self.model.to(self.device)
        self.__load_checkpoint(checkpoint, self.model, self.optimizer)
        if self.use_amp:
            self.model, self.optimizer = self.model.configure_apex(amp, self.model, self.optimizer, "O1")
        self.model.train()
        dataloader = model.train_dataloader()
        samples = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        self.validate(self.model, fast_validate=True)
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.on_epoch_start(epoch)
            with tqdm(total=samples) as pbar:
                pbar.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")
                for i, batch in enumerate(dataloader):
                    if self.use_gpu:
                        batch = self.transfer_batch_to_gpu(batch)
                    output = self.model.training_step(batch, i)
                    self.model.backward(output['loss'], self.optimizer, self.use_amp)
                    self.model.optimizer_step(self.optimizer)

                    logs = self.__process_logs(output)
                    pbar.set_postfix(logs)
                    self.__log_metrics(output)
                    processed = min((i + 1) * batch_size, samples)
                    pbar.n = processed

            if self.val_percent > 0.0:
                logs = self.validate(self.model)
                self.__log_metrics(logs)
            else:
                print("Skipping validation")
                logs = None
            logs = self.__process_logs(logs)
            self.__create_checkpoint(logs)
            if self.early_stop_callback != None:
                stop_training = self.early_stop_callback.on_epoch_end(epoch=epoch, logs=logs)
                if stop_training:
                    break

            self.model.on_epoch_end(epoch)

    @torch.no_grad()
    def validate(self, model, fast_validate=False, checkpoint=None):
        model.trainer = self
        model.to(self.device)
        self.__load_checkpoint(checkpoint, model)
        model.eval()
        dataloader = model.val_dataloader()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset) * self.val_percent)
            max_batches = int(len(dataloader) * self.val_percent)

        outputs = []
        description = 'Check validation step' if fast_validate else 'Validation'
        with tqdm(total=samples) as pbar:
            for i, batch in enumerate(dataloader):
                pbar.set_description(description)

                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch)
                output = model.validation_step(batch, i)
                outputs.append(output)
                processed = min((i + 1) * batch_size, samples)
                pbar.n = processed

                if i >= max_batches:
                    break

        model.train()
        if fast_validate:
            return {}
        results = model.validation_end(outputs)
        return results

    @torch.no_grad()
    def test(self, model, checkpoint=None):
        model.trainer = self
        model.to(self.device)
        self.__load_checkpoint(checkpoint, model)
        model.eval()
        dataloader = model.test_dataloader()
        samples = int(len(dataloader.dataset) * self.test_percent)
        batch_size = dataloader.batch_size
        max_batches = int(len(dataloader) * self.test_percent)

        outputs = []
        with tqdm(total=samples) as pbar:
            for i, batch in enumerate(dataloader):
                pbar.set_description("Test")
                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch)
                output = model.test_step(batch, i)
                outputs.append(output)
                processed = min((i + 1) * batch_size, samples)
                pbar.n = processed

                if i >= max_batches:
                    break

        model.train()
        results = model.test_end(outputs)
        return results

    def __process_logs(self, logs):
        metrics = {}
        logs = logs or {}
        for key, value in logs.items():
            if key == 'log':
                continue
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value

        return metrics

    def __log_metrics(self, outputs):
        if self.logger != None and 'log' in outputs:
            processed_logs = self.__process_logs(outputs['log'])
            self.logger.log_metrics(processed_logs)

    def __create_checkpoint(self, logs=None):
        logs = logs or {}
        if self.checkpoint_callback != None:
            self.checkpoint_callback.on_epoch_end(self.current_epoch, save_func=self.save_checkpoint, seed=self.seed, logs=logs)

    def __load_checkpoint(self, checkpoint, model, optimizer=None):
        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None and 'state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def transfer_batch_to_gpu(self, batch):
        if callable(getattr(batch, 'to', None)):
            return batch.to(self.device)
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x)
            return batch
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x)
            return tuple(batch)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v)

            return batch

        return batch

    def save_checkpoint(self, filepath):
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
