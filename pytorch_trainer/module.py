from torch import nn, optim

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class Module(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_num):
        """
        return loss(optimal), dict with metrics to display
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_num):
        raise NotImplementedError

    def test_step(self, batch, batch_num):
        raise NotImplementedError

    def validation_end(self, outputs):
        raise NotImplementedError

    def test_end(self, outputs):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def optimizer_step(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()

    def backward(self, loss, optimizer, use_amp):
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def configure_apex(self, amp, model, optimizers, amp_level):
        model, optimizers = amp.initialize(
            model, optimizers, opt_level=amp_level,
        )

        return model, optimizers

    def summarize(self, input_size):
        summary(self, input_size, device="cpu")
