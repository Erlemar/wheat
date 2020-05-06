import pytorch_lightning as pl


class MetricsCallback(pl.Callback):
    """
    PyTorch Lightning metric callback.

    Use it like this:
    >>> metrics_callback = MetricsCallback()

    >>> trainer = pl.Trainer(callbacks=[metrics_callback])
    >>> trainer.fit(model)

    >>> print(metrics_callback.metrics)

    """

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
