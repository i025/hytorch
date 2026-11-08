from pathlib import Path

from hytorch.recorder import MetricRecorder


class Callback:
    trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, step, logs=None):
        pass

    def on_batch_end(self, step, logs=None):
        pass


class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])

    def append(self, callback):
        self.callbacks.append(callback)

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs or {})

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs or {})

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs or {})

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs or {})

    def on_batch_begin(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(step, logs or {})

    def on_batch_end(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(step, logs or {})


class HistoryCallback(Callback):
    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        record = {"epoch": epoch}
        record.update(logs or {})
        self.history.append(record)


class MetricLoggerCallback(Callback):
    def __init__(self, recorder=None, log_batches=True):
        self.recorder = recorder if recorder is not None else MetricRecorder()
        self.log_batches = log_batches

    def on_train_begin(self, logs=None):
        self.recorder.log("event", event="train_begin", **(logs or {}))

    def on_train_end(self, logs=None):
        self.recorder.log("event", event="train_end", **(logs or {}))
        self.recorder.save_summary(logs or {})

    def on_batch_end(self, step, logs=None):
        logs = logs or {}
        if self.log_batches:
            self.recorder.log_batch(
                step=step,
                epoch=logs.get("epoch", 0),
                phase=logs.get("phase", "train"),
                **{k: v for k, v in logs.items() if k not in {"epoch", "phase"}},
            )

    def on_epoch_end(self, epoch, logs=None):
        self.recorder.log_epoch(epoch=epoch, phase="epoch", **(logs or {}))


class CheckpointCallback(Callback):
    def __init__(self, path=None, monitor="val_loss", mode="min", save_best_only=True, every=1):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.every = every
        self.best = None

    def _is_better(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best
        return value < self.best

    def _checkpoint_path(self, epoch, logs):
        if self.path is not None:
            return Path(str(self.path).format(epoch=epoch, **logs))
        recorder = getattr(self.trainer, "recorder", None)
        if recorder is not None:
            name = "best.npz" if self.save_best_only else f"epoch-{epoch}.npz"
            return recorder.checkpoint_path(name)
        return Path("checkpoints") / ("best.npz" if self.save_best_only else f"epoch-{epoch}.npz")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.every and epoch % self.every != 0:
            return
        if self.save_best_only:
            value = logs.get(self.monitor)
            if value is None or not self._is_better(value):
                return
            self.best = value
        path = self._checkpoint_path(epoch, logs)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.model.save_weights(path)


class EarlyStoppingCallback(Callback):
    def __init__(self, monitor="val_loss", mode="min", patience=5, min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0

    def _is_better(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        if self._is_better(value):
            self.best = value
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.trainer.stop_training = True


class VisualizationCallback(Callback):
    def __init__(self, visualizer=None):
        self.visualizer = visualizer

    def on_epoch_end(self, epoch, logs=None):
        if self.visualizer is None:
            return
        update = getattr(self.visualizer, "update", None)
        if callable(update):
            update(epoch, logs or {})
