from datetime import timedelta
from setproctitle import setproctitle
from time import time


class PytorchProcessName:
    def __init__(self, total_epochs, name="ML"):
        self.name = name
        self.epochs = total_epochs
        self.times = []
        self.epoch = 0

        setproctitle(self.name)

    def start(self):
        self.times.append(time())

    def update_epoch(self, epoch):
        self.times.append(time())

        if len(self.times) == 1:
            remaining = "estimating..."
        else:
            avg_epoch_duration = np.ediff1d(np.array(self.times)[-10:]).mean()
            epochs_left = self.epochs - epoch
            time_left_in_secs = np.ceil(avg_epoch_duration * epochs_left)
            remaining = timedelta(seconds=time_left_in_secs)

        proc_name = self.name + ' remaining: %s' % remaining

        setproctitle(proc_name)