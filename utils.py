import numpy as np
import matplotlib.pyplot as plt

class Logger:
  def __init__(self):
    self.epoch_loss = []
    self.epoch_acc = []
    # self.step_loss = []
    # self.step_acc = []
    self.samples = 0
    self.loss_mean = 0
    self.acc_mean = 0

  def log_step(self, loss, acc, samples):
    # self.step_loss.append(loss)
    # self.step_acc.append(acc)
    if self.samples == 0:
      self.loss_mean = loss
      self.acc_mean = acc
    else:
      self.loss_mean = self.loss_mean / (1+samples/self.samples) + loss / (1+self.samples/samples)
      self.acc_mean = self.acc_mean / (1+samples/self.samples) + acc / (1+self.samples/samples)
    self.samples += samples

  def log_epoch(self):
    self.epoch_loss.append(self.loss_mean)
    self.epoch_acc.append(self.acc_mean)
    self.samples = 0
    self.loss_mean = 0
    self.acc_mean = 0

  def plot(self, mode='epoch', path=None):
    if (mode=='step'):
      raise NotImplementedError
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax1.plot(np.arange(len(self.epoch_loss))+1, self.epoch_loss, linewidth=1, linestyle="solid", label="loss")
    ax1.legend()
    ax1.set_title('Loss Curve')
    ax2.plot(np.arange(len(self.epoch_acc))+1, self.epoch_acc, linewidth=1, linestyle="solid", label="acc")
    ax2.legend()
    ax2.set_title('Accuracy Curve')
    plt.show()
    if path is not None:
      plt.savefig(path)