https://drive.google.com/file/d/1cmUzFz0x90zkcqrbzv1_Wu2YiU37n1X-/view?usp=sharing

### 数据增强方法

训练集和测试集的同一类别鸟主体的位置，方向，大小等差别较大，需要通过数据增强手段来弥补训练集中样本多样性不足的问题。

实验中使用到的数据变换包括：随机裁剪、随机水平翻转，随机色调改变、随机旋转等

```python
  train_transform = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomResizedCrop(size=(input_size, input_size), scale=(0.4, 1), ratio=(0.5, 2)),
      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
      transforms.RandomRotation(20),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # image_net
  ])

  test_transform = transforms.Compose([
      transforms.Resize((infer_size, infer_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # image_net
  ])
```

为了在测试集上获得更好的效果，测试时的图像大小为256略大于训练大小224

### 优化方法

实验中选用nesterov动量的SGD优化器进行梯度下降，动量参数使得SGD下降的方向受噪声影响较小，nesterov提前估计使得整体优化过程收敛更快

实验中设置基础的学习率为1e-2，微调主干部分的学习率为1e-3，参数正则项weight_decay为1e-4

```yaml
optimizer:
  lr: 0.01
  momentum: 0.9
  name: SGD
  weight_decay: 0.0001
  nesterov: True
```

此外，实验中选用余弦退火的方式调节学习率变化，余弦值变化的初始周期为200

```yaml
scheduler:
  name: Cosine
  T_0: 200
```

### 数据加载

实验过程中，模型训练的主要瓶颈在于数据加载，为了充分利用多核cpu的性能，实验中使用多个线程对数据进行预加载

```yaml
dataloader:
  batch_size: 16
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

batch_size经过多次试验后设置为16，以获得最好的实验效果
