# Deep Learning Lab6

## 1. Introduction

Implement a conditional Denoising Diffusion Probabilistic Model (DDPM) to generate synthetic images according to multi-label conditions. To achieve higher generation capacity, especially in computer vision, DDPM is proposed and has been widely applied to style transfer and image synthesis.  For example, given “blue cube” and “yellow cylinder”, model should generate the synthetic images with a blue cube and a yellow cylinder and meanwhile, input your generated images to a pre-trained classifier for evaluation.

## 2. Implementation details 

### A. Model setting details 

-   model: diffusers.UNet2DModel
-   noise scheduler: DDPMScheduler
-   loss function: nn.MSELoss
-   learning rate scheduler: get_cosine_schedule_with_warmup

#### UNet

```python
UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    class_embed_type=None,
    # num_class_embeds = 2325, #C (24, 3) + 1
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        # a regular ResNet upsampling block
        "UpBlock2D",
        # a ResNet upsampling block with spatial self-attention
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
model.class_embedding = nn.Linear(24, 512)
```

I use diffusers UNet2DModel as my backbone. UNet has a series of dwonsampling and upsampling operations. The model is referenced from the [link](https://zhuanlan.zhihu.com/p/634358636)

#### DataLoader

I copy from Lab2 LeukemiaLoader, and change transformation

```python
self.transforms = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
```

#### Train

```python
for img, label in (pbar := tqdm(self.train_loader)):
    inputs = img.to(self.device, dtype=torch.float32)
    labels = label.to(self.device, dtype=torch.float32).squeeze()

    noise = torch.randn(inputs.shape).to(self.device)
    timesteps = torch.randint(
        0, self.timestep - 1, (inputs.shape[0],),
        dtype=torch.long, device=self.device
    )
    noisy_inputs = self.noise_scheduler.add_noise(inputs, noise, timesteps)
    noisy_inputs = noisy_inputs.to(self.device)

    pred = self.model(noisy_inputs, timesteps, class_labels=labels).sample
    loss = self.criterion(pred, noise)

    self.optimizer.zero_grad()
    self.accelerator.backward(loss)
    self.lr_scheduler.step()
    self.optimizer.step()
```

#### Eval

```python
for _, label in self.test_loader:
    xt = torch.randn(32, 3, 64, 64, device=self.device)
    labels = label.to(self.device, dtype=torch.float32).squeeze()

    for t in tqdm(range(self.timestep - 1, 0, -1)):
        xt = xt.to(self.device)
        outputs = self.model(xt, t, class_labels=labels).sample
        xt = self.noise_scheduler.step(outputs, t, xt).prev_sample

    acc = self.eval_model.eval(xt, labels)
    print(f'Accuracy: {acc}')
    img = self.rev_transforms(xt)
    if epoch >= 0:
        save_image(img, f'{self.args.test_root}/test_{epoch}.png')
    else:
        save_image(img, f'{self.args.test_root}/test.png')
```

### B. Hyperparameters

-   learning rate: 1e-4
-   epochs: 80
-   timestep: 1200
-   batch size: 96
-   optimizer: AdamW

## 3. Results and discussion

### A. Show your accuracy screenshot based on the testing data.

|                         test (0.875)                         |                       new_test (0.869)                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](/Users/shiheng/Documents/Github/Deep_Learning/Lab6/image/test_acc.png) | ![](/Users/shiheng/Documents/Github/Deep_Learning/Lab6/image/new_test_acc.png) |



### B. Show your synthetic image grids and a progressive generation image.

<img src="/Users/shiheng/Documents/Github/Deep_Learning/Lab6/image/test_prog.png" style="zoom:200%;" />

|                             test                             |                           new_test                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](/Users/shiheng/Documents/Github/Deep_Learning/Lab6/image/test.png) | ![](/Users/shiheng/Documents/Github/Deep_Learning/Lab6/image/new_test.png) |





