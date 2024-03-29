import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from tqdm import tqdm
from safetensors import safe_open
from accelerate import Accelerator
from argparse import ArgumentParser, Namespace
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataloader import iclevrLoader
from evaluator import evaluation_model


class DDPM:
    def __init__(
        self,
        args: Namespace,
    ) -> None:
        self.args = args
        self.epochs = args.epochs
        self.device = args.device
        self.timestep = args.timestep

        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            class_embed_type=None,
            # num_class_embeds = 2325, #C (24, 3) + 1
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.model.class_embedding = nn.Linear(24, 512)
        self.model = self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.noise_scheduler = DDPMScheduler(
            self.timestep, beta_schedule='squaredcos_cap_v2')

        self.accelerator = Accelerator()
        self.train_loader, self.test_loader = self.__get_dataloader()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )

        self.model, self.optimizer, self.train_loader, self.test_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.lr_scheduler
        )

        self.rev_transforms = transforms.Compose([
            transforms.Normalize((0, 0, 0), (2, 2, 2)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
        ])

        self.eval_model = evaluation_model()

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
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

                self.__tqdm_bar(
                    pbar=pbar,
                    epoch=epoch,
                    loss=loss.detach().cpu().item(),
                    lr=self.lr_scheduler.get_last_lr()[0]
                )

            acc = self.eval(epoch)
            self.model.save_pretrained(
                '{}/epoch-{:2d}-{:2d}'.format(self.args.checkpoints,
                                              epoch, round(100 * acc)),
                variant='non_ema'
            )

    def eval(self, epoch, progressive=False) -> float:
        self.model.eval()
        with torch.no_grad():
            for idx, (_, label) in enumerate(self.test_loader):
                xt = torch.randn(
                    self.args.test_batch_size, 3, 64, 64, device=self.device)
                labels = label.to(self.device, dtype=torch.float32).squeeze()
                
                prog_list = []
                for t in tqdm(range(self.timestep - 1, 0, -1)):
                    xt = xt.to(self.device)
                    outputs = self.model(xt, t, class_labels=labels).sample
                    xt = self.noise_scheduler.step(outputs, t, xt).prev_sample

                    if progressive and t % 200 == 0:
                        imgt = self.rev_transforms(xt)
                        prog_list.append(imgt)

                acc = self.eval_model.eval(xt, labels)
                print(f'Accuracy: {acc}')
                img = self.rev_transforms(xt)
                prog_list.append(img)

                if progressive:
                    imgt = make_grid(prog_list, nrow=6)
                    save_image(imgt, f'{self.args.test_root}/{idx}_test_prog.png')

                if epoch >= 0:
                    save_image(img, f'{self.args.test_root}/test_{epoch}.png')
                else:
                    save_image(img, f'{self.args.test_root}/{idx}_test.png')
        return acc

    def load_pretrained(self) -> None:
        self.model = UNet2DModel.from_pretrained(
            pretrained_model_name_or_path=self.args.ckpt_path,
            variant="non_ema",
            from_tf=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )

        with safe_open(
            f'{self.args.ckpt_path}/diffusion_pytorch_model.non_ema.safetensors',
            framework='pt',
            device=self.device
        ) as pt_file:
            pt = {key: pt_file.get_tensor(key) for key in pt_file.keys()}
            filtered_state_dict = {k[16:]: v for k, v in pt.items(
            ) if k == "class_embedding.weight" or k == "class_embedding.bias"}

        class_embedding = nn.Linear(24, 512)
        class_embedding.load_state_dict(filtered_state_dict)
        self.model.class_embedding = class_embedding
        self.model = self.model.to(self.device)

    def __get_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            iclevrLoader(mode='test'),
            batch_size=self.args.batch_size,
            num_workers=4,
            shuffle=True
        )
        test_loader = DataLoader(
            iclevrLoader(mode=self.args.test_dataset),
            batch_size=self.args.test_batch_size
        )

        return train_loader, test_loader

    def __tqdm_bar(self, pbar, epoch, loss, lr) -> None:
        pbar.set_description(f"Epoch {epoch}, lr:{round(lr, 4)}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-bs', '--batch_size', default=96)
    parser.add_argument('-e', '--epochs', type=int, default=80)
    parser.add_argument('-t', '--timestep', type=int, default=1200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--test_dataset', type=str, default='test')
    parser.add_argument('--test_root', type=str, default='./test_result')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    model = DDPM(args)

    if args.test_only:
        model.load_pretrained()
        model.eval(-1, progressive=True)
    else:
        model.train()


if __name__ == '__main__':
    main()
