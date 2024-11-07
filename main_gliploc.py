import torch 
import torchvision.transforms as transforms
import clip

from datasets.vigor_plus import VigorPlus

from utils import *
from run_utils import *
from glip_loc_lora import run_lora

def main():
    args = get_arguments()

    set_random_seed(args.seed)

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    print("Preparing dataset.")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    ])
    root_path = '/home/erzurumlu.1/yunus/my_link/yunus/VIGOR'
    train_dataset = VigorPlus(root_path, ['NewYork', 'Seattle'], transform=train_transform)
    val_dataset = VigorPlus(root_path, ['Chicago', 'SanFrancisco'], transform=val_transform, val=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}") 
    run_lora(args, clip_model, logit_scale, train_loader, val_loader)

if __name__ == '__main__':
    main()