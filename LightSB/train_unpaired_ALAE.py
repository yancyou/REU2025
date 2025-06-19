import os, sys
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_root, "ALAE"))
sys.path.insert(0, os.path.join(repo_root, "src"))
sys.path.insert(0, repo_root)
import argparse
import torch
_old_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _old_load(*args, **kwargs)
torch.load = patched_load
import torch.nn.functional as F
F._verify_spatial_size = lambda size: None
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import Dataset
import glob
from PIL import Image
import dlutils            
from light_sb import LightSB    
import wandb                               
from alae_ffhq_inference import load_model, encode, decode

class UnpairedImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=("jpg", "jpeg", "png")):
        self.transform = transform
        exts = tuple(ext.lower() for ext in extensions)
        # Collect all image files in `root` (non-recursive)
        self.paths = [
            p
            for p in glob.glob(os.path.join(root, "*"))
            if p.split(".")[-1].lower() in exts
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return a dummy label of 0 (unused)
        return img, 0

def get_dataloader(root, phase, img_size, batch_size):
    """
    Create a DataLoader for unpaired images in `root/phase`.
    Resizes & crops to `img_size`, normalizes to [-1,1].
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    folder = os.path.join(root, phase)
    dataset = UnpairedImageFolder(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/Horse2Zebra",
        help="Path to Horse2Zebra dataset root",
    )
    parser.add_argument(
        "--img_size", type=int, default=1024, help=" Resize / crop size for images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=512, help="ALAE latent space dimension"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of discretization steps for SB sampler",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=8,
        help="Number of Gaussian mixture components",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Diffusion strength ε"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (cuda or cpu)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    device = torch.device(args.device)

    # dataset
    loaderA = get_dataloader(args.root, 'trainA', args.img_size, args.batch_size)
    loaderB = get_dataloader(args.root, 'trainB', args.img_size, args.batch_size)

    os.chdir(os.path.join(repo_root, "ALAE"))

    cfg_path = os.path.join(repo_root, "ALAE", "configs", "ffhq.yaml")
    ckpt_dir = os.path.join(repo_root, "ALAE", "training_artifacts", "ffhq")
    last_file = os.path.join(ckpt_dir, "last_checkpoint")
    if not os.path.isfile(last_file):
        with open(last_file, "w") as f:
            f.write("model_submitted.pth")

    alae = load_model(cfg_path, ckpt_dir) 

    alae = alae.to(device)
    for block in alae.decoder.decode_block:
        block.noise_weight_1 = block.noise_weight_1.to(device)
        block.noise_weight_2 = block.noise_weight_2.to(device)
      
    alae.eval()
    for param in alae.parameters():
        param.requires_grad = False

    solver_AB = LightSB(
        dim=args.latent_dim,
        n_potentials=args.n_comp,
        epsilon=args.epsilon,
        is_diagonal=True,
        sampling_batch_size=args.batch_size
    ).to(device)
    solver_BA = LightSB(
        dim=args.latent_dim,
        n_potentials=args.n_comp,
        epsilon=args.epsilon,
        is_diagonal=True,
        sampling_batch_size=args.batch_size
    ).to(device)
    
    optim = torch.optim.Adam(
        list(solver_AB.parameters()) + list(solver_BA.parameters()),
        lr=args.lr,
    )
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    
    # train
    for epoch in range(0, args.epochs+1):
        totalAB = totalBA = count = 0
        for (xA, _), (xB, _) in zip(loaderA, loaderB):
            xA, xB = xA.to(device), xB.to(device)
            # Encode images to latent space (no grad)
            with torch.no_grad():
              z_A = encode(alae, xA) # [B, L, 512]
              z_B = encode(alae, xB) # [B, L, 512]

            zA = z_A.mean(dim=1) # [B, 512]
            zB = z_B.mean(dim=1) # [B, 512]

            # Compute A→B loss:  E[ -log p*(zB) + log C(zA) ]
            logp_B = solver_AB.get_log_potential(zB) # [B]
            logC_A = solver_AB.get_log_C(zA) # [B]
            lossAB = (-logp_B + logC_A).mean()
            
            # Compute B→A loss similarly
            logp_A = solver_BA.get_log_potential(zA)
            logC_B = solver_BA.get_log_C(zB)
            lossBA = (-logp_A + logC_B).mean()      
    
            loss = lossAB + lossBA

            optim.zero_grad()
            loss.backward()
            optim.step()

            totalAB += lossAB.item()
            totalBA += lossBA.item()
            count += 1

        print(f"Epoch {epoch}: lossAB={totalAB/count:.4f}, lossBA={totalBA/count:.4f}")

        if epoch % 10 == 0:
            solver_AB.eval()
            solver_BA.eval()
            
            xA_vis, _ = next(iter(loaderA))
            ZA_vis = encode(alae, xA_vis.to(device))
            zA_vis = ZA_vis.mean(dim=1) 
            trajB = solver_AB.sample_euler_maruyama(zA_vis, args.n_steps)
            zB_pred = trajB[:, -1, :].to(device)
            xAB = decode(alae, zB_pred).to(device)
            save_image(xAB, f"results/AB_epoch{epoch}.png", nrow=2, normalize=True)

            xB_vis, _ = next(iter(loaderB))
            ZB_vis = encode(alae, xB_vis.to(device))
            zB_vis = ZB_vis.mean(dim=1)                
            trajA = solver_BA.sample_euler_maruyama(zB_vis, args.n_steps)
            zA_pred = trajA[:, -1, :].to(device)
            xBA = decode(alae, zA_pred).to(device)
            save_image(xBA, f"results/BA_epoch{epoch}.png", nrow=2, normalize=True)
            solver_AB.train()
            solver_BA.train()

    torch.save(solver_AB.state_dict(), "checkpoints/solver_AB.pth")
    torch.save(solver_BA.state_dict(), "checkpoints/solver_BA.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()