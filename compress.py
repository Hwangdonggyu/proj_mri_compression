import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import struct  # 바이트 데이터를 변환하기 위한 모듈
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"]="1"

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.add_argument("--comp", action="store_true", help="Compression mode")
    parser.add_argument("--decomp", action="store_true", help="Decompression mode")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)
    net.eval()
    count = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)

    if args.comp:
        compressed_folder = "compressed_data"
        os.makedirs(compressed_folder, exist_ok=True)
        net.update()
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)     
            try:
                with torch.no_grad():
                    out_enc = net.compress(x_padded)
                    
                    # 압축된 데이터 저장
                    compressed_file = os.path.join(compressed_folder, f"{img_name}.bin")
                    with open(compressed_file, "wb") as f:
                        # 압축된 shape 정보 저장 (out_enc["shape"] 사용)
                        f.write(struct.pack("II", *out_enc["shape"]))  # (batch, C, H, W)
                        #f.write(struct.pack("II", img.shape[1], img.shape[2]))  # 여기 고치기 이미지의 h ,w 값
                        f.write(struct.pack("IIII", x_padded.shape[2], x_padded.shape[3], img.shape[1], img.shape[2]))
                        f.write(struct.pack("I", len(out_enc["strings"])))  # 스트링 개수 저장
                        for s in out_enc["strings"]:
                            f.write(struct.pack("I", len(s[0])))  # 현재 스트링의 길이 저장
                            f.write(s[0])  # 실제 데이터 저장
                    
                    print(f"Compressed data saved to: {compressed_file}")
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")
                error_folder = "error_images"
                os.makedirs(error_folder, exist_ok=True)
                error_img = Image.open(img_path)
                error_img.save(os.path.join(error_folder, f"{img_name}.webp"), format="WEBP")
                print(f"Saved error image as WebP: {img_name}.webp")

    
    if args.decomp:
        decoded_folder = "decoded_images"
        os.makedirs(decoded_folder, exist_ok=True)
        net.update()
        
        for bin_file in os.listdir(path):
            bin_path = os.path.join(path, bin_file)
            
            with open(bin_path, "rb") as f:
                # 저장된 shape 정보 읽기 (batch, C, H, W)
                H, W = struct.unpack("II", f.read(8))
                H_pad, W_pad, img_H, img_W = struct.unpack("IIII", f.read(16))
                # img_H,img_W = struct.unpack("II", f.read(8))
                # 저장된 strings 개수 읽기
                num_strings = struct.unpack("I", f.read(4))[0]
                
                # strings 데이터 읽기
                strings = []
                for _ in range(num_strings):
                    length = struct.unpack("I", f.read(4))[0]  # 스트링 길이
                    strings.append([f.read(length)])  # 해당 길이만큼 읽기
                
            # ⚠️ `shape`을 올바르게 전달 (압축된 데이터의 shape을 그대로 사용)
            with torch.no_grad():
                out_dec = net.decompress(strings, (H, W))
                # out_dec["x_hat"] = out_dec["x_hat"][..., :img_H, :img_W]  # 원본 크기대로 crop
                # ⚠️ 복원된 이미지 크기가 원본 크기보다 작은 경우 예외 처리
                restored_H, restored_W = out_dec["x_hat"].shape[2], out_dec["x_hat"].shape[3]
                img_H = min(img_H, restored_H)
                img_W = min(img_W, restored_W)

                # 패딩을 고려한 crop 적용
                padding_top = (H_pad - img_H) // 2
                padding_left = (W_pad - img_W) // 2
                out_dec["x_hat"] = out_dec["x_hat"][..., padding_top:padding_top + img_H, padding_left:padding_left + img_W]

                decoded_img = out_dec["x_hat"].squeeze(0).clamp(0, 1).cpu()
                decoded_file = os.path.join(decoded_folder, f"decoded_{bin_file.replace('.bin', '.png')}")
                transforms.ToPILImage()(decoded_img).save(decoded_file)
                print(f"Decoded image saved to: {decoded_file}")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
