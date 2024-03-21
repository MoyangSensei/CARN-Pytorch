import os
from PIL import Image
import argparse
import numpy as np
import math
import cv2
import pandas as pd
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--target_path", type=str, required=True)
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--file_format", type=str, default=".png")

def calculate_useSKIMAGE(gt_path, target_path, gt_file_list ,target_file_list):
    print("------------------")
    print("calculate_useSKIMAGE...")
    result_p = []
    result_s = []
    with tqdm(total=len(gt_file_list)) as bar:
        for i in range(len(gt_file_list)):
            gt_file = gt_path + "/" + gt_file_list[i]
            target_file = target_path + "/" + target_file_list[i]
            IMG1 = cv2.imread(gt_file, 1)
            IMG2 = cv2.imread(target_file, 1)
            img1 = np.float64(IMG1)
            img2 = np.float64(IMG2)
            if len(img2) == 1023:
                IMG2 = cv2.copyMakeBorder(IMG2, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
                img2 = np.float64(IMG2)
            result_p.append(compare_psnr(img1,img2,data_range=255))
            result_s.append(compare_ssim(img1,img2,multichannel=True))
            bar.update(1)
    return result_p,result_s

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(gt_path, target_path, gt_file_list ,target_file_list):
    print("------------------")
    print("calculate_ssim...")
    result = []
    with tqdm(total=len(gt_file_list)) as bar:
        for i in range(len(gt_file_list)):
            gt_file = gt_path + "/" + gt_file_list[i]
            target_file = target_path + "/" + target_file_list[i]
            # img1 = cv2.imread(gt_file, 1)
            # img2 = cv2.imread(target_file, 1)
            # img1 = img1.astype(np.float64)
            # img2 = img2.astype(np.float64)
            IMG1 = cv2.imread(gt_file, 1)
            IMG2 = cv2.imread(target_file, 1)
            img1 = np.float64(IMG1)
            img2 = np.float64(IMG2)
            if len(img2) == 1023:
                IMG2 = cv2.copyMakeBorder(IMG2, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
                img2 = np.float64(IMG2)
            if not img1.shape == img2.shape:
                raise ValueError('Input images must have the same dimensions.')
            if img1.ndim == 2:
                result.append(ssim(img1, img2))
            elif img1.ndim == 3:
                if img1.shape[2] == 3:
                    ssims = []
                    for i in range(3):
                        ssims.append(ssim(img1, img2))
                    result.append(np.array(ssims).mean())
                elif img1.shape[2] == 1:
                    result.append(ssim(np.squeeze(img1), np.squeeze(img2)))
            else:
                raise ValueError('Wrong input image dimensions.')
            bar.update(1)
    return result

def calculate_psnr(gt_path, target_path, gt_file_list, target_file_list):
    print("------------------")
    print("calculate_psnr...")
    result=[]
    with tqdm(total=len(gt_file_list)) as bar:
        for i in range(len(gt_file_list)):
            gt_file=gt_path+"/"+gt_file_list[i]
            target_file=target_path+"/"+target_file_list[i]
            IMG1 = cv2.imread(gt_file, 1)
            IMG2 = cv2.imread(target_file, 1)
            img1 = np.float64(IMG1)
            img2 = np.float64(IMG2)
            if len(img2)==1023:
                IMG2 = cv2.copyMakeBorder(IMG2, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
                img2 = np.float64(IMG2)
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                result.append(100)
            PIXEL_MAX = 255.0
            result.append( 20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
            bar.update(1)
    return result

def calculate_lpips(gt_path, target_path, gt_file_list, target_file_list,NET):
    print("------------------")
    print("calculate_lpips... (",NET,")")
    import lpips
    use_gpu = True  # Whether to use GPU
    spatial = True  # Return a spatial map of perceptual distance.
    loss_fn = lpips.LPIPS(net=NET, spatial=spatial)  # Can also set net = 'squeeze' or 'vgg'
    if (use_gpu):
        loss_fn.cuda()
    result=[]
    with tqdm(total=len(gt_file_list)) as bar:
        for i in range(len(gt_file_list)):
            gt_file=gt_path+"/"+gt_file_list[i]
            target_file=target_path+"/"+target_file_list[i]
            dummy_img0 = lpips.im2tensor(lpips.load_image(gt_file))
            dummy_img1 = lpips.im2tensor(lpips.load_image(target_file))
            if (use_gpu):
                dummy_img0 = dummy_img0.cuda()
                dummy_img1 = dummy_img1.cuda()
            dist = loss_fn.forward(dummy_img0, dummy_img1)
            result.append(dist.mean().item())
            bar.update(1)
    return result

def get_csv(data,csv_path):
    print("------------------")
    print("get_csv: ",csv_path)
    data=map(list, zip(*data))
    name = ["num", "id", "psnr","ssim","psnr use skimage","ssim use skimage","lpips(alex)","lpips(vgg)"]
    test = pd.DataFrame(columns=name, data=data)  # 数据有三列，列名分别为one,two,three
    test.to_csv(csv_path, encoding="gbk",float_format='%.3f',index=0)

def check_file_name(gt_path,target_path,file_format):
    if not os.path.exists(gt_path) or not os.path.exists(target_path):
        print("check_file_name: file path not exist")
    else:
        gt_file_list = [file for file in os.listdir(gt_path) if file.endswith(file_format)]
        gt_file_list.sort()
        gt_file_list=gt_file_list[-200:]
        gt_file_list_id = [file[:4] for file in os.listdir(gt_path) if file.endswith(file_format)]
        gt_file_list_id.sort()
        gt_file_list_id=gt_file_list_id[-200:]
        # print(gt_file_list_id)

        trarget_file_list = [file for file in os.listdir(target_path) if file.endswith(file_format)]
        trarget_file_list.sort()
        trarget_file_list_id = [file[:4] for file in os.listdir(target_path) if file.endswith(file_format)]
        trarget_file_list_id.sort()
        # print(trarget_file_list_id)

        if len(gt_file_list_id)!=len(trarget_file_list_id) or list(set(gt_file_list_id) ^ set(trarget_file_list_id))!=[]:
            print("check_file_name: file can't one to one correspondence")
            return [False]
        else:
            print("check_file_name: ", len(gt_file_list))
            num = list(range(1, len(gt_file_list)+1))
            return[True,num,gt_file_list,trarget_file_list]

def calculate_all(gt_path,target_path,csv_path,file_format):
    print("----------------------------------------------------------------")
    print("gt_path:",gt_path)
    print("target_path:", target_path)
    check_file=check_file_name(gt_path,target_path,file_format)
    if check_file[0]:
        result=[]
        result.append(check_file[1])
        result.append(check_file[2])
        result.append(calculate_psnr(gt_path, target_path, check_file[2], check_file[3]))
        result.append(calculate_ssim(gt_path, target_path, check_file[2], check_file[3]))
        result_useSKIMAGE=calculate_useSKIMAGE(gt_path, target_path, check_file[2], check_file[3])
        result.append(result_useSKIMAGE[0])
        result.append(result_useSKIMAGE[1])
        result.append(calculate_lpips(gt_path, target_path, check_file[2], check_file[3],'alex'))
        result.append(calculate_lpips(gt_path, target_path, check_file[2], check_file[3], 'vgg'))
        get_csv(result,csv_path)

        print("----------------------------------------------------------------")
    else:
        exit(0)

if __name__=='__main__':
    args = parser.parse_args()
    gt_path = args.gt_path
    target_path = args.target_path
    csv_path = args.csv_path
    file_format = args.file_format

    calculate_all(gt_path,target_path,csv_path,file_format)

