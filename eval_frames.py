import json
import cv2
import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser("Sequences evaluation")
parser.add_argument("--video", "-v", action='store_true', default=False)
parser.add_argument("--metrics", "-m", action='store_true', default=False)
parser.add_argument("--start", type=int, default = 0)
parser.add_argument("--end", type=int, default = 299)
parser.add_argument("--itr_I", type=int, default = 15000)
parser.add_argument("--itr_P", type=int, default = 300)
parser.add_argument("--lmbda", type=float, default = 0.004)
parser.add_argument("--P_lmbda", type=float, default = 1.0)
parser.add_argument("--path", type=str, default = "/amax/tangly/LD/iFVC/outputs_N3DV/coffee_martini")

args = parser.parse_args()

def find_lines_with_string(file_path, target_string):
    lines_with_string = []
    with open(file_path, 'r') as file:
        for line in file:
            if target_string in line:
                lines_with_string.append(line)
    return lines_with_string

def get_enc_time(path):
    target_string = "EncTime"
    target_line = find_lines_with_string(path, target_string)[-1]
    enc_time = float(target_line.split(" ")[-1])
    
    return enc_time

def get_dec_time(path):
    target_string = "DecTime"
    target_line = find_lines_with_string(path, target_string)[-1]
    dec_time = float(target_line.split(" ")[-1])
    
    return dec_time

def get_size(path):
    target_string = "Encoded sizes in MB"
    target_line = find_lines_with_string(path, target_string)[-1]
    size = float(target_line.split(",")[-2].split(" ")[-1])

    return size

def get_ntc_size(path):
    target_string = "EncTime"
    target_line = find_lines_with_string(path, target_string)[-1]
    split_line = target_line.split(",")

    ntc_2D = float(split_line[1].split(":")[-1].split(" ")[-1])
    ntc_3D = float(split_line[2].split(" ")[-1])
    ntc_mlp = float(split_line[3].split(" ")[-1])
    size_wo_3D = float(target_line.split(",")[-2].split(" ")[-1])
    total_size = float(target_line.split(",")[-3].split(" ")[-1])

    return [total_size, ntc_2D, ntc_3D, ntc_mlp, size_wo_3D]

def get_inference_time(path):
    target_string = "Query_ntc_time"
    target_line = find_lines_with_string(path, target_string)[-1]
    split_line = target_line.split(",")

    query_ntc_time = float(split_line[0].split(":")[-1])
    transform_time = float(split_line[1].split(" ")[-1])

    return [query_ntc_time, transform_time]

def get_training_time(path):
    target_string = "Total Training time"
    target_line = find_lines_with_string(path, target_string)[-1]
    training_time = float(target_line.split(" ")[-1])

    return training_time

def get_testing_fps(path):
    target_string = "Test FPS:"
    target_line = find_lines_with_string(path, target_string)[-1]
    testing_fps = float(target_line.split("m")[1].split("\x1b[0")[0])

    return testing_fps

path = args.path

start = args.start
end = args.end

itr_I = args.itr_I
itr_P = args.itr_P

lmbda = args.lmbda
P_lmbda = args.P_lmbda

path = os.path.join(path, f"{lmbda}_{P_lmbda}")
view = 0

imgs = []
height, width, layers = None, None, None
psnr, ssim, lpips = [], [], []
enc_time, dec_time, size = [], [], []
training_time, testing_fps = [], []

ntc_2D_size = 0.0
ntc_3D_size = 0.0
ntc_mlp_size = 0.0

query_ntc_total_time = 0.0
transform_total_time = 0.0
for id in range(start, end+1):
    frame_name = "frame{:06d}".format(id)
    if id == 0:
        itr = itr_I
    else:
        itr = itr_P
    iter_name = f"ours_{itr}"

    if args.video:
        image_path = os.path.join(path, frame_name, "test", iter_name, "renders", "{:05d}.png".format(view))
        img = cv2.imread(image_path)
        if height is None:
            height, width, layers = img.shape
        imgs.append(img)

    if args.metrics:
        json_path = os.path.join(path, frame_name, "results.json")
        with open(json_path, 'r') as file:
            data = json.load(file)
            res = data[iter_name]
            psnr.append(float(res["PSNR"]))
            ssim.append(float(res["SSIM"]))
            lpips.append(float(res["LPIPS"]))
        if id == 0:
            log_path = os.path.join(path, frame_name, "outputs.log")
            enc_time.append(get_enc_time(log_path))
            dec_time.append(get_dec_time(log_path))
            size.append(get_size(log_path))
            training_time.append(get_training_time(log_path))
            testing_fps.append(get_testing_fps(log_path))
        else:
            # log_path1 = os.path.join(path, frame_name+"_offsets", "outputs.log")
            log_path2 = os.path.join(path, frame_name, "outputs.log")
            enc_time.append(get_enc_time(log_path2))
            dec_time.append(get_dec_time(log_path2))
            [total_size, ntc_2D, ntc_3D, ntc_mlp, size_wo_3D] = get_ntc_size(log_path2)
            # [query_ntc_time, transform_time] = get_inference_time(log_path2)

            ntc_2D_size += ntc_2D
            ntc_3D_size += ntc_3D
            ntc_mlp_size += ntc_mlp

            # query_ntc_total_time += query_ntc_time
            # transform_total_time += transform_time

            size.append(total_size)
            training_time.append(get_training_time(log_path2))
            testing_fps.append(get_testing_fps(log_path2))

if args.metrics:
    print("psnr:", np.mean(psnr))
    print("ssim:", np.mean(ssim))
    print("lpips:", np.mean(lpips))
    print("enc_time:", np.mean(enc_time[1:]))
    print("dec_time:", np.mean(dec_time[1:]))
    print("training_time:", np.mean(training_time[5:]))
    print("testing_fps:", np.mean(testing_fps[5:]))
    print("size:", np.mean(size))
    print("sum size:", np.sum(size))

    file_path = os.path.join(path, f"metrics_{lmbda}_{P_lmbda}.txt")
    file = open(file_path, 'w')
    file.write(f"Frame PSNR SSIM LPIPS ENC_TIME DEC_TIME SIZE Train FPS\n")
    file.write("Mean  {:.2f} {:.3f} {:.3f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f}\n".format(np.mean(psnr), np.mean(ssim), np.mean(lpips), np.mean(enc_time), np.mean(dec_time), np.mean(size), np.mean(training_time[5:]), np.mean(testing_fps[5:])))
    for i in range(start, end+1):
        line = "{:<5} {:.2f} {:.3f} {:.3f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f}\n".format(i, psnr[i-start], ssim[i-start], lpips[i-start], enc_time[i-start], dec_time[i-start], size[i-start], training_time[i-start], testing_fps[i-start])
        file.write(line)
    file.close()

if args.video:
    video_name = os.path.join(path, f"frames_{view}_{start}_{end}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 20, (width, height))
    # 将图片合成为视频
    for img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    