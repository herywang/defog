from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DW_GAN_model import fusion_net
from torchvision.utils import save_image as imwrite
import time
import re
import cv2


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cpu")

# --- Define the network --- #
net = fusion_net()

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net)
net.load_state_dict(torch.load('./weights/dehaze.pkl', map_location='cpu'))
transform = transforms.Compose([transforms.ToTensor()])
file_path = "/Users/wangheng/workspace/pycharmworkspace/defog/NH-HAZE/50_hazy.png"
# --- Test --- #
with torch.no_grad():
    net.eval()
    start_time = time.time()
    img = cv2.imread(file_path)
    init_size = img.shape
    print(img.shape)
    img = cv2.resize(img, (1600, 1200))
    print(img.shape)
    img = transform(img)
    img = img[None, :, :, :]
    print(img.shape)
    hazy_up = img[:, :, 0:1152, :]
    hazy_down = img[:, :, 48:1200, :]
    print("hazy up shape", hazy_up.shape, "--", hazy_down.shape)

    frame_out_up = net(hazy_up)
    frame_out_down = net(hazy_down)
    frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)], 1)).permute(0, 3, 1, 2)

    print(frame_out.shape)
    im = torch.squeeze(frame_out).numpy().transpose(1, 2, 0)
    print(im.shape)
    cv2.imshow("name", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_time = time.time() - start_time
print(test_time)
