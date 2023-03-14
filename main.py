# This is a sample Python script.
import sys
from torchvision import transforms
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
from model import *
from models import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel
from PyQt5.QtCore import Qt
from ui_mainwindow import Ui_MainWindow
from threading import Thread

gps=3
blocks=19
device = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor()])
net = None

def setToLabel(dst1, dst2, showLabel1:QLabel, showLabel2:QLabel):
    qt_img2 = QImage(dst2.data.tobytes(), dst2.shape[1], dst2.shape[0], QImage.Format.Format_RGB888)
    pix_map2 = QPixmap.fromImage(qt_img2)
    pix_map2 = pix_map2.scaled(showLabel2.size(), Qt.AspectRatioMode.KeepAspectRatio)
    showLabel2.setPixmap(pix_map2)
    showLabel2.setAlignment(Qt.AlignmentFlag.AlignCenter)


def model1(fileName, showLabel1:QLabel, showLabel2:QLabel):
    showLabel1.clear()
    showLabel2.clear()
    source, target = dehaze(fileName)
    dst = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    dst2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    setToLabel(dst, dst2, showLabel1, showLabel2)
    

def model2(image_type:str, filePath,showLabel1:QLabel, showLabel2:QLabel, combox):
    showLabel1.clear()
    showLabel2.clear()
    showLabel2.setAlignment(Qt.AlignmentFlag.AlignCenter)
    showLabel2.setText("处理中,请稍等......")
    model_dir = './trained_models/'+image_type+'_train_ffa_3_19.pk'
    global net
    if net is None:
        ckp=torch.load(model_dir,map_location=device)
        net=FFA(gps=gps,blocks=blocks)
        net=nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
    net.eval()

    haze = cv2.imread(filePath)
    # haze = Image.open(filePath)
    haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
    qt_img1 = QImage(haze.data, haze.shape[1], haze.shape[0], haze.shape[1]*3, QImage.Format.Format_RGB888)
    pix_map1 = QPixmap.fromImage(qt_img1)
    pix_map1 = pix_map1.scaled(showLabel1.size(), Qt.AspectRatioMode.KeepAspectRatio)
    showLabel1.setPixmap(pix_map1)
    showLabel1.setAlignment(Qt.AlignmentFlag.AlignCenter)

    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    # haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
        print("开始推理...")
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1))
    ts *= 255
    ts = torch.permute(ts,(1,2,0))
    ts = ts.numpy()
    ts = ts.astype(np.uint8)
    combox.setEnabled(True)
    print("推理结束..")
    setToLabel(haze,ts, showLabel1, showLabel2)


class MainWindow(Ui_MainWindow):
    def __init__(self, qmainwindow: QMainWindow) -> None:
        super(MainWindow, self).__init__()
        self.mainwindow = qmainwindow
        self.setupUi(qmainwindow)
        self.pushButton.clicked.connect(lambda: self.openfile())
        self.model2 = None
        self.filePath = None

    def openfile(self):
        fileDialog = QFileDialog.getOpenFileName(self.mainwindow, "选择文件", "./", "Image files(*.jpg *jpeg *png)")
        fileName = fileDialog[0]
        if fileName is None or len(fileName) == 0:
            return
        self.filePath = fileName

        index = self.comboBox.currentIndex()
        self.comboBox.setEnabled(False)
        if index == 0:
            thread1 = Thread(target=model1, args=(fileName, self.label, self.label_2))
            thread1.setDaemon(True)
            thread1.start()
        elif index == 1:
            # 室内去雾识别
            thread2 = Thread(target=model2, args=('its', fileName, self.label, self.label_2, self.comboBox))
            thread2.setDaemon(True)
            thread2.start()
        else:
            # 室外去雾识别
            thread3 = Thread(target=model2, args=('ots', fileName, self.label, self.label_2, self.comboBox))
            thread3.setDaemon(True)
            thread3.start()

        


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("计算设备:", device)
    app = QApplication(sys.argv)
    qmainwindow = QMainWindow()
    ui = MainWindow(qmainwindow)
    qmainwindow.show()  # 显示窗口
    sys.exit(app.exec())

    # dehaze('images/img.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
