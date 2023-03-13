# This is a sample Python script.
import sys
from torchvision import transforms
import cv2
import torch
import torch.nn as nn
from model import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from ui_mainwindow import Ui_MainWindow
from DW_GAN_model import fusion_net

device = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor()])


def model1(fileName):
    source, target = dehaze(fileName)
    dst = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    dst2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    return dst, dst2


class MainWindow(Ui_MainWindow):
    def __init__(self, qmainwindow: QMainWindow) -> None:
        super(MainWindow, self).__init__()
        self.mainwindow = qmainwindow
        self.setupUi(qmainwindow)
        self.pushButton.clicked.connect(lambda: self.openfile())
        self.model2 = None

    def openfile(self):
        fileDialog = QFileDialog.getOpenFileName(self.mainwindow, "选择文件", "./", "Image files(*.jpg *jpeg *png)")
        fileName = fileDialog[0]
        if fileName is None or len(fileName) == 0:
            return

        index = self.comboBox.currentIndex()
        if index == 0:
            QMessageBox.warning(self.mainwindow, "警告信息", "当前算法没有实现")
            return
        elif index == 1:
            dst, dst2 = model1(fileName)
        elif index == 2:
            if self.model2 is None:
                self.model2 = fusion_net()
                self.model2 = self.model2.to(device)
                self.model2 = nn.DataParallel(self.model2)
                self.label_7.setText("加载神经网络模型中......")
                self.model2.load_state_dict(torch.load('./weights/dehaze.pkl', map_location='cpu'))
                self.label_7.setText("神经网络模型加载完成!")
            with torch.no_grad():
                self.label_7.setText("开始执行推理......(大约1分钟)")
                self.model2.eval()
                img = cv2.imread(fileName)
                init_img = img.copy()
                if img.shape[0] != 1200 or img.shape[1] != 1600:
                    img = cv2.resize(img, (1600, 1200))
                img = transform(img)
                img = img[None, :, :, :]
                hazy_up = img[:, :, 0:1152, :]
                hazy_down = img[:, :, 48:1200, :]

                frame_out_up = self.model2(hazy_up)
                frame_out_down = self.model2(hazy_down)
                frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)], 1)).permute(0, 3, 1, 2)

                im = torch.squeeze(frame_out).numpy().transpose(1, 2, 0)
                im *= 255
                im = im.clip(0, 255).astype(np.uint8)
                init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                dst, dst2 = init_img, im
        else:
            QMessageBox.warning(self.mainwindow, "警告信息", "当前算法没有实现")
            return

        qt_img1 = QImage(dst.data, dst.shape[1], dst.shape[0], dst.shape[1]*3, QImage.Format.Format_RGB888)
        qt_img2 = QImage(dst2.data, dst.shape[1], dst.shape[0], dst.shape[1]*3, QImage.Format.Format_RGB888)

        pix_map1 = QPixmap.fromImage(qt_img1)
        pix_map2 = QPixmap.fromImage(qt_img2)

        pix_map1 = pix_map1.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        pix_map2 = pix_map2.scaled(self.label_2.size(), Qt.AspectRatioMode.KeepAspectRatio)

        self.label.setPixmap(pix_map1)
        self.label_2.setPixmap(pix_map2)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    qmainwindow = QMainWindow()
    ui = MainWindow(qmainwindow)
    qmainwindow.show()  # 显示窗口
    sys.exit(app.exec())

    # dehaze('images/img.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
