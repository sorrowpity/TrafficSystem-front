import os
import csv
import json
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QTextBrowser,
    QSlider, QDoubleSpinBox, QComboBox, QGridLayout, QMessageBox, QGroupBox, QRadioButton, QToolButton
)
from PySide6.QtCore import Qt, QTimer, QDir,QPoint
from PySide6.QtGui import QPixmap, QImage, QIcon, QPalette,QPen,QPainter,QBrush,QFont,QColor,QPaintEvent
import cv2
from ultralytics import YOLO
import torch

# 动态检查 TensorFlow 是否可用
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_type = None  # 当前选择的模型类型
        self.timer = QTimer()
        self.timer1 = QTimer()
        self.cap = None
        self.file_path = None
        self.base_name = None
        self.value = 0.5  # 默认置信度阈值
        self.is_streaming = False  # 是否正在处理实时视频流
        self.is_sidebar_expanded = False  # 初始化侧边栏展开状态
        self.detection_results_file = "detection_results.csv"  # 检测结果保存路径

        self.init_gui()

    def init_gui(self):
        self.setFixedSize(1300, 700)
        self.setWindowTitle('目标检测')
        self.setWindowIcon(QIcon("logo.jpg"))
        self.set_background_image('bg.jpg')  # 设置窗口背景图片

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        topLayout = QHBoxLayout()

        # 左侧原视频展示区
        self.oriVideoLabel = QLabel(self)
        self.oriVideoLabel.setFixedSize(530, 400)
        self.oriVideoLabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top: 75px;')
        self.oriVideoLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.draw_label(self.oriVideoLabel, "原视频")

        # 右侧检测结果展示区
        self.detectlabel = QLabel(self)
        self.detectlabel.setFixedSize(530, 400)
        self.detectlabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top: 75px;')
        self.detectlabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.draw_label(self.detectlabel, "检测结果")


        # 将原视频和检测结果展示区添加到布局
        topLayout.addWidget(self.oriVideoLabel)
        topLayout.addWidget(self.detectlabel)
        main_layout.addLayout(topLayout)


        # 创建日志打印区域
        self.outputField = QTextBrowser()
        self.outputField.setFixedSize(1050, 180)
        main_layout.addWidget(self.outputField)


        # 创建底部操作区域
        bottomLayout = QHBoxLayout()

        # 文件上传按钮
        self.openImageBtn = QPushButton('🖼️文件上传')
        self.openImageBtn.setFixedSize(100, 50)
        self.openImageBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.openImageBtn.clicked.connect(self.upload_file)
        bottomLayout.addWidget(self.openImageBtn)

        # 撤销文件上传按钮
        self.clearImageBtn = QPushButton('🗑️撤销上传')
        self.clearImageBtn.setFixedSize(100, 50)
        self.clearImageBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.clearImageBtn.clicked.connect(self.clear_image)
        self.clearImageBtn.setEnabled(False)
        bottomLayout.addWidget(self.clearImageBtn)

        # 侧栏展开/收起按钮
        self.sidebarBtn = QToolButton(self)
        self.sidebarBtn.setText('📂')
        self.sidebarBtn.setFixedSize(50, 50)
        self.sidebarBtn.setStyleSheet("""
            QToolButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QToolButton:hover {
                background-color: #f0f0f0;
            }
            QToolButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        self.sidebarBtn.clicked.connect(self.toggle_sidebar)
        bottomLayout.addWidget(self.sidebarBtn)

        # 模型选择侧栏
        self.model_selection_group = QGroupBox("选择模型类型")
        self.model_selection_layout = QVBoxLayout()

        self.yolo_radio = QRadioButton("YOLO")
        self.yolo_radio.setChecked(True)  # 默认选择 YOLO
        self.model_selection_layout.addWidget(self.yolo_radio)

        self.tensorflow_radio = QRadioButton("TensorFlow")
        self.model_selection_layout.addWidget(self.tensorflow_radio)

        self.pytorch_radio = QRadioButton("PyTorch")
        self.model_selection_layout.addWidget(self.pytorch_radio)

        self.model_selection_group.setLayout(self.model_selection_layout)
        self.model_selection_group.setVisible(False)  # 默认收起
        bottomLayout.addWidget(self.model_selection_group)



        # 导入模型按钮
        self.importModelBtn = QPushButton('📂导入模型')
        self.importModelBtn.setFixedSize(100, 50)
        self.importModelBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.importModelBtn.clicked.connect(self.import_model)
        bottomLayout.addWidget(self.importModelBtn)

        # 加载模型按钮
        self.loadModel = QPushButton('🔄️加载模型')
        self.loadModel.setFixedSize(100, 50)
        self.loadModel.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.loadModel.clicked.connect(self.load_model)
        bottomLayout.addWidget(self.loadModel)

        # 保存检测结果按钮
        self.saveResultBtn = QPushButton('💾导出结果')
        self.saveResultBtn.setFixedSize(100, 50)
        self.saveResultBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.saveResultBtn.clicked.connect(self.save_result)
        self.saveResultBtn.setEnabled(False)  # 默认禁用，只有在检测完成后才启用
        bottomLayout.addWidget(self.saveResultBtn)

        # 置信度阈值滑动条部分
        self.con_label = QLabel('置信度阈值', self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(99)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.updateSpinBox)
        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.spinbox.setMinimum(0.01)
        self.spinbox.setMaximum(0.99)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setValue(0.5)
        self.spinbox.valueChanged.connect(self.updateSlider)
        self.confudence_slider = QWidget()
        confidence_layout = QVBoxLayout()
        hlayout = QHBoxLayout()
        self.confudence_slider.setFixedSize(250, 64)
        confidence_layout.addWidget(self.con_label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spinbox)
        confidence_layout.addLayout(hlayout)
        self.confudence_slider.setLayout(confidence_layout)
        self.confudence_slider.setEnabled(False)
        bottomLayout.addWidget(self.confudence_slider)

        # 开始检测按钮
        self.start_detect = QPushButton('🔍开始检测')
        self.start_detect.setFixedSize(100, 50)
        self.start_detect.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.start_detect.clicked.connect(self.show_detect)
        self.start_detect.setEnabled(False)
        bottomLayout.addWidget(self.start_detect)

        # 停止检测按钮
        self.stopDetectBtn = QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        self.stopDetectBtn.setEnabled(False)
        bottomLayout.addWidget(self.stopDetectBtn)

        # 实时视频流按钮
        self.startStreamBtn = QPushButton('🎥实时视频流')
        self.startStreamBtn.setFixedSize(100, 50)
        self.startStreamBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
            }
        """)
        self.startStreamBtn.clicked.connect(self.start_stream)
        self.startStreamBtn.setEnabled(True)
        bottomLayout.addWidget(self.startStreamBtn)

        # 返回登录按钮
        self.returnLoginBtn = QPushButton('🔙返回登录')
        self.returnLoginBtn.setFixedSize(100, 50)
        self.returnLoginBtn.setStyleSheet("""
        QPushButton {
            background-color: white;
            color: black;
            border: 2px solid gray;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        QPushButton:pressed {
            background-color: #e0e0e0;
        }
        QPushButton:disabled {
            background-color: #d3d3d3;
        }
        """)
        self.returnLoginBtn.clicked.connect(self.close_user)  # 绑定点击事件
        bottomLayout.addWidget(self.returnLoginBtn)  # 将按钮添加到底部布局

        main_layout.addLayout(bottomLayout)

    def set_background_image(self, image_path):
        """设置窗口背景图片"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)  # 修正缩放方式
        palette = QPalette()
        palette.setBrush(QPalette.Window, scaled_pixmap)
        self.setPalette(palette)

    def save_detection_result(self, vehicle_type, confidence, bbox):
        """保存检测结果到文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "timestamp": timestamp,
            "vehicle_type": vehicle_type,
            "confidence": confidence,
            "bbox": bbox
        }

        # 保存为 CSV 文件
        with open(self.detection_results_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, vehicle_type, confidence, bbox])

        # 保存为 JSON 文件（可选）
        with open("detection_results.json", mode="a", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")

        self.outputField.append(f'{timestamp} - 检测结果已保存: {vehicle_type} ({confidence:.2f})')



    def toggle_sidebar(self):
        """切换侧栏的展开/收起状态"""
        print("toggle_sidebar 被调用")  # 调试信息
        self.is_sidebar_expanded = not self.is_sidebar_expanded
        print(f"侧栏展开状态: {self.is_sidebar_expanded}")  # 调试信息
        self.model_selection_group.setVisible(self.is_sidebar_expanded)
        print(f"侧栏可见性: {self.model_selection_group.isVisible()}")  # 调试信息

        # 强制更新布局
        self.adjustSize()
        self.update()

    def import_model(self):
        """导入模型"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择模型文件", filter="*.pt *.h5 *.pth")
        if file_path:
            try:
                if self.yolo_radio.isChecked():
                    if not file_path.endswith('.pt'):
                        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 文件格式不匹配，请选择 YOLO 模型文件 (.pt)')
                        return
                    self.model = YOLO(file_path)
                    self.model_type = "YOLO"
                elif self.tensorflow_radio.isChecked():
                    if not file_path.endswith('.h5'):
                        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 文件格式不匹配，请选择 TensorFlow 模型文件 (.h5)')
                        return
                    if not TENSORFLOW_AVAILABLE:
                        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - TensorFlow 未安装，无法加载模型')
                        return
                    self.model = tf.saved_model.load(file_path)
                    self.model_type = "TensorFlow"
                elif self.pytorch_radio.isChecked():
                    if not file_path.endswith('.pth'):
                        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 文件格式不匹配，请选择 PyTorch 模型文件 (.pth)')
                        return
                    self.model = torch.load(file_path)
                    self.model_type = "PyTorch"
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载成功: {file_path}')
                self.start_detect.setEnabled(True)
                self.stopDetectBtn.setEnabled(True)
                self.openImageBtn.setEnabled(True)
                self.confudence_slider.setEnabled(True)
            except Exception as e:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载失败: {str(e)}')

    def close_user(self):
        self.close()  # 接受关闭事件

    def closeEvent(self, event):
        """重写 closeEvent，在关闭窗口时返回登录界面"""
        event.accept()


    def save_result(self):
        if self.file_path and self.model:
            # 获取当前检测结果的图像
            try:
                if self.file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    frame = cv2.imread(self.file_path)
                elif self.file_path.endswith('.mp4') or self.is_streaming:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 无法读取视频帧！')
                        return
                else:
                    self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 无法识别的文件类型！')
                    return

                # 进行目标检测
                results = self.model(frame, conf=self.value)
                rendered_frame = results.render()[0]

                # 保存检测结果的图像
                output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测结果已保存到: {output_path}')

                # 保存识别信息到文本文件
                output_txt = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(output_txt, 'w') as f:
                    for det in results.xyxy[0].cpu().numpy():
                        if det[4] >= self.value:  # 置信度阈值过滤
                            x1, y1, x2, y2, conf, cls = det
                            label = self.model.names[int(cls)]
                            f.write(f"{label}: {conf:.2f}, BBox: ({x1}, {y1}), ({x2}, {y2})\n")
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 识别信息已保存到: {output_txt}')
            except Exception as e:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 保存检测结果失败: {str(e)}')
            else:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请先加载模型并进行检测！')

    def load_model_list(self):
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                base_name = os.path.splitext(filename)[0]
                self.selectModel.addItem(base_name)
        if self.selectModel.count() == 0:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 警告: 模型文件夹中没有找到任何模型文件。请手动加载模型文件。')
        else:
            self.loadModel.setEnabled(True)

    def load_model(self):
        filename = self.selectModel.currentText()
        full_path = os.path.join(self.folder_path, filename + '.pt')
        if os.path.exists(full_path):
            self.model = YOLO(full_path)
            self.start_detect.setEnabled(True)
            self.stopDetectBtn.setEnabled(True)
            self.openImageBtn.setEnabled(True)
            self.confudence_slider.setEnabled(True)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载成功: {filename}')
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择置信度阈值')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型文件 "{full_path}" 不存在！')

    def updateSpinBox(self, value):
        self.spinbox.setValue(value / 100.0)
        self.value = value / 100.0

    def updateSlider(self, value):
        self.slider.setValue(int(value * 100))
        self.value = value

    def upload_file(self):
        file_dialog = QFileDialog()
        file_dialog.setDirectory(QDir("./valid_file"))
        file_path, file_type = file_dialog.getOpenFileName(self, "选择检测文件", filter='*.jpg *.png *.jpeg *.mp4')
        if file_path:
            self.file_path = file_path
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 文件上传成功: {file_path}')
            if file_path.endswith('.jpg'):
                pixmap = QPixmap(file_path)
                # 使用 KeepAspectRatioByExpanding 保持宽高比并填充空白
                scaled_pixmap = pixmap.scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatioByExpanding)
                self.oriVideoLabel.setPixmap(scaled_pixmap)
            elif file_path.endswith('.mp4'):
                self.cap = cv2.VideoCapture(file_path)
                ret, frame = self.cap.read()
                if ret:
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    self.oriVideoLabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
                self.timer1.timeout.connect(self.video_show)
                self.timer1.start(30)
            self.clearImageBtn.setEnabled(True)
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 文件上传失败！')

    def clear_image(self):
        self.file_path = None
        self.oriVideoLabel.clear()
        self.clearImageBtn.setEnabled(False)
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 文件已撤销上传。')

    def video_show(self):
        ret, frame = self.cap.read()
        if ret:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.oriVideoLabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
        else:
            self.timer1.stop()

    def show_detect(self):
        """显示检测结果并保存"""
        if self.file_path and self.model:
            if self.file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                frame = cv2.imread(self.file_path)
                results = self.model(frame, conf=self.value)
                rendered_frame = results.render()[0]

                # 保存检测结果
                for det in results.xyxy[0].cpu().numpy():
                    if det[4] >= self.value:  # 置信度阈值过滤
                        x1, y1, x2, y2, conf, cls = det
                        label = self.model.names[int(cls)]
                        self.save_detection_result(label, conf, [x1, y1, x2, y2])

                height, width, channel = rendered_frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(rendered_frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.detectlabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.detectlabel.size(), Qt.KeepAspectRatio))
                self.saveResultBtn.setEnabled(True)
            elif self.file_path.endswith('.mp4'):
                self.timer.timeout.connect(self.video_detect)
                self.timer.start(30)
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请先加载模型并上传文件！')



    def video_detect(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, conf=self.value)
            rendered = results.render()[0]
            height, width, channel = rendered.shape
            bytesPerLine = 3 * width
            qImg = QImage(rendered.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.detectlabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.detectlabel.size(), Qt.KeepAspectRatio))
            self.saveResultBtn.setEnabled(True)
        else:
            self.timer.stop()

    def stop_detect(self):
        """视频检测并保存结果"""
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, conf=self.value)
            rendered = results.render()[0]

            # 保存检测结果
            for det in results.xyxy[0].cpu().numpy():
                if det[4] >= self.value:  # 置信度阈值过滤
                    x1, y1, x2, y2, conf, cls = det
                    label = self.model.names[int(cls)]
                    self.save_detection_result(label, conf, [x1, y1, x2, y2])

            height, width, channel = rendered.shape
            bytesPerLine = 3 * width
            qImg = QImage(rendered.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.detectlabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.detectlabel.size(), Qt.KeepAspectRatio))
            self.saveResultBtn.setEnabled(True)
        else:
            self.timer.stop()

    def draw_label(self, label, text):
        pixmap = QPixmap(label.size())
        pixmap.fill(Qt.transparent)  # 填充透明背景
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(0, 0, 0), 2))  # 设置画笔颜色和宽度
        painter.setFont(QFont("Arial", 16, QFont.Bold))  # 设置字体
        painter.drawText(QPoint(20, 20), text)  # 绘制文本
        painter.end()
        label.setPixmap(pixmap)



    def start_stream(self):
        if not self.is_streaming:
            self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
            if not self.cap.isOpened():
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 无法打开摄像头！')
                return
            self.is_streaming = True
            self.timer.timeout.connect(self.stream_show)  # 连接定时器到显示方法
            self.timer.start(30)  # 设置定时器间隔为30ms
            self.startStreamBtn.setText('⏹️停止视频流')
        else:
            self.stop_stream()

    def stream_show(self):
        ret, frame = self.cap.read()
        if ret:
            # 将捕获的帧转换为QImage并显示
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.oriVideoLabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 摄像头捕获失败！')
            self.stop_stream()

    def stop_stream(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_streaming = False
        self.startStreamBtn.setText('🎥实时视频流')
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频流已停止！')
        self.oriVideoLabel.clear()