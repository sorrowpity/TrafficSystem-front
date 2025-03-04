import os
import csv
import json
from datetime import datetime
import threading
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QTextBrowser,
    QSlider, QDoubleSpinBox, QComboBox, QGridLayout, QMessageBox, QGroupBox, QRadioButton, QToolButton,QDialog,QFormLayout,QLineEdit,QDialogButtonBox,
    QSpinBox,QFrame
)
from PySide6.QtCore import Qt, QTimer, QDir,QPoint,QSize,QEvent,QRect,Signal
from PySide6.QtGui import QPixmap, QImage, QIcon, QPalette,QPen,QPainter,QBrush,QFont,QColor,QPaintEvent,QAction,QRegion,QPainterPath
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from user_manager import UserManager

# 动态检查 TensorFlow 是否可用
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class AvatarCropDialog(QDialog):
    crop_complete = Signal(QPixmap)  # 定义裁剪完成信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("裁剪头像")
        self.setFixedSize(500, 500)

        # 添加鼠标跟踪
        self.setMouseTracking(True)
        
        
        # 初始化变量
        self.original_pixmap = None
        self.drag_start = None
        self.selection_center = QPoint()  # 圆形选区的中心点
        self.selection_radius = 0  # 圆形选区的半径
        self.scale_factor = 1.0
        
        # 创建界面元素
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(self.size())  # 设置图片显示区域大小
        self.image_label.setMouseTracking(True)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept_crop)
        btn_box.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(btn_box)
        self.setLayout(layout)

    def set_image(self, file_path):
        # 加载并缩放图片以适应窗口
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(True)

    def mousePressEvent(self, event):
        # 转换为图片控件的相对坐标
        pos = self.image_label.mapFrom(self, event.pos())
        if self.image_label.rect().contains(pos):
            self.drag_start = pos
            self.selection_center = pos
            self.selection_radius = 0
            self.image_label.update()


    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            # 计算圆形半径
            pos = self.image_label.mapFrom(self, event.pos())
            dx = pos.x() - self.selection_center.x()
            dy = pos.y() - self.selection_center.y()
            self.selection_radius = int((dx**2 + dy**2)**0.5)
            self.image_label.update()

    def mouseReleaseEvent(self, event):
        self.drag_start = None

    def paintEvent(self, event):
        # 只在图片标签上绘制
        painter = QPainter(self.image_label)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.selection_radius > 0:
            # 绘制圆形选区
            painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
            painter.drawEllipse(self.selection_center, self.selection_radius, self.selection_radius)
            
            # 绘制半透明遮罩
            painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
            path = QPainterPath()
            path.addEllipse(self.selection_center, self.selection_radius, self.selection_radius)
            painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
            painter.drawRect(self.image_label.rect())
        painter.end()

    def accept_crop(self):
        if self.selection_center.isNull():
            QMessageBox.warning(self, "警告", "请选择要裁剪的区域")
            return
            
        # 计算原始图片的裁剪区域
        scaled_center = QPoint(
            int(self.selection_center.x() / self.scale_factor),
            int(self.selection_center.y() / self.scale_factor)
        )
        scaled_radius = int(self.selection_radius / self.scale_factor)
        
        # 创建圆形头像
        cropped = self.original_pixmap.copy(
            scaled_center.x() - scaled_radius,
            scaled_center.y() - scaled_radius,
            scaled_radius * 2,
            scaled_radius * 2
        )
        circular = self.make_circular_avatar(cropped)
        self.crop_complete.emit(circular)
        self.accept()

    @staticmethod
    def make_circular_avatar(pixmap):
        size = min(pixmap.width(), pixmap.height())
        target = QPixmap(size, size)
        target.fill(Qt.transparent)
        
        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap.scaled(
            size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        ))
        painter.end()
        
        return target


class MyWindow(QMainWindow):
    def __init__(self,username):
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
        self.current_mode = None  # 当前模式：'image', 'video', 'camera'
        self.save_folder_path = os.path.join(os.getcwd(), "detection_results")  # 默认保存路径
        self.video_writer = None  # 新增视频写入器实例
        os.makedirs(self.save_folder_path, exist_ok=True)
        self.user_manager = UserManager()  # 确保有用户管理实例
        # 从用户管理器获取用户信息
        self.current_username = username
        user_info = self.user_manager.get_user_info(username)
        self.user_info = {
            'name': username,
            'role': user_info.get('role', 'user'),
            'avatar': self.user_manager.get_user_avatar(username)  # 获取存储的头像路径
        }
        # 初始化时加载头像
        self.current_username = username  # 新增当前用户名存储
        self.rtsp_url = ""  # 新增RTSP地址存储
        # 在MyWindow类中添加以下成员变量：
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = datetime.now()
        self.init_gui()
        self.init_user_panel()


    def init_gui(self):
        self.setFixedSize(1300, 700)
        self.setWindowTitle('目标检测')
        self.setWindowIcon(QIcon("logo.jpg"))
        # self.setStyleSheet("background-color: #F0F0F0;")
        self.set_background_image('5.png')  # 设置窗口背景图片

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 修改标题区域
        title_layout = QHBoxLayout()
        
        # 标题标签
        title_label = QLabel("车辆检测系统")
        title_label.setStyleSheet("font-size: 24px; color: black; font-weight: bold; background-color: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # 用户头像按钮
        self.avatar_btn = QToolButton()
        self.avatar_btn.setFixedSize(40, 40)
        self.avatar_btn.setIcon(QIcon(self.user_info['avatar']))
        self.avatar_btn.setIconSize(QSize(36, 36))
        self.avatar_btn.setStyleSheet("""
            QToolButton {
                background-color: white;
                color: black;
                border: 2px solid gray;
                border-radius: 20px;
            }
            QToolButton:hover {
                background-color: #f0f0f0;
            }
            QToolButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        self.avatar_btn.installEventFilter(self)
        self.avatar_btn.setIcon(QIcon(self.user_info['avatar']))
        
        # 将标题和头像按钮添加到布局中
        title_layout.addStretch()  # 添加左侧伸展器，将标题推到中间
        title_layout.addWidget(title_label)
        title_layout.addStretch()  # 添加右侧伸展器，确保标题居中
        title_layout.addWidget(self.avatar_btn)  # 头像按钮放在右侧

        # 设置标题布局的对齐方式为居中（确保标题标签居中）
        title_layout.setAlignment(title_label, Qt.AlignCenter)
        title_layout.setAlignment(self.avatar_btn, Qt.AlignRight)

        # 将标题布局插入到主布局顶部
        main_layout.insertLayout(0, title_layout)


        # 创建一个水平布局用于放置标签和显示区域
        topLayout = QHBoxLayout()

        # 创建原视频显示区域
        self.oriVideoLabel = QLabel(self)
        self.oriVideoLabel.setFixedSize(530, 330)
        self.oriVideoLabel.setStyleSheet('border: 3px solid black; border-radius: 10px; background-color: #F0F0F0;')
        self.oriVideoLabel.setAlignment(Qt.AlignCenter)
        self.oriVideoLabel.setScaledContents(True)  # 设置为True以自动缩放内容

        # 创建检测结果显示区域
        self.detectlabel = QLabel(self)
        self.detectlabel.setFixedSize(530, 330)
        self.detectlabel.setStyleSheet('border: 3px solid black; border-radius: 10px; background-color: #F0F0F0;')
        self.detectlabel.setAlignment(Qt.AlignCenter)
        self.detectlabel.setScaledContents(True)  # 设置为True以自动缩放内容

        # 创建原视频标题
        oriTitleLabel = QLabel("原视频", self)
        oriTitleLabel.setStyleSheet("font-size: 20px; color: black; font-weight: bold; background-color: transparent;")
        oriTitleLabel.setAlignment(Qt.AlignLeft)

        # 创建检测结果标题
        detectTitleLabel = QLabel("检测结果", self)
        detectTitleLabel.setStyleSheet("font-size: 20px; color: black; font-weight: bold; background-color: transparent;")
        detectTitleLabel.setAlignment(Qt.AlignLeft)

        # 创建一个垂直布局用于放置标题和显示区域
        oriLayout = QVBoxLayout()
        oriLayout.addWidget(oriTitleLabel)
        oriLayout.addWidget(self.oriVideoLabel)
        topLayout.addLayout(oriLayout)

        detectLayout = QVBoxLayout()
        detectLayout.addWidget(detectTitleLabel)
        detectLayout.addWidget(self.detectlabel)
        topLayout.addLayout(detectLayout)

        main_layout.addLayout(topLayout)

        # 创建日志打印区域
        log_layout = QHBoxLayout()  # 改为水平布局
        # 创建日志打印区域
        self.outputField = QTextBrowser()
        self.outputField.setFixedSize(1050, 180)
        self.outputField.setStyleSheet("""
            QTextBrowser {
                background-color: #CCCCCC;
                color: #000000;
                border: 2px solid #CCCCCC;
                border-radius: 10px;
            }
        """)

        # 创建状态信息面板
        status_widget = QWidget()
        status_widget.setFixedWidth(200)
        status_layout = QVBoxLayout(status_widget)

        # 帧率显示
        self.fps_label = QLabel("帧率: 0.0")
        self.fps_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #333333;
                background-color: #CCCCCC;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        self.fps_label.setAlignment(Qt.AlignCenter)

        # 目标数量显示
        self.obj_count_label = QLabel("目标数量: 0")
        self.obj_count_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #333333;
                background-color: #CCCCCC;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        self.obj_count_label.setAlignment(Qt.AlignCenter)

        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.obj_count_label)
        status_layout.addStretch()

        log_layout.addWidget(self.outputField)
        log_layout.addWidget(status_widget)

        main_layout.addLayout(log_layout)  # 替换原有的addWidget(self.outputField)



        # 创建底部操作区域
        bottomLayout = QHBoxLayout()

        # 文件上传按钮
        self.openImageBtn = QPushButton('🖼️文件上传')
        self.openImageBtn.setFixedSize(100, 50)
        self.openImageBtn.setStyleSheet("""
            QPushButton {
            background-color: #0078D7;
            color: white;
            border: 1px solid #005EA6;
            border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #00448C;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.openImageBtn.clicked.connect(self.upload_file)
        bottomLayout.addWidget(self.openImageBtn)

        # 撤销文件上传按钮
        self.clearImageBtn = QPushButton('🗑️撤销上传')
        self.clearImageBtn.setFixedSize(100, 50)
        self.clearImageBtn.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
            }
            QPushButton:pressed {
                background-color: #D3D3D3;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.clearImageBtn.clicked.connect(self.clear_image)
        self.clearImageBtn.setEnabled(False)
        bottomLayout.addWidget(self.clearImageBtn)

        # 侧栏展开/收起按钮
        self.sidebarBtn = QToolButton(self)
        self.sidebarBtn.setText('⚙️')
        self.sidebarBtn.setFixedSize(50, 50)
        self.sidebarBtn.setStyleSheet("""
            QToolButton {
                background-color: white;
                color: #0078D7;
                border: 1px solid #0078D7;
                border-radius: 10px;
            }
            QToolButton:hover {
                background-color: #0078D7;
                color: white;
            }
            QToolButton:pressed {
                background-color: #005EA6;
                color: white;
            }
            QToolButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.sidebarBtn.clicked.connect(self.toggle_model_selection)
        bottomLayout.addWidget(self.sidebarBtn)

        main_layout.addLayout(bottomLayout)

        # 模型选择面板
        self.model_selection_panel = QFrame(self)
        self.model_selection_panel.setFixedSize(200, 200)  # 设置面板宽度
        self.model_selection_panel.move(self.width(), 0)  # 初始位置在窗口右侧外
        self.model_selection_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
        """)
        self.model_selection_panel.setLayout(QVBoxLayout())

        # 添加提示语
        self.tip=QLabel("请选择适配的模型👇")
        self.tip.setStyleSheet("""
            QLabel {
                color: #000000;
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
            }
        """)
        self.model_selection_panel.layout().addWidget(self.tip)
        self.yolo_radio = QRadioButton("YOLO")
        self.yolo_radio.setStyleSheet("""
            QRadioButton {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
                padding: 8px;
            }
            QRadioButton:hover {
                background-color: #F0F0F0;
            }
            QRadioButton:pressed {
                background-color: #E0E0E0;
            }
            QRadioButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.yolo_radio.setChecked(True)
        self.tensorflow_radio = QRadioButton("TensorFlow")
        self.tensorflow_radio.setStyleSheet("""
            QRadioButton {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
                padding: 8px;
            }
            QRadioButton:hover {
                background-color: #F0F0F0;
            }
            QRadioButton:pressed {
                background-color: #E0E0E0;
            }
            QRadioButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.pytorch_radio = QRadioButton("PyTorch")
        self.pytorch_radio.setStyleSheet("""
            QRadioButton {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
                padding: 8px;
            }
            QRadioButton:hover {
                background-color: #F0F0F0;
            }
            QRadioButton:pressed {
                background-color: #E0E0E0;
            }
            QRadioButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)

        self.model_selection_panel.layout().addWidget(self.yolo_radio)
        self.model_selection_panel.layout().addWidget(self.tensorflow_radio)
        self.model_selection_panel.layout().addWidget(self.pytorch_radio)

        self.model_selection_panel.setVisible(False)  # 默认隐藏


        # 导入模型按钮
        self.importModelBtn = QPushButton('📂导入模型')
        self.importModelBtn.setFixedSize(100, 50)
        self.importModelBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: 2px solid #005EA6;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #00448C;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.importModelBtn.clicked.connect(self.import_model)
        bottomLayout.addWidget(self.importModelBtn)


        # 保存检测结果按钮
        self.saveResultBtn = QPushButton('💾导出结果')
        self.saveResultBtn.setFixedSize(100, 50)
        self.saveResultBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: 2px solid #005EA6;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #00448C;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
            
        """)
        self.saveResultBtn.clicked.connect(self.save_result)
        self.saveResultBtn.setEnabled(False)  # 默认禁用，只有在检测完成后才启用
        bottomLayout.addWidget(self.saveResultBtn)

        # 保存路径选择按钮
        self.selectFolderBtn = QPushButton('📂选择保存路径')
        self.selectFolderBtn.setFixedSize(100, 50)
        self.selectFolderBtn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #0078D7;
                border: 1px solid #0078D7;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #0078D7;
                color: white;
            }
            QPushButton:pressed {
                background-color: #005EA6;
                color: white;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.selectFolderBtn.clicked.connect(self.select_save_folder)
        bottomLayout.addWidget(self.selectFolderBtn)

        # 置信度阈值滑动条部分
        self.con_label = QLabel('置信度阈值', self)
        self.con_label.setStyleSheet("""
            QLabel {
                color: #000000;
            }
        """)
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
        self.confudence_slider.setStyleSheet("background-color: #CCCCCC;")

        # 开始检测按钮
        self.start_detect = QPushButton('🔍开始检测')
        self.start_detect.setFixedSize(100, 50)
        self.start_detect.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: 2px solid #005EA6;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #00448C;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
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
                background-color: #E04545;
                color: white;
                border: 2px solid #C12C2C;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #C12C2C;
            }
            QPushButton:pressed {
                background-color: #A42121;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        self.stopDetectBtn.setEnabled(False)
        bottomLayout.addWidget(self.stopDetectBtn)

        # # 实时视频流按钮
        # self.startStreamBtn = QPushButton('🎥实时视频流')
        # self.startStreamBtn.setFixedSize(100, 50)
        # self.startStreamBtn.setStyleSheet("""
        #     QPushButton {
        #         background-color: white;
        #         color: black;
        #         border: 2px solid gray;
        #         border-radius: 10px;
        #     }
        #     QPushButton:hover {
        #         background-color: #f0f0f0;
        #     }
        #     QPushButton:pressed {
        #         background-color: #e0e0e0;
        #     }
        #     QPushButton:disabled {
        #         background-color: #d3d3d3;
        #     }
        # """)
        # self.startStreamBtn.clicked.connect(self.start_stream)
        # self.startStreamBtn.setEnabled(True)
        # bottomLayout.addWidget(self.startStreamBtn)

        #RTSP网络流
        self.rtsp_action = QPushButton('🌐RTSP网络流')
        self.rtsp_action.setFixedSize(100, 50)
        self.rtsp_action.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: 2px solid #005EA6;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #00448C;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                border: 1px solid #CCCCCC;
            }
        """)
        self.rtsp_action.clicked.connect(self.setup_rtsp_stream)
        bottomLayout.addWidget(self.rtsp_action)

        # 返回登录按钮
        self.returnLoginBtn = QPushButton('🔙返回登录')
        self.returnLoginBtn.setFixedSize(100, 50)
        self.returnLoginBtn.setStyleSheet("""
        QPushButton {
            background-color: white;
            color: #0078D7;
            border: 1px solid #0078D7;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: #0078D7;
            color: white;
        }
        QPushButton:pressed {
            background-color: #005EA6;
            color: white;
        }
        QPushButton:disabled {
            background-color: #d3d3d3;
            border: 1px solid #CCCCCC;
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
        
        # 确保 bbox 是一个列表
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.tolist()  # 如果 bbox 是 Tensor，转换为列表
        elif isinstance(bbox, list) and all(isinstance(item, torch.Tensor) for item in bbox):
            bbox = [item.tolist() for item in bbox]  # 如果 bbox 是 Tensor 列表，逐个转换为列表

        result = {
            "timestamp": timestamp,
            "vehicle_type": vehicle_type,
            "confidence": float(confidence),  # 确保置信度是浮点数
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

    def process_frame(self, frame):
        """统一处理帧的检测逻辑"""
        if self.model_type == "YOLO":
            results = self.model(frame, conf=self.value)
            rendered = results[0].plot()
        elif self.model_type == "TensorFlow":
            # TensorFlow模型推理逻辑（需要根据具体模型实现）
            pass
        elif self.model_type == "PyTorch":
            # PyTorch模型推理逻辑（需要根据具体模型实现）
            pass

         # 更新帧率统计
        self.frame_count += 1
        elapsed = (datetime.now() - self.last_fps_update).total_seconds()
        # 统计目标数量
        obj_count = len(results[0]) if results else 0
        self.obj_count = obj_count
        if elapsed >= 1.0:  # 每秒更新一次
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = datetime.now()
            self.update_status_display()  # 更新界面显示

        return rendered
    
    # 添加状态更新方法
    def update_status_display(self):
        self.fps_label.setText(f"帧率: {self.fps:.1f}")
        self.obj_count_label.setText(f"目标数量: {self.obj_count}")

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
        exit()


    def save_result(self):
        """导出结果统一入口"""
        if not self.model:
            self.outputField.append("请先加载模型！")
            return

        try:
            if self.current_mode == 'image':
                self.save_image_result()
            elif self.current_mode in ['video', 'camera']:
                self.save_video_result()
        except Exception as e:
            self.outputField.append(f"保存失败: {str(e)}")


    def save_image_result(self):
        """保存图片检测结果"""
        frame = cv2.imread(self.file_path)
        results = self.model(frame, conf=self.value)
        rendered = results[0].plot()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.save_folder_path,
            f"output_image_{timestamp}.jpg"
        )
        
        cv2.imwrite(output_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        self.outputField.append(f"图片已保存: {output_path}")


    def save_video_result(self):
        """手动触发视频保存"""
        if self.video_writer:
            self.finalize_video_writer()
        else:
            self.outputField.append("没有需要保存的视频内容")
    
    def save_video(self):
        """保存带有检测结果的视频"""
        if not self.file_path or not self.file_path.endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm')):
            return False

        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.save_folder_path, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=self.value)
            rendered_frame = results.plot()[0]
            out.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频检测结果已保存到: {output_path}')
        return True


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
        # 支持的文件类型包括常见的图像和视频格式
        file_path, file_type = file_dialog.getOpenFileName(self, "选择检测文件", filter='*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.mp4 *.avi *.mov *.wmv *.mkv *.webm')
        if file_path:
            self.file_path = file_path
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 文件上传成功: {file_path}')
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                pixmap = QPixmap(file_path)
                # 使用 KeepAspectRatioByExpanding 保持宽高比并填充空白
                # 确保图片大小适应窗口
                scaled_pixmap = pixmap.scaled(
                    self.oriVideoLabel.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )
                self.oriVideoLabel.setPixmap(scaled_pixmap)
                self.oriVideoLabel.setScaledContents(True)  # 保持内容缩放
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm')):
                self.cap = cv2.VideoCapture(file_path)
                ret, frame = self.cap.read()
                if ret:
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                    # 放大并填充界面
                    scaled_img = qImg.scaled(
                        self.oriVideoLabel.size(),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation
                    )
                    self.oriVideoLabel.setPixmap(QPixmap.fromImage(scaled_img))
                    self.oriVideoLabel.setScaledContents(True)

                    
                self.timer1.timeout.connect(self.video_show)
                self.timer1.start(30)
            self.clearImageBtn.setEnabled(True)
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 文件上传失败！')

    def is_image_file(filename):
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        _, extension = os.path.splitext(filename)
        return extension.lower() in image_extensions

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
            # 放大并填充界面
            scaled_img = qImg.scaled(
                self.oriVideoLabel.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            self.oriVideoLabel.setPixmap(QPixmap.fromImage(scaled_img))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.timer1.stop()

    def show_detect(self):
        """开始检测（适配图片/视频/摄像头）"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        if self.file_path:
            if self.file_path.endswith(('.jpg', '.jpeg', '.png')):
                self.current_mode = 'image'
                frame = cv2.imread(self.file_path)
                rendered = self.process_frame(frame)
                self.display_result(rendered)
                
            elif self.file_path.endswith(('.mp4', '.avi', '.mov')):
                self.current_mode = 'video'
                self.cap = cv2.VideoCapture(self.file_path)
                self.timer.timeout.connect(self.detect_stream)
                self.timer.start(30)
                
            self.stopDetectBtn.setEnabled(True)
            self.start_detect.setEnabled(False)
            self.saveResultBtn.setEnabled(True)

    def detect_stream(self):
        """处理视频流检测（视频文件或摄像头）"""
        ret, frame = self.cap.read()
        if ret:
            # 初始化视频写入器（仅首次运行）
            if self.current_mode in ['video', 'camera'] and not self.video_writer:
                self.init_video_writer(frame)
                
            # 处理并显示原帧
            self.display_original(frame)
            
            # 进行目标检测
            rendered = self.process_frame(frame)
            self.display_result(rendered)
            
            # 写入视频帧
            if self.video_writer:
                self.video_writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        else:
            self.timer.stop()
            self.finalize_video_writer()
            if self.current_mode == 'video':
                self.cap.release()

    def init_video_writer(self, frame):
        """初始化视频写入器"""
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.current_mode == 'video' else 30.0
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = os.path.join(
            self.save_folder_path,
            f"output_{self.current_mode}_{timestamp}.mp4"
        )
        
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        self.outputField.append(f"视频保存路径: {output_path}")

    def finalize_video_writer(self):
        """关闭视频写入器"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.outputField.append("视频文件已保存")

    def display_original(self, frame):
        """显示原始帧"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.oriVideoLabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.oriVideoLabel.size(), Qt.KeepAspectRatio))
        
    def display_result(self, rendered):
        """显示检测结果"""
        rgb_frame = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.detectlabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.detectlabel.size(), Qt.KeepAspectRatio))
        
    def save_stream_results(self, frame):
        """保存流式检测结果"""
        if not hasattr(self, 'save_folder_path'):
            self.save_folder_path = "./detection_results"  # 默认保存路径
            os.makedirs(self.save_folder_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        output_path = os.path.join(self.save_folder_path, f"output_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        self.outputField.append(f"检测结果已保存：{output_path}")

    def video_detect(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, conf=self.value)
            if results:
                rendered = results[0].plot()[0]  # 修改这里
                height, width, _ = rendered.shape
                qImg = QImage(rendered.data, width, height, QImage.Format_RGB888).rgbSwapped()
                self.detectlabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.detectlabel.size(), Qt.KeepAspectRatio))
                self.saveResultBtn.setEnabled(True)
        else:
            self.timer.stop()

    def stop_detect(self):
        """停止检测"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.finalize_video_writer()
        self.start_detect.setEnabled(True)
        self.stopDetectBtn.setEnabled(False)
        if self.current_mode == 'camera':
            self.startStreamBtn.setEnabled(True)

    def show_stream(self):
        """显示原始摄像头流"""
        ret, frame = self.cap.read()
        if ret:
            self.display_original(frame)

    def draw_label(self, label, text):
        pixmap = QPixmap(label.size())
        pixmap.fill(Qt.transparent)  # 填充透明背景
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(255, 255, 255), 2))  # 设置画笔颜色和宽度
        painter.setFont(QFont("Arial", 14, QFont.Bold))  # 设置字体
        painter.drawText(QPoint(10, 25), text)  # 绘制文本
        painter.end()
        label.setPixmap(pixmap)



    # def start_stream(self):
    #     if not self.is_streaming:
    #         self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
    #         if not self.cap.isOpened():
    #             self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 无法打开摄像头！')
    #             return
    #         self.is_streaming = True
    #         self.timer.timeout.connect(self.stream_show)  # 连接定时器到显示方法
    #         self.timer.start(30)  # 设置定时器间隔为30ms
    #         self.startStreamBtn.setText('⏹️停止视频流')
    #     else:
    #         self.stop_stream()

    # def stream_show(self):
    #     ret, frame = self.cap.read()
    #     if ret:
    #         # 将捕获的帧转换为QImage并显示
    #         height, width, channel = frame.shape
    #         bytesPerLine = 3 * width
    #         qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    #         self.oriVideoLabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
    #     else:
    #         self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 摄像头捕获失败！')
    #         self.stop_stream()

    # def stop_stream(self):
    #     """停止摄像头流"""
    #     self.timer.stop()
    #     if self.cap:
    #         self.cap.release()
    #     self.is_streaming = False
    #     self.startStreamBtn.setText("🎥 实时摄像头")
    #     self.oriVideoLabel.clear()

    def select_save_folder(self):
        """选择保存目录"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder_path:
            self.save_folder_path = folder_path
            os.makedirs(folder_path, exist_ok=True)
            self.outputField.append(f"保存路径已更新: {folder_path}")


    def setup_rtsp_stream(self):
        """设置RTSP网络流"""
        dialog = QDialog(self)
        dialog.setWindowTitle("RTSP设置")
        layout = QFormLayout(dialog)
        
        self.rtsp_input = QLineEdit("rtsp://[用户名]:[密码]@[IP地址]/[路径]")
        test_btn = QPushButton("测试连接")
        test_btn.clicked.connect(self.test_rtsp_connection)
        
        layout.addRow("RTSP地址:", self.rtsp_input)
        layout.addRow(test_btn)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(lambda: self.start_rtsp_stream(self.rtsp_input.text()))
        btn_box.rejected.connect(dialog.reject)
        
        layout.addRow(btn_box)
        dialog.exec()

    def start_rtsp_stream(self, url):
        """启动RTSP流处理"""
        try:
            # 释放原有资源
            if self.cap:
                self.cap.release()
            
            # 初始化视频捕获
            self.rtsp_url = url
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                raise ConnectionError("无法连接RTSP流")
            
            # 启动处理线程
            self.rtsp_thread = threading.Thread(target=self.process_rtsp_stream)
            self.rtsp_thread.daemon = True
            self.rtsp_thread.start()
            
            # 启动显示定时器
            self.timer.start(30)
            self.outputField.append(f"已连接RTSP流: {url}")
            
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"RTSP连接错误: {str(e)}")

    def process_rtsp_stream(self):
        """RTSP流处理线程"""
        while self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.outputField.append("RTSP流中断，尝试重连...")
                    self.cap.reopen()  # 自定义重连方法
                    continue
                
                # 进行目标检测
                results = self.model(frame, device='0',conf=self.value)
                rendered = results[0].plot(line_width=self.line_width)
                
                # 发送信号更新UI
                self.update_signal.emit(rendered)
                
            except Exception as e:
                self.outputField.append(f"RTSP处理错误: {str(e)}")
                break

    def test_rtsp_connection(self):
        """测试RTSP连接"""
        test_cap = cv2.VideoCapture(self.rtsp_input.text())
        if test_cap.isOpened():
            QMessageBox.information(self, "连接成功", "RTSP流测试连接成功！")
            test_cap.release()
        else:
            QMessageBox.warning(self, "连接失败", "无法连接指定RTSP地址")


    def init_user_panel(self):
        """初始化用户管理面板"""
        self.user_panel = QWidget(self)
        self.user_panel.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.user_panel.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 1px solid #0078D7;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            QPushButton {
                text-align: left;
                padding: 8px 20px;
                border: none;
                min-width: 120px;
                background-color: #0078D7;
                color: white;
                border-radius: 4px;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QLabel {
                color: #000000;
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(self.user_panel)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # 用户信息展示
        self.user_info_label = QLabel(f"{self.user_info['name']}\n({self.user_info['role']})")
        self.user_info_label.setStyleSheet("""
            QLabel {
                color: #000000;
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
            }
        """)
        self.user_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.user_info_label)
        
        # 功能按钮
        self.profile_btn = QPushButton("个人信息")
        self.profile_btn.setStyleSheet("""
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
        self.profile_btn.clicked.connect(self.show_profile)
        layout.addWidget(self.profile_btn)

        # 添加更换头像按钮
        self.change_avatar_btn = QPushButton("更换头像")
        self.change_avatar_btn.setStyleSheet("""
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
        self.change_avatar_btn.clicked.connect(self.change_avatar)
        layout.insertWidget(1, self.change_avatar_btn)  # 插入到个人信息下方
        
        self.change_pwd_btn = QPushButton("修改密码")
        self.change_pwd_btn.setStyleSheet("""
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
        self.change_pwd_btn.clicked.connect(self.show_change_password)
        layout.addWidget(self.change_pwd_btn)
        
        if self.user_info['role'] == 'admin':
            self.manage_users_btn = QPushButton("用户管理")
            self.manage_users_btn.setStyleSheet("""
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
            self.manage_users_btn.clicked.connect(self.show_user_management)
            layout.addWidget(self.manage_users_btn)
        
        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setStyleSheet("""
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
        self.logout_btn.clicked.connect(self.close_user)
        layout.addWidget(self.logout_btn)
        
        self.user_panel.adjustSize()
        self.user_panel.hide()

    
    def eventFilter(self, obj, event):
        """处理头像按钮的悬停事件"""
        if obj == self.avatar_btn:
            if event.type() == QEvent.Enter:
                self.show_user_panel()
            elif event.type() == QEvent.Leave:
                QTimer.singleShot(200, self.check_mouse_position)
        return super().eventFilter(obj, event)

    def show_user_panel(self):
        """显示用户面板"""
        pos = self.avatar_btn.mapToGlobal(QPoint(0, self.avatar_btn.height()))
        self.user_panel.move(pos)
        self.user_panel.show()

    def check_mouse_position(self):
        """检查鼠标是否仍在面板区域"""
        if not self.user_panel.underMouse():
            self.user_panel.hide()

    def show_profile(self):
        """显示个人信息"""
        profile_dialog = QDialog(self)
        profile_dialog.setWindowTitle("个人信息")
        layout = QFormLayout(profile_dialog)
        
        info = [
            ("用户名", self.user_info['name']),
            ("角色", self.user_info['role']),
            ("注册时间", "2023-01-01"),
            ("最后登录", "2023-12-01")
        ]
        
        for label, value in info:
            layout.addRow(QLabel(label), QLabel(value))
        
        profile_dialog.exec()

    def show_change_password(self):
        """显示修改密码对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("修改密码")
        layout = QFormLayout(dialog)
        
        # 创建输入框
        old_pwd = QLineEdit()
        old_pwd.setEchoMode(QLineEdit.Password)
        new_pwd = QLineEdit()
        new_pwd.setEchoMode(QLineEdit.Password)
        confirm_pwd = QLineEdit()
        confirm_pwd.setEchoMode(QLineEdit.Password)
        
        # 添加输入验证提示
        new_pwd.setPlaceholderText("至少6位字符")
        confirm_pwd.setPlaceholderText("再次输入新密码")
        
        # 创建按钮框
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        # 添加组件到布局
        layout.addRow("原密码:", old_pwd)
        layout.addRow("新密码:", new_pwd)
        layout.addRow("确认密码:", confirm_pwd)
        layout.addRow(btn_box)
        
        # 设置对话框固定大小
        dialog.setFixedSize(300, 180)
        
        # 连接信号
        def validate_and_save():
            # 获取输入值
            old = old_pwd.text().strip()
            new = new_pwd.text().strip()
            confirm = confirm_pwd.text().strip()
            
            # 基础验证
            if not all([old, new, confirm]):
                QMessageBox.warning(dialog, "错误", "所有字段必须填写！")
                return
                
            if new != confirm:
                QMessageBox.warning(dialog, "错误", "两次输入的新密码不一致！")
                return
                
            # 调用用户管理器修改密码
            success, msg = self.user_manager.update_password(
                self.current_username,
                old,
                new
            )
            
            if success:
                QMessageBox.information(self, "成功", msg)
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "错误", msg)
        
        btn_box.accepted.connect(validate_and_save)
        btn_box.rejected.connect(dialog.reject)
        
        # 显示对话框
        dialog.exec()

    def change_avatar(self):
        """更换头像入口方法"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择头像图片", 
            filter="图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
    
        if file_path:
            self.show_crop_dialog(file_path)

    def show_crop_dialog(self, file_path):
        """显示裁剪对话框"""
        dialog = AvatarCropDialog(self)
        dialog.set_image(file_path)
        dialog.crop_complete.connect(self.update_avatar)
        dialog.exec()

    def update_avatar(self, pixmap):
        """更新头像显示并保存"""
        # 创建avatars目录
        avatar_dir = "avatars"
        os.makedirs(avatar_dir, exist_ok=True)
        
        # 生成用户专属文件名
        save_path = os.path.join(avatar_dir, f"{self.current_username}.png")
        
        try:
            # 保存图片并更新用户数据
            pixmap.save(save_path, "PNG", quality=100)
            self.user_manager.update_avatar(self.current_username, save_path)
            
            # 更新界面显示
            self.user_info['avatar'] = save_path
            self.avatar_btn.setIcon(QIcon(pixmap))
            self.avatar_btn.setIconSize(QSize(36, 36))
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"头像保存失败: {str(e)}")

    def toggle_model_selection(self):
        """切换模型选择面板的显示状态"""
        button_pos = self.sidebarBtn.pos()  # 获取按钮的位置
        if self.model_selection_panel.isVisible():
            self.model_selection_panel.setVisible(False)  # 隐藏面板
        else:
            # 设置面板的初始位置为按钮右侧
            panel_x = button_pos.x()
            panel_y = button_pos.y() - self.model_selection_panel.height() - 10  # 10 是面板与按钮之间的间距
            self.model_selection_panel.move(panel_x, panel_y)
            self.model_selection_panel.setVisible(True)

# 在原有代码中添加视频流重连逻辑
    class VideoCaptureWithReconnect(cv2.VideoCapture):
        def __init__(self, url, max_retries=3):
            super().__init__(url)
            self.url = url
            self.max_retries = max_retries
            self.retry_count = 0
            
        def reopen(self):
            if self.retry_count < self.max_retries:
                self.release()
                self.open(self.url)
                self.retry_count += 1
                return self.isOpened()
            return False
