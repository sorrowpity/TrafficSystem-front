# admin_window.py
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit,
    QLabel, QMessageBox, QListWidget, QGridLayout, QDateEdit, QComboBox, QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from user_manager import UserManager


class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.user_manager = UserManager()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(1300, 700)
        self.setWindowTitle("用户管理")
        self.setWindowIcon(QIcon("logo.jpg"))  # 设置窗口图标

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("用户管理系统")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        main_layout.addLayout(title_layout)

        # 用户列表区域
        user_list_frame = QFrame()
        user_list_frame.setFrameShape(QFrame.StyledPanel)
        user_list_frame.setFrameShadow(QFrame.Raised)
        user_list_layout = QVBoxLayout(user_list_frame)
        self.user_list = QListWidget()
        self.user_list.itemClicked.connect(self.on_user_list_item_clicked)
        self.populate_user_list()
        user_list_layout.addWidget(self.user_list)
        main_layout.addWidget(user_list_frame)

        # 用户管理操作区域
        user_management_frame = QFrame()
        user_management_frame.setFrameShape(QFrame.StyledPanel)
        user_management_frame.setFrameShadow(QFrame.Raised)
        user_management_layout = QHBoxLayout(user_management_frame)

        self.add_user_button = QPushButton("添加用户")
        self.add_user_button.setFixedSize(100, 50)
        self.add_user_button.clicked.connect(self.add_user)
        user_management_layout.addWidget(self.add_user_button)

        self.delete_user_button = QPushButton("删除用户")
        self.delete_user_button.setFixedSize(100, 50)
        self.delete_user_button.clicked.connect(self.delete_user)
        user_management_layout.addWidget(self.delete_user_button)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("输入用户名")
        user_management_layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        user_management_layout.addWidget(self.password_input)

        self.role_input = QComboBox()
        self.role_input.addItems(["admin", "user"])
        user_management_layout.addWidget(self.role_input)

        main_layout.addWidget(user_management_frame)

        # 状态标签
        self.status_label = QLabel("状态：")
        self.status_label.setStyleSheet("color: gray; font-size: 14px;")
        main_layout.addWidget(self.status_label)

        # 返回登录按钮
        self.back_button = QPushButton("返回登录")
        self.back_button.setFixedSize(100, 50)
        self.back_button.clicked.connect(self.close_admin)
        main_layout.addWidget(self.back_button)

        # 用户详细信息区域
        detail_frame = QFrame()
        detail_frame.setFrameShape(QFrame.StyledPanel)
        detail_frame.setFrameShadow(QFrame.Raised)
        detail_layout = QGridLayout(detail_frame)

        # 用户名
        detail_layout.addWidget(QLabel("用户名："), 0, 0)
        self.detail_username_label = QLabel()
        detail_layout.addWidget(self.detail_username_label, 0, 1)

        # 角色（权限）
        detail_layout.addWidget(QLabel("角色："), 1, 0)
        self.detail_role_label = QLabel()
        detail_layout.addWidget(self.detail_role_label, 1, 1)

        # 注册时间
        detail_layout.addWidget(QLabel("注册时间："), 2, 0)
        self.detail_registration_date_label = QLabel()
        detail_layout.addWidget(self.detail_registration_date_label, 2, 1)

        # 用户账号（邮箱或其他标识）
        detail_layout.addWidget(QLabel("账号："), 3, 0)
        self.detail_account_label = QLabel()
        detail_layout.addWidget(self.detail_account_label, 3, 1)

        main_layout.addWidget(detail_frame)

    def populate_user_list(self):
        self.user_list.clear()
        for username in self.user_manager.list_users():
            item = QListWidgetItem(username)
            item.setData(Qt.UserRole, username)  # 存储原始用户名
            self.user_list.addItem(item)

    def add_user(self):
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.role_input.currentText()
        if not username or not password:
            self.status_label.setText("状态：请输入用户名和密码")
            return
        success, message = self.user_manager.add_user(username, password, role)
        self.status_label.setText(f"状态：{message}")
        self.populate_user_list()

    def delete_user(self):
        username = self.username_input.text().strip()
        success,message = self.user_manager.delete_user(username)
        self.status_label.setText(f"状态：{message}")
        self.populate_user_list()
        print(f"尝试删除用户: {username}, 结果: {message}")

    def close_admin(self):
        self.close()  # 关闭当前窗口

    def closeEvent(self, event):
        """重写 closeEvent，在关闭窗口时返回登录界面"""
        event.accept()  # 接受关闭事件

    def on_user_list_item_clicked(self, item):
        username = item.data(Qt.UserRole)
        user_info = self.user_manager.get_user_info(username)
        if user_info:
            self.detail_username_label.setText(user_info.get("username", "未知"))
            self.detail_role_label.setText(user_info.get("role", "未知"))
            self.detail_registration_date_label.setText(user_info.get("registration_date", "未知"))
            self.detail_account_label.setText(user_info.get("account", "未知"))

            # 将用户名显示在输入框中
            self.username_input.setText(username)
        else:
            self.detail_username_label.setText("未知")
            self.detail_role_label.setText("未知")
            self.detail_registration_date_label.setText("未知")
            self.detail_account_label.setText("未知")