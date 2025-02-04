# login.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QApplication,
                                QLineEdit, QFormLayout,QComboBox)
from user_manager import UserManager
from PySide6.QtGui import QIcon,QFont
from PySide6.QtCore import Qt

class RegisterDialog(QDialog):
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("注册")
        self.setFixedSize(400, 400)
        self.setWindowIcon(QIcon("logo.jpg"))
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)
        

        self.username_label = QLabel("用户名:")
        self.username_label.setStyleSheet("color: black; font-size: 14px;")  # 设置标签字体颜色为黑色
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("输入用户名")
        self.username_input.setFixedWidth(300)
        layout.addRow(self.username_label, self.username_input)

        self.password_label = QLabel("密码:")
        self.password_label.setStyleSheet("color: black; font-size: 14px;")  # 设置标签字体颜色为黑色
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("输入密码")
        self.password_input.setFixedWidth(300)
        layout.addRow(self.password_label, self.password_input)

        self.confirm_password_label = QLabel("确认密码:")
        self.confirm_password_label.setStyleSheet("color: black; font-size: 14px;")  # 设置标签字体颜色为黑色
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setPlaceholderText("确认密码")
        self.confirm_password_input.setFixedWidth(300)
        layout.addRow(self.confirm_password_label, self.confirm_password_input)

        self.email_label = QLabel("邮箱:")
        self.email_label.setStyleSheet("color: black; font-size: 14px;")  # 设置标签字体颜色为黑色
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("输入邮箱")
        self.email_input.setFixedWidth(300)
        layout.addRow(self.email_label, self.email_input)

        self.role_label = QLabel("角色:")
        self.role_label.setStyleSheet("color: black; font-size: 14px;")  # 设置标签字体颜色为黑色
        self.role_input = QComboBox()
        self.role_input.addItems(["user"])
        self.role_input.setFixedWidth(300)
        layout.addRow(self.role_label, self.role_input)

        self.register_button = QPushButton("注册")
        self.register_button.setFont(QFont("Arial", 12))
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #004E8C;
            }
        """)
        self.register_button.clicked.connect(self.handle_register)
        layout.addRow(self.register_button)

    def handle_register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        email = self.email_input.text().strip()
        role = self.role_input.currentText()

        if not username or not password or not confirm_password or not email:
            QMessageBox.warning(self, "错误", "所有字段均为必填项！")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致！")
            return

        if self.user_manager.add_user(username, password, role, email):
            QMessageBox.information(self, "成功", "注册成功！")
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "注册失败，用户名已存在！")


class CustomMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QLabel {
                color: black;
            }
            QPushButton {
                background-color: #f0f0f0;
                color: black;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)

class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_manager = UserManager()
        self.setWindowTitle("登录")
        self.setFixedSize(400, 350)
        self.setWindowIcon(QIcon("logo.jpg"))
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("目标检测系统登录")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title_label)

        self.username_label = QLabel("用户名:")
        self.username_label.setStyleSheet("font-size: 14px;color: #333;")
        self.username_input = QLineEdit()
        self.username_input.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)

        self.password_label = QLabel("密码:")
        self.password_label.setStyleSheet("font-size: 14px;color: #333;")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("登录")
        self.login_button.setFont(QFont("Arial", 12))
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #004E8C;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)

        self.register_button = QPushButton("注册")
        self.register_button.setFont(QFont("Arial", 12))
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005EA6;
            }
            QPushButton:pressed {
                background-color: #004E8C;
            }
        """)
        self.register_button.clicked.connect(self.show_register_dialog)
        layout.addWidget(self.register_button)


        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                margin: 5px 0;
            }
            QLineEdit {
                margin: 5px 0;
            }
        """)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if self.user_manager.validate_user(username, password):
            role = self.user_manager.get_user_role(username)
            if role == "admin":
                self.accept()  # 登录成功，关闭对话框并返回 Accepted
            else:
                self.accept()  # 登录成功，关闭对话框并返回 Accepted
        else:
            # 使用自定义的 CustomMessageBox 显示警告信息
            msg_box = CustomMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("错误")
            msg_box.setText("用户名或密码错误！")
            msg_box.exec()

    def show_register_dialog(self):
        register_dialog = RegisterDialog(self.user_manager, self)
        register_dialog.exec()

    def accept_admin(self, username):
        # 显示管理员界面
        from admin_window import AdminWindow
        self.admin_window = AdminWindow()
        self.admin_window.show()
        self.close()

    def accept_user(self, username):
        # 显示普通用户界面
        from main_window import MyWindow
        self.user_window = MyWindow()
        self.user_window.show()
        self.close()

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()
        