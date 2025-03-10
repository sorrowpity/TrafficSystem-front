import sys
from PySide6.QtWidgets import QApplication, QDialog
from login import LoginWindow
from admin_window import AdminWindow
from main_window import MyWindow


def main():
    app = QApplication(sys.argv)



    while True:
        login_window = LoginWindow()
        result = login_window.exec()
        if result == QDialog.DialogCode.Accepted:
            username = login_window.username_input.text()
            user_manager = login_window.user_manager
            role = user_manager.get_user_role(username)

            if role == "admin":
                admin_window = AdminWindow()
                admin_window.show()
                app.exec()  # 进入主事件循环，等待管理员窗口关闭
            else:
                main_window = MyWindow(username)
                main_window.show()
                app.exec()  # 进入主事件循环，等待用户窗口关闭
        else:
            break  # 如果用户取消登录，退出循环

    sys.exit()  # 退出程序


if __name__ == "__main__":
    main()