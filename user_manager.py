# user_manager.py
import json
import os
from datetime import datetime


class UserManager:
    def __init__(self, data_file="users.json"):
        self.data_file = data_file
        self.users = self._load_users()  # 从文件加载用户数据

    def _load_users(self):
        """从文件加载用户数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # 如果文件损坏或为空，初始化默认用户
                return self._initialize_default_users()
        else:
            # 如果文件不存在，初始化默认用户
            return self._initialize_default_users()
        
        
    def _initialize_default_users(self):
        """初始化默认用户"""
        default_users = {
            "admin": {
                "username": "admin",
                "password": "123456",
                "role": "admin",
                "registration_date": "2023-01-01",
                "account": "admin@example.com"
            },
            "user": {
                "username": "user",
                "password": "123456",
                "role": "user",
                "registration_date": "2023-02-01",
                "account": "user@example.com"
            }
        }
        self._save_users(default_users)  # 保存默认用户数据
        return default_users
        
    def _save_users(self, users):
        """保存用户数据到文件"""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存用户数据失败: {e}")
            return False

    def add_user(self, username, password, role="user", email=None):
        if username in self.users:
            return False, "用户已存在"
        self.users[username] = {
            "username": username,
            "password": password,
            "role": role,
            "registration_date": datetime.now().strftime("%Y-%m-%d"),
            "account": email if email else username + "@example.com"
        }
        self._save_users(self.users)  # 保存更新后的用户数据
        return True, "用户添加成功"

    def delete_user(self, username):
        print(f"删除用户请求: {username}")
        if username in self.users:
            del self.users[username]
            if self._save_users(self.users):  # 保存更新后的用户数据
                return True, "用户删除成功"
            else:
                return False, "用户删除失败，保存数据时出错"
        return False, "用户不存在"

    def validate_user(self, username, password):
        user = self.users.get(username)
        if user is None:
            return False  # 用户不存在
        return user["password"] == password

    def get_user_role(self, username):
        user = self.users.get(username)
        return user["role"] if user else None

    def list_users(self):
        return list(self.users.keys())
    
    def get_user_info(self, username):
        """获取用户的详细信息"""
        return self.users.get(username)