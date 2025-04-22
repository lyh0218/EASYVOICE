import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QSplitter)
from PyQt6.QtCore import Qt


class MediaPlayerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口初始大小为屏幕的60%
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen_geometry.width() * 0.6), int(screen_geometry.height() * 0.6))
        self.setWindowTitle("EASYVOICE")

        # 主窗口部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # 主布局 - 垂直布局（顶部+底部）
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.main_layout.setContentsMargins(5,5,5,5)
        self.main_layout.setSpacing(5)

        # 1. 顶部区域 (1/4高度)
        self.create_top_area()

        # 2. 底部区域 (水平分割为左右两部分)
        self.create_bottom_area()

    def create_top_area(self):
        """创建顶部区域 (固定1/4高度)"""
        self.top_frame = QWidget()
        self.top_frame.setFixedHeight(int(self.height() * 0.2))

        # 顶部区域布局
        self.top_layout = QHBoxLayout(self.top_frame)
        self.top_layout.setContentsMargins(5,5,5,5)

        # 左侧区域
        self.create_top_left_area()

        # 右侧区域
        self.create_top_right_area()

        # 添加top区域
        self.main_layout.addWidget(self.top_frame)

    def create_top_left_area(self):
        self.top_left = QWidget()
        self.top_left.setStyleSheet("background-color: #2c3e50;")
        self.top_layout.addWidget(self.top_left, stretch=2)  # 左侧占2份

    def create_top_right_area(self):
        self.top_right = QWidget()
        self.top_right.setStyleSheet("background-color: #34495e;")
        self.top_layout.addWidget(self.top_right, stretch=1)  # 右侧占1份




    def create_bottom_area(self):
        """创建底部区域 (水平分割为左右两部分)"""
        # 使用QSplitter实现可调整的分区
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧区域 (1/5宽度)
        self.create_bottom_left_area()

        # 右侧区域 (4/5宽度)
        self.create_bottom_right_area()

        # 添加到splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)

        # 设置初始比例 (1:4)
        self.splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])

        self.main_layout.addWidget(self.splitter)

    def create_bottom_left_area(self):
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("background-color: #ecf0f1;")
        self.left_panel.setMinimumWidth(200)  # 设置最小宽度

    def create_bottom_right_area(self):
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("background-color: #ffffff;")

    def resizeEvent(self, event):
        """窗口大小改变时保持顶部区域高度比例"""
        self.top_frame.setFixedHeight(int(self.height() * 0.2))
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MediaPlayerApp()
    window.show()
    sys.exit(app.exec())