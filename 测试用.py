import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        layout = QVBoxLayout()
        # 创建一个Figure对象
        fig = Figure()
        # 创建一个FigureCanvas对象，并将Figure对象传递给它
        canvas = FigureCanvas(fig)
        # 将FigureCanvas对象添加到布局中
        layout.addWidget(canvas)
        # 在Figure对象上创建一个子图，并返回Axes对象
        ax = fig.add_subplot(111)
        # 使用Matplotlib的绘图函数在子图上绘制曲线
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        # 创建一个NavigationToolbar对象，并将FigureCanvas对象传递给它
        toolbar = NavigationToolbar(canvas, self)
        # 将NavigationToolbar对象添加到布局中
        layout.addWidget(toolbar)
        
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())