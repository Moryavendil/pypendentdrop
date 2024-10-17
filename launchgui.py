import sys
from PySide6.QtWidgets import QApplication
from ppd.gui import ppd_mainwindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainwindow = ppd_mainwindow()

    mainwindow.show()
    sys.exit(app.exec())