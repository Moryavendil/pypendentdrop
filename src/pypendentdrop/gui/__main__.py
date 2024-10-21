import pyqtgraph as pg
from .mainwindow import ppd_mainwindow

if __name__ == '__main__':
    app = pg.mkQApp("PyPendentDrop")

    mainwindow = ppd_mainwindow()

    mainwindow.show()
    
    pg.exec()
