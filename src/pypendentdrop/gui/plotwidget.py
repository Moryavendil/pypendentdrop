import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, List
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QRectF
from pyqtgraph.Qt import QtGui

import pypendentdrop as ppd

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOption('background', (255,255,255, 100))


# Custom Isocurve item with an offset
class IsocurveItemWithROI(pg.IsocurveItem):

    def __init__(self, offset=None):
        super().__init__(self)
        self.roi = [0,0, None, None] # TLx, TLy, BRx, BRy

    def setROI(self, ROIpos, ROIsize):
        self.roi = [int(ROIpos[0]), int(ROIpos[1]),
                    int(ROIpos[0]+ROIsize[0]), int(ROIpos[1]+ROIsize[1])] # TLx, TLy, BRx, BRy
        ppd.trace(f'isoCurve: setROI: roi is now {self.roi}')

        self.generatePath()
        self.update()

    def generatePath(self):
        if self.data is None:
            self.path = None
            return
        
        # if self.axisOrder == 'row-major':
        #     data = self.data.T
        # else:
        #     data = self.data
        data = self.data

        # lines = pg.isocurve(data, self.level, connected=True, extendToEdge=True)
        lines = ppd.detect_contourlines(data, self.level, roi=self.roi)
        self.path = QtGui.QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)

class ppd_plotWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parentWidget):
        super().__init__(parentWidget)

        # A plot area (ViewBox + axes) for displaying the image
        self.plot = self.addPlot(title="")
        self.plot.autoRange()

        self.defaultLevel:float = 127
        
        # Item for displaying image data
        self.imageItem = pg.ImageItem()
        self.plot.addItem(self.imageItem)
        self.imageItem.getViewBox().invertY(True)
        self.imageItem.getViewBox().setAspectLocked(lock=True, ratio=1)
        self.imageItem.getViewBox().enableAutoRange()
        
        self.load_image(filepath=None)
        self.computedProfilePlotDataItem = None
        self.dropTipPlotDataItem = None
        
        self._setupIsocurve()
        self._setupROI()
        self._setupHistogram()
        
        # self.resize(800, 800)
        # self.show()
        
    def load_image(self, filepath:str = None) -> bool:
        success, self.data = ppd.import_image(filepath)
            
        self.imgheight, self.imgwidth = self.data.shape
        self.imageItem.clear()
        self.imageItem.setImage(self.data)
        # self.imageItem.getViewBox().setLimits(xMin=0-max(0, (height-width)/2), xMax=width+max(0, (height-width)/2), 
        #                                       yMin=0-max(0, (width-height)/2), yMax=height+max(0, (width-height)/2))
        
        if success:
            self.iso.setData(self.data, level=self.defaultLevel)

            ROIpos = [self.imgwidth*(1-self.initialROIfract)/2, self.imgheight*(1-self.initialROIfract)/2]
            ROIsize = [self.imgwidth*self.initialROIfract, self.imgheight*self.initialROIfract]
            ROIpos = np.array(ROIpos).astype(int)
            ROIsize = np.array(ROIsize).astype(int)
            ppd.trace(f'plotWidget: load_image: ROI position = {ROIpos} | ROI size = {ROIsize}')
            ROImaxBounds = QRectF(0-.5, 0-.5, self.imgwidth+1, self.imgheight+1)
            self.roi.maxBounds = ROImaxBounds
            self.roi.setPos(ROIpos)
            self.roi.setSize(ROIsize)

            self.iso.setData(self.data, level=self.defaultLevel)
            self.iso.setROI(ROIpos, ROIsize)
            self.isoCtrlLine.setValue(self.defaultLevel)

            self.hist.setLevels(self.data.min(), self.data.max())

            if self.computedProfilePlotDataItem is not None:
                self.computedProfilePlotDataItem.setVisible(False)

        return success
        
    def _setupIsocurve(self, level=None):
        if level is None:
            level = self.defaultLevel
        ### ISOCURVE
        self.iso = IsocurveItemWithROI()
        self.iso.setPen(pg.mkPen(color='g', width=2))
        self.iso.setLevel(level)
        self.iso.setParentItem(self.imageItem)
        self.iso.setZValue(5)
        # We can also use smnoothed data for the isoline, giving smoother results: 
        # self.iso.setData(pg.gaussianFilter(data, (2, 2)))
        # Here we decide to use the raw data
        self.iso.setData(self.data) 
    
    def _setupROI(self):        
        # Custom ROI for selecting an image region
        self.initialROIfract = .95
        ROIpos = [self.imgwidth*(1-self.initialROIfract)/2, self.imgheight*(1-self.initialROIfract)/2]
        ROIsize = [self.imgwidth*self.initialROIfract, self.imgheight*self.initialROIfract]
        ROIpen = pg.mkPen(color='y', width=1)
        ROIhoverPen = pg.mkPen(color='b', width=1)
        ROIhandlePen = pg.mkPen(color='y', width=2)
        ROIhandleHoverPen = pg.mkPen(color=QtGui.QColor('cyan'), width=2)
        ROImaxBounds = QRectF(0-.5, 0-.5, self.imgwidth+1, self.imgheight+1)
        ROImaxBounds = QRectF(0-.5, 0-.5, self.imgwidth+1, self.imgheight+1)
                
        self.roi = pg.ROI(ROIpos, 
                          ROIsize, 
                          rotatable=False, resizable = False, 
                          pen=ROIpen, hoverPen=ROIhoverPen, 
                          handlePen = ROIhandlePen, handleHoverPen = ROIhandleHoverPen,
                          maxBounds=ROImaxBounds)
        # Side handles
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.])
        self.roi.addScaleHandle([0.5, 0], [0.5, 1.])
        self.roi.addScaleHandle([1, 0.5], [0, 0.5])
        self.roi.addScaleHandle([0, 0.5], [1, 0.5])
        
        # corner handles
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([1, 0], [0, 1])
        
        self.roi.setZValue(10)  # make sure ROI is drawn above image
        self.plot.addItem(self.roi)
        
        # self.roi.sigRegionChanged.connect(self.updateROI)
        # self.updateROI()

    def _setupHistogram(self):
        # Contrast/color control with histogram
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.imageItem)
        self.hist.vb.setMouseEnabled(y=False) # disable axis movement
        self.hist.setHistogramRange(0, 255)
        self.hist.region.setBounds((0, 255))
        self.hist.vb.setLimits(yMin=0-5, yMax=255+5)
        self.hist.setLevels(self.data.min(), self.data.max())
        
        self.addItem(self.hist)
        
        # Draggable line for setting isocurve level
        self.isoCtrlLine = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color='g', width=2))
        self.isoCtrlLine.setValue(self.iso.level)
        self.isoCtrlLine.setZValue(1000) # bring iso line above contrast controls
        self.isoCtrlLine.setMovable(False) # disabled by default
        # self.isoCtrlLine.sigDragged.connect(self.updateIsocurve)
        
        self.hist.vb.addItem(self.isoCtrlLine)

    def set_manualIsoCurve_immobile(self, immobile:bool):
        self.isoCtrlLine.setMovable(not(immobile))

    def disable_isoCurve(self):
        self.iso.visible(False)
        
    def enable_isoCurve(self):
        self.iso.visible(False)
        
    def isoCurve_level(self, level:Optional[float]=None):
        if level is None:
            level = self.defaultLevel
        self.isoCtrlLine.setValue(level)
        return ppd.detect_main_contour(self.iso.data, level, roi=self.iso.roi)

    def plot_computed_profile(self, x, y):

        # imv_v = self.imageItem.getView()
        if self.computedProfilePlotDataItem is None:
            # todo: replace this PlotCurveItem by a PlotDataItem (pyqtgraph recomends doing so)
            self.computedProfilePlotDataItem = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color='r', width=2))
            self.plot.addItem(self.computedProfilePlotDataItem)
        else:
            self.computedProfilePlotDataItem.setData(x=x, y=y)
            self.computedProfilePlotDataItem.setVisible(True)

    def hide_computed_profile(self):
        if self.computedProfilePlotDataItem is not None:
            self.computedProfilePlotDataItem.setVisible(False)

    def scatter_droptip(self, xy:Tuple[float, float]):
        if self.dropTipPlotDataItem is None:
            self.dropTipPlotDataItem = pg.PlotDataItem(x=[xy[0]], y=[xy[1]], symbol='o',
                                                       symbolPen=pg.mkPen((0, 255, 0), width=3), symbolBrush=(0, 0, 0))

            self.dropTipPlotDataItem.setZValue(100)
            self.plot.addItem(self.dropTipPlotDataItem)
        else:
            self.dropTipPlotDataItem.setData(x=[xy[0]], y=[xy[1]])
            self.dropTipPlotDataItem.setVisible(True)
    def hide_scatter_droptip(self):
        if self.dropTipPlotDataItem is not None:
            self.dropTipPlotDataItem.setVisible(False)


































