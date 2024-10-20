# Main window
from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np

from pyqtgraph.Qt.QtGui import QPixmap
from pyqtgraph.Qt.QtWidgets import QMainWindow, QFileDialog

from mainwindow_ui import Ui_PPD_MainWindow
from plotwidget import ppd_plotWidget

from .. import analyze


RAD_PER_DEG = np.pi/180
DEG_PER_RAD = 180/np.pi

class ppd_mainwindow(QMainWindow, Ui_PPD_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self) # we benefit of the double

        ### LEFT SIDE

        # The widget #0 of displayStackedWidget is the welcome image illustration
        pixmap = QPixmap('assets/images/ppd_illustration.png')
        self.welcomeIcon.setPixmap(pixmap)

        # # The widget #1 of displayStackedWidget is the image shown by pyqtgraph
        self.plotWidget = ppd_plotWidget(self)
        self.displayStackedWidget.addWidget(self.plotWidget)

        self.plotWidget.roi.sigRegionChanged.connect(self.ROIMoved)
        self.ROI_TL_x_spinBox.editingFinished.connect(self.ROIChanged)
        self.ROI_TL_y_spinBox.editingFinished.connect(self.ROIChanged)
        self.ROI_BR_x_spinBox.editingFinished.connect(self.ROIChanged)
        self.ROI_BR_y_spinBox.editingFinished.connect(self.ROIChanged)

        self.plotWidget.isoCtrlLine.sigDragged.connect(self.thresholdMoved)
        self.customThresholdSpinBox.editingFinished.connect(self.thresholdChanged)

        self.autoThresholdCheckBox.toggled.connect(self.plotWidget.set_manualIsoCurve_immobile)

        ### RIGHT SIDE

        ### IMAGE AND THRESHOLD TAB
        ## IMAGE GROUPBOX
        self.imageFileBrowsePushButton.clicked.connect(self.choose_image_file)

        ## THRESHOLD GROUPBOX
        # we hide for now the threshold options, might like them later
        self.subpixelCheckBox.setVisible(False)
        self.smoothingCheckBox.setVisible(False)
        self.SmoothingDistanceLabel.setVisible(False)
        self.smoothingDistanceSpinBox.setVisible(False)

        ### MEASUREMENT TAB
        ## GUESS + FIT
        self.analysisTabs.setTabEnabled(1, False)
        self.pixelDensitySpinBox.editingFinished.connect(self.harmonizePixelSize)
        self.pixelSizeSpinbox.editingFinished.connect(self.harmonizePixelDensity)
        self.autoGuessPushButton.clicked.connect(self.guessParameters)


        self.anglegSpinBox.valueChanged.connect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.connect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.connect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.connect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.connect(self.caplength_manualchange)

        self.parameters = [0., 0., 0., 0., 2.7]
        self.applyParameters()

        self.optimizePushButton.clicked.connect(self.optimizeParameters)

        
    def choose_image_file(self):
        
        dialog = QFileDialog(self)
        
        dialog.setFileMode(QFileDialog.ExistingFile) # single file
        dialog.setViewMode(QFileDialog.Detail)
        ## ADD other images types...
        dialog.setNameFilter("Images (*.png *.tif *.tiff *.jpg *.jpeg)")
        
        if dialog.exec():
            fileName = dialog.selectedFiles()[0]
            print(fileName)
            self.imageFileLineEdit.setText(fileName)
            
            if self.plotWidget.load_image(filepath=fileName):
                self.displayStackedWidget.setCurrentIndex(1)
                self.analysisTabs.setTabEnabled(1, True)

    ### WORK ON IMAGE

    def ROIMoved(self):
        ROIpos = self.plotWidget.roi.pos()
        ROIsize = self.plotWidget.roi.size()
        RL = [int(ROIpos[0]), int(ROIpos[1])]
        BR = [int(ROIpos[0] + ROIsize[0]), int(ROIpos[1] + ROIsize[1])]
        if self.ROI_TL_x_spinBox.value() != RL[0]:
            self.ROI_TL_x_spinBox.setValue(RL[0])
        if self.ROI_TL_y_spinBox.value() != RL[1]:
            self.ROI_TL_y_spinBox.setValue(RL[1])
        if self.ROI_BR_x_spinBox.value() != BR[0]:
            self.ROI_BR_x_spinBox.setValue(BR[0])
        if self.ROI_BR_y_spinBox.value() != BR[1]:
            self.ROI_BR_y_spinBox.setValue(BR[1])
        self.plotWidget.iso.setROI(ROIpos, ROIsize)
    
    def ROIChanged(self):
        ROI_TL_x = self.ROI_TL_x_spinBox.value()
        ROI_TL_y = self.ROI_TL_y_spinBox.value()
        ROI_BR_x = self.ROI_BR_x_spinBox.value()
        ROI_BR_y = self.ROI_BR_y_spinBox.value()
        ROIpos = (ROI_TL_x, ROI_TL_y)
        ROIsize = (ROI_BR_x - ROI_TL_x, ROI_BR_y - ROI_TL_y)
        self.plotWidget.roi.setPos(ROIpos)
        self.plotWidget.roi.setSize(ROIsize)

        self.plotWidget.iso.setROI(ROIpos, ROIsize)
        # self.imgShowWidget.iso.offset = np.array([ROIpos[0], ROIpos[1]])
        # self.imgShowWidget.iso.setData(self.imgShowWidget.roi.getArrayRegion(self.imgShowWidget.data, self.imgShowWidget.imageItem))
        
    def thresholdMoved(self):
        level = self.plotWidget.isoCtrlLine.value()
        if self.customThresholdSpinBox.value() != level:
            self.customThresholdSpinBox.setValue(level)
        
    def thresholdChanged(self, level = 127):
        if self.plotWidget.isoCtrlLine.value() != level:
            self.plotWidget.isoCtrlLine.setValue(level)
        
        self.plotWidget.iso.setLevel(level)

    ### ESTIMATE PARAMETERS

    def harmonizePixelDensity(self, pixelSize:Optional[float]=None):
        if pixelSize is None:
            pixelSize = self.pixelSizeSpinbox.value()
        self.pixelDensitySpinBox.editingFinished.disconnect(self.harmonizePixelSize)
        if pixelSize > 0:
            self.pixelDensitySpinBox.setValue(1/pixelSize)
            if not self.autoGuessPushButton.isEnabled():
                self.autoGuessPushButton.setEnabled(True)
        else:
            self.pixelDensitySpinBox.setValue(0)
            if not self.autoGuessPushButton.isEnabled():
                self.autoGuessPushButton.setEnabled(False)
        self.pixelDensitySpinBox.editingFinished.connect(self.harmonizePixelSize)

    def harmonizePixelSize(self, pixelDensity:float=None):
        if pixelDensity is None:
            pixelDensity = self.pixelDensitySpinBox.value()
        self.pixelSizeSpinbox.editingFinished.disconnect(self.harmonizePixelDensity)
        if pixelDensity > 0:
            self.pixelSizeSpinbox.setValue(1/pixelDensity)
            if not self.autoGuessPushButton.isEnabled():
                self.autoGuessPushButton.setEnabled(True)
        else:
            self.pixelSizeSpinbox.setValue(0)
            if not self.autoGuessPushButton.isEnabled():
                self.autoGuessPushButton.setEnabled(False)
        self.pixelSizeSpinbox.editingFinished.connect(self.harmonizePixelDensity)

    def angleg_manualchange(self, angleg:Optional[float]=None):
        if angleg is None:
            angleg = self.anglegSpinBox.value()
        self.parameters[0] = angleg * RAD_PER_DEG
        self.actualizeComputedCurve()

    def tipx_manualchange(self, tipx:Optional[float]=None):
        if tipx is None:
            tipx = self.tipxSpinBox.value()
        self.parameters[2] = tipx
        self.actualizeComputedCurve()

    def tipy_manualchange(self, tipy:Optional[float]=None):
        if tipy is None:
            tipy = self.tipySpinBox.value()
        self.parameters[1] = tipy
        self.actualizeComputedCurve()

    def r0_manualchange(self, r0:Optional[float]=None):
        if r0 is None:
            r0 = self.r0SpinBox.value()
        self.parameters[3] = r0
        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def caplength_manualchange(self, caplength:Optional[float]=None):
        if caplength is None:
            caplength = self.caplengthSpinBox.value()
        self.parameters[4] = caplength
        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def applyParameters(self):
        gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = self.parameters

        self.anglegSpinBox.valueChanged.disconnect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.disconnect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.disconnect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.disconnect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.disconnect(self.caplength_manualchange)

        self.anglegSpinBox.setValue(gravity_angle * DEG_PER_RAD)
        self.tipxSpinBox.setValue(x_tip_position)
        self.tipySpinBox.setValue(y_tip_position)
        self.r0SpinBox.setValue(r0_mm)
        self.caplengthSpinBox.setValue(capillary_length_mm)

        self.anglegSpinBox.valueChanged.connect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.connect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.connect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.connect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.connect(self.caplength_manualchange)

        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def guessParameters(self):
        px_per_mm = self.pixelDensitySpinBox.value()
        # print('DEBUG GUESS PARAMETERS')
        threshold = self.customThresholdSpinBox.value()

        mainContour = self.plotWidget.isoCurve_level(level=threshold)
        # print('DEBUG:', f'Contour shape: {mainContour.shape} (expect (2, N))')

        image_centre = analyze.image_centre(np.array(self.plotWidget.iso.data))
        # print('DEBUG:', f'image centre (rotation centre): {image_centre}')

        self.parameters = analyze.estimate_parameters(image_centre, mainContour, px_per_mm=px_per_mm)

        analyze.talk_params(self.parameters, px_per_mm=px_per_mm)

        self.applyParameters()

    ### OPTIMIZE PARAMETERS

    def areParametersValid(self) -> bool:
        " Coucou "
        canDoOptimization = not( (self.pixelDensitySpinBox.value() == 0) or np.prod(np.array(self.parameters[3:])==0) )

        self.optimizePushButton.setEnabled(canDoOptimization)
        if not(canDoOptimization):
            self.gammaSpinBox.setValue(0)
            self.bondSpinBox.setValue(0)
        return canDoOptimization

    def optimizeParameters(self):
        if self.areParametersValid():

            px_per_mm = self.pixelDensitySpinBox.value()
            # print('DEBUG OPTIMIZE PARAMETERS')
            threshold = self.customThresholdSpinBox.value()
            mainContour = self.plotWidget.isoCurve_level(level=threshold)
            # print('DEBUG:', f'Contour shape: {mainContour.shape} (expect (2, N))')

            to_fit=[self.anglegCheckBox.isChecked(),
                    self.tipyCheckBox.isChecked(),
                    self.tipxCheckBox.isChecked(),
                    self.r0CheckBox.isChecked(),
                    self.caplengthCheckBox.isChecked()]

            print('DEBUG:', f'To fit: {to_fit}')

            opti_success, self.parameters = analyze.optimize_profile(mainContour, px_per_mm=px_per_mm,
                                                       parameters_initialguess=self.parameters, to_fit=to_fit)

            analyze.talk_params(self.parameters, px_per_mm=px_per_mm)

            self.applyParameters()

    def actualizeComputedCurve(self):
        if self.areParametersValid():
            px_per_mm = self.pixelDensitySpinBox.value()
            R, Z = analyze.integrated_contour(px_per_mm, self.parameters)

            self.plotWidget.plot_computed_profile(R, Z)

    ### PHYSICS

    def actualizeSurfaceTension(self):
        if self.areParametersValid():
            r0_mm = self.parameters[3]
            caplength_mm = self.parameters[4]
            self.surface_tension = self.rhogSpinBox.value() * caplength_mm**2
            self.bond = (r0_mm / caplength_mm)**2
            self.gammaSpinBox.setValue(self.surface_tension)
            self.bondSpinBox.setValue(self.bond)



        



