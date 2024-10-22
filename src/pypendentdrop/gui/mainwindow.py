# Main window
from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np

# from pyqtgraph.Qt.QtGui import QPixmap
from pyqtgraph.Qt.QtWidgets import QMainWindow, QFileDialog

from ... import pypendentdrop as ppd

from .mainwindow_ui import Ui_PPD_MainWindow
from .plotwidget import ppd_plotWidget


RAD_PER_DEG = np.pi/180
DEG_PER_RAD = 180/np.pi

class ppd_mainwindow(QMainWindow, Ui_PPD_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self) # we benefit of the double

        ### LEFT SIDE

        # # The widget #0 of displayStackedWidget is the welcome image illustration
        # pixmap = QPixmap('assets/images/ppd_illustration.png')
        # self.welcomeIcon.setPixmap(pixmap)

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
        self.pixelDensitySpinBox.editingFinished.connect(self.harmonizePixelSpacing)
        self.pixelSizeSpinbox.editingFinished.connect(self.harmonizePixelDensity)
        self.autoGuessPushButton.clicked.connect(self.guessParameters)


        self.anglegSpinBox.valueChanged.connect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.connect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.connect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.connect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.connect(self.caplength_manualchange)

        self.parameters:ppd.Parameters = ppd.Parameters()
        self.applyParameters()

        self.rhogSpinBox.valueChanged.connect(self.rhog_manualchange)
        self.rhog_manualchange()

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

    def harmonizePixelDensity(self, pixelSpacing:Optional[float]=None):
        if pixelSpacing is None:
            pixelSpacing = self.pixelSizeSpinbox.value()
        ppd.debug(f'harmonizePixelDensity with spacing={pixelSpacing} px/mm')
        self.pixelDensitySpinBox.editingFinished.disconnect(self.harmonizePixelSpacing)

        self.parameters.set_px_spacing(pixelSpacing)
        self.pixelDensitySpinBox.setValue(self.parameters.get_px_density() or 0)

        self.autoGuessPushButton.setEnabled(self.parameters.can_estimate())

        self.pixelDensitySpinBox.editingFinished.connect(self.harmonizePixelSpacing)

    def harmonizePixelSpacing(self, pixelDensity:float=None):
        if pixelDensity is None:
            pixelDensity = self.pixelDensitySpinBox.value()
        ppd.debug(f'harmonizePixelSpacing with density={pixelDensity} px/mm')
        self.pixelSizeSpinbox.editingFinished.disconnect(self.harmonizePixelDensity)

        self.parameters.set_px_density(pixelDensity)
        self.pixelSizeSpinbox.setValue(self.parameters.get_px_spacing() or 0)

        self.autoGuessPushButton.setEnabled(self.parameters.can_estimate())

        self.pixelSizeSpinbox.editingFinished.connect(self.harmonizePixelDensity)

    def angleg_manualchange(self, angleg:Optional[float]=None):
        if angleg is None:
            angleg = self.anglegSpinBox.value()
        self.parameters.set_a_deg(angleg)
        self.actualizeComputedCurve()

    def tipx_manualchange(self, tipx:Optional[float]=None):
        if tipx is None:
            tipx = self.tipxSpinBox.value()
        self.parameters.set_x_px(tipx)
        self.actualizeComputedCurve()

    def tipy_manualchange(self, tipy:Optional[float]=None):
        if tipy is None:
            tipy = self.tipySpinBox.value()
        self.parameters.set_y_px(tipy)
        self.actualizeComputedCurve()

    def r0_manualchange(self, r0:Optional[float]=None):
        if r0 is None:
            r0 = self.r0SpinBox.value()
        self.parameters.set_r_mm(r0)
        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def caplength_manualchange(self, caplength:Optional[float]=None):
        if caplength is None:
            caplength = self.caplengthSpinBox.value()
        self.parameters.set_l_mm(caplength)
        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def applyParameters(self):
        self.anglegSpinBox.valueChanged.disconnect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.disconnect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.disconnect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.disconnect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.disconnect(self.caplength_manualchange)

        self.anglegSpinBox.setValue(self.parameters.get_a_deg() or 0)
        self.tipxSpinBox.setValue(self.parameters.get_x_px() or 0)
        self.tipySpinBox.setValue(self.parameters.get_y_px() or 0)
        self.r0SpinBox.setValue(self.parameters.get_r_mm() or 0)
        self.caplengthSpinBox.setValue(self.parameters.get_l_mm() or 0)

        self.anglegSpinBox.valueChanged.connect(self.angleg_manualchange)
        self.tipxSpinBox.valueChanged.connect(self.tipx_manualchange)
        self.tipySpinBox.valueChanged.connect(self.tipy_manualchange)
        self.r0SpinBox.valueChanged.connect(self.r0_manualchange)
        self.caplengthSpinBox.valueChanged.connect(self.caplength_manualchange)

        self.actualizeComputedCurve()
        self.actualizeSurfaceTension()

    def guessParameters(self):
        if self.parameters.can_estimate():
            px_per_mm = self.parameters.get_px_density()
            threshold = self.customThresholdSpinBox.value()

            mainContour = self.plotWidget.isoCurve_level(level=threshold)
            # print('DEBUG:', f'Contour shape: {mainContour.shape} (expect (2, N))')

            imagecentre = ppd.image_centre(np.array(self.plotWidget.iso.data))
            # print('DEBUG:', f'image centre (rotation centre): {imagecentre}')

            self.parameters = ppd.estimate_parameters(imagecentre, mainContour, px_per_mm=px_per_mm)
            self.rhog_manualchange()

            self.parameters.describe(printfn=ppd.info, name='estimated')

            self.applyParameters()

    ### OPTIMIZE PARAMETERS

    def canComputeProfile(self) -> bool:
        canDoOptimization = self.parameters.can_optimize()

        self.optimizePushButton.setEnabled(canDoOptimization)
        if not(canDoOptimization):
            self.gammaSpinBox.setValue(0)
            self.bondSpinBox.setValue(0)
        return canDoOptimization

    def optimizeParameters(self):
        if self.canComputeProfile():

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

            opti_success, self.parameters = ppd.optimize_profile(mainContour, parameters_initialguess=self.parameters, to_fit=to_fit)

            self.parameters.describe(printfn=ppd.info, name='optimized')

            self.applyParameters()
            self.actualizeSurfaceTension()

    def actualizeComputedCurve(self):
        if self.parameters.can_show_tip_position():
            self.plotWidget.scatter_droptip(self.parameters.get_xy_px())
        else:
            self.plotWidget.hide_scatter_droptip()
        if self.canComputeProfile():
            R, Z = ppd.integrated_contour(self.parameters)

            self.plotWidget.plot_computed_profile(R, Z)
        else:
            self.plotWidget.hide_computed_profile()

    ### PHYSICS
    def rhog_manualchange(self, rhog:Optional[float]=None):
        if rhog is None:
            rhog = self.rhogSpinBox.value()
        self.parameters.set_densitycontrast(rhog)
        self.actualizeSurfaceTension()

    def actualizeSurfaceTension(self):
        if self.canComputeProfile():
            self.gammaSpinBox.setValue(self.parameters.get_surface_tension() or 0)
            self.bondSpinBox.setValue(self.parameters.get_bond() or 0)



        



