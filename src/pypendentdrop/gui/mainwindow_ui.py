# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ppd_mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

# + : Replaced PySide6 or PyQt6 by pyqtgraph.Qt

from pyqtgraph.Qt.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from pyqtgraph.Qt.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from pyqtgraph.Qt.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QStackedWidget, QTabWidget, QVBoxLayout,
    QWidget)

class Ui_PPD_MainWindow(object):
    def setupUi(self, PPD_MainWindow):
        if not PPD_MainWindow.objectName():
            PPD_MainWindow.setObjectName(u"PPD_MainWindow")
        PPD_MainWindow.resize(1619, 894)
        self.centralwidget = QWidget(PPD_MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.displayStackedWidget = QStackedWidget(self.centralwidget)
        self.displayStackedWidget.setObjectName(u"displayStackedWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.displayStackedWidget.sizePolicy().hasHeightForWidth())
        self.displayStackedWidget.setSizePolicy(sizePolicy)
        self.display_welcomePage = QWidget()
        self.display_welcomePage.setObjectName(u"display_welcomePage")
        sizePolicy.setHeightForWidth(self.display_welcomePage.sizePolicy().hasHeightForWidth())
        self.display_welcomePage.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.display_welcomePage)
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.welcomeIcon = QLabel(self.display_welcomePage)
        self.welcomeIcon.setObjectName(u"welcomeIcon")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.welcomeIcon.sizePolicy().hasHeightForWidth())
        self.welcomeIcon.setSizePolicy(sizePolicy1)
        self.welcomeIcon.setScaledContents(True)
        self.welcomeIcon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_5.addWidget(self.welcomeIcon)

        self.displayStackedWidget.addWidget(self.display_welcomePage)

        self.horizontalLayout.addWidget(self.displayStackedWidget)

        self.analysisTabs = QTabWidget(self.centralwidget)
        self.analysisTabs.setObjectName(u"analysisTabs")
        self.tab_imagecontour = QWidget()
        self.tab_imagecontour.setObjectName(u"tab_imagecontour")
        self.verticalLayout_4 = QVBoxLayout(self.tab_imagecontour)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.imageSelectionGroupBox = QGroupBox(self.tab_imagecontour)
        self.imageSelectionGroupBox.setObjectName(u"imageSelectionGroupBox")
        self.formLayout = QFormLayout(self.imageSelectionGroupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFormAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.imageFileLabel = QLabel(self.imageSelectionGroupBox)
        self.imageFileLabel.setObjectName(u"imageFileLabel")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.imageFileLabel)

        self.imageFileHLayout = QHBoxLayout()
        self.imageFileHLayout.setObjectName(u"imageFileHLayout")
        self.imageFileLineEdit = QLineEdit(self.imageSelectionGroupBox)
        self.imageFileLineEdit.setObjectName(u"imageFileLineEdit")

        self.imageFileHLayout.addWidget(self.imageFileLineEdit)

        self.imageFileBrowsePushButton = QPushButton(self.imageSelectionGroupBox)
        self.imageFileBrowsePushButton.setObjectName(u"imageFileBrowsePushButton")

        self.imageFileHLayout.addWidget(self.imageFileBrowsePushButton)


        self.formLayout.setLayout(0, QFormLayout.FieldRole, self.imageFileHLayout)

        self.subregionLabel = QLabel(self.imageSelectionGroupBox)
        self.subregionLabel.setObjectName(u"subregionLabel")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.subregionLabel)

        self.subregionGridLayout = QGridLayout()
        self.subregionGridLayout.setObjectName(u"subregionGridLayout")
        self.descr1ROITL = QLabel(self.imageSelectionGroupBox)
        self.descr1ROITL.setObjectName(u"descr1ROITL")
        self.descr1ROITL.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.subregionGridLayout.addWidget(self.descr1ROITL, 0, 0, 1, 1)

        self.ROI_TL_x_spinBox = QSpinBox(self.imageSelectionGroupBox)
        self.ROI_TL_x_spinBox.setObjectName(u"ROI_TL_x_spinBox")
        self.ROI_TL_x_spinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ROI_TL_x_spinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ROI_TL_x_spinBox.setMaximum(9999)

        self.subregionGridLayout.addWidget(self.ROI_TL_x_spinBox, 0, 1, 1, 1)

        self.ROI_TL_y_spinBox = QSpinBox(self.imageSelectionGroupBox)
        self.ROI_TL_y_spinBox.setObjectName(u"ROI_TL_y_spinBox")
        self.ROI_TL_y_spinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ROI_TL_y_spinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ROI_TL_y_spinBox.setMaximum(99999)

        self.subregionGridLayout.addWidget(self.ROI_TL_y_spinBox, 0, 3, 1, 1)

        self.ROI_BR_y_spinBox = QSpinBox(self.imageSelectionGroupBox)
        self.ROI_BR_y_spinBox.setObjectName(u"ROI_BR_y_spinBox")
        self.ROI_BR_y_spinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ROI_BR_y_spinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ROI_BR_y_spinBox.setMaximum(99999)

        self.subregionGridLayout.addWidget(self.ROI_BR_y_spinBox, 1, 3, 1, 1)

        self.ROI_BR_x_spinBox = QSpinBox(self.imageSelectionGroupBox)
        self.ROI_BR_x_spinBox.setObjectName(u"ROI_BR_x_spinBox")
        self.ROI_BR_x_spinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ROI_BR_x_spinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ROI_BR_x_spinBox.setMaximum(9999)

        self.subregionGridLayout.addWidget(self.ROI_BR_x_spinBox, 1, 1, 1, 1)

        self.descr2ROITL = QLabel(self.imageSelectionGroupBox)
        self.descr2ROITL.setObjectName(u"descr2ROITL")

        self.subregionGridLayout.addWidget(self.descr2ROITL, 0, 2, 1, 1)

        self.descr3ROITL = QLabel(self.imageSelectionGroupBox)
        self.descr3ROITL.setObjectName(u"descr3ROITL")
        sizePolicy.setHeightForWidth(self.descr3ROITL.sizePolicy().hasHeightForWidth())
        self.descr3ROITL.setSizePolicy(sizePolicy)

        self.subregionGridLayout.addWidget(self.descr3ROITL, 0, 4, 1, 1)

        self.descr1ROIBR = QLabel(self.imageSelectionGroupBox)
        self.descr1ROIBR.setObjectName(u"descr1ROIBR")
        self.descr1ROIBR.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.subregionGridLayout.addWidget(self.descr1ROIBR, 1, 0, 1, 1)

        self.descr2ROIBR = QLabel(self.imageSelectionGroupBox)
        self.descr2ROIBR.setObjectName(u"descr2ROIBR")

        self.subregionGridLayout.addWidget(self.descr2ROIBR, 1, 2, 1, 1)

        self.descr3ROIBR = QLabel(self.imageSelectionGroupBox)
        self.descr3ROIBR.setObjectName(u"descr3ROIBR")

        self.subregionGridLayout.addWidget(self.descr3ROIBR, 1, 4, 1, 1)


        self.formLayout.setLayout(1, QFormLayout.FieldRole, self.subregionGridLayout)


        self.verticalLayout_4.addWidget(self.imageSelectionGroupBox)

        self.contourGroupBox = QGroupBox(self.tab_imagecontour)
        self.contourGroupBox.setObjectName(u"contourGroupBox")
        self.formLayout_2 = QFormLayout(self.contourGroupBox)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.autoThresholdCheckBox = QCheckBox(self.contourGroupBox)
        self.autoThresholdCheckBox.setObjectName(u"autoThresholdCheckBox")
        self.autoThresholdCheckBox.setChecked(True)

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.autoThresholdCheckBox)

        self.thresholdHLayout = QHBoxLayout()
        self.thresholdHLayout.setObjectName(u"thresholdHLayout")
        self.customThresholdLabel = QLabel(self.contourGroupBox)
        self.customThresholdLabel.setObjectName(u"customThresholdLabel")
        self.customThresholdLabel.setEnabled(False)

        self.thresholdHLayout.addWidget(self.customThresholdLabel)

        self.customThresholdSpinBox = QSpinBox(self.contourGroupBox)
        self.customThresholdSpinBox.setObjectName(u"customThresholdSpinBox")
        self.customThresholdSpinBox.setEnabled(False)
        self.customThresholdSpinBox.setMaximum(255)
        self.customThresholdSpinBox.setValue(127)

        self.thresholdHLayout.addWidget(self.customThresholdSpinBox)


        self.formLayout_2.setLayout(0, QFormLayout.FieldRole, self.thresholdHLayout)

        self.subpixelCheckBox = QCheckBox(self.contourGroupBox)
        self.subpixelCheckBox.setObjectName(u"subpixelCheckBox")
        self.subpixelCheckBox.setChecked(True)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.subpixelCheckBox)

        self.smoothingCheckBox = QCheckBox(self.contourGroupBox)
        self.smoothingCheckBox.setObjectName(u"smoothingCheckBox")
        self.smoothingCheckBox.setChecked(True)

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.smoothingCheckBox)

        self.smoothingHLayout = QHBoxLayout()
        self.smoothingHLayout.setObjectName(u"smoothingHLayout")
        self.SmoothingDistanceLabel = QLabel(self.contourGroupBox)
        self.SmoothingDistanceLabel.setObjectName(u"SmoothingDistanceLabel")

        self.smoothingHLayout.addWidget(self.SmoothingDistanceLabel)

        self.smoothingDistanceSpinBox = QSpinBox(self.contourGroupBox)
        self.smoothingDistanceSpinBox.setObjectName(u"smoothingDistanceSpinBox")
        self.smoothingDistanceSpinBox.setMinimum(1)
        self.smoothingDistanceSpinBox.setMaximum(999)
        self.smoothingDistanceSpinBox.setSingleStep(2)
        self.smoothingDistanceSpinBox.setValue(11)

        self.smoothingHLayout.addWidget(self.smoothingDistanceSpinBox)


        self.formLayout_2.setLayout(2, QFormLayout.FieldRole, self.smoothingHLayout)


        self.verticalLayout_4.addWidget(self.contourGroupBox)

        self.analysisTabs.addTab(self.tab_imagecontour, "")
        self.tab_measure = QWidget()
        self.tab_measure.setObjectName(u"tab_measure")
        self.verticalLayout = QVBoxLayout(self.tab_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.parametersGroupBox = QGroupBox(self.tab_measure)
        self.parametersGroupBox.setObjectName(u"parametersGroupBox")
        self.verticalLayout_3 = QVBoxLayout(self.parametersGroupBox)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.pixelSizeLabel = QLabel(self.parametersGroupBox)
        self.pixelSizeLabel.setObjectName(u"pixelSizeLabel")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.pixelSizeLabel)

        self.pixelSizeSpinbox = QDoubleSpinBox(self.parametersGroupBox)
        self.pixelSizeSpinbox.setObjectName(u"pixelSizeSpinbox")
        self.pixelSizeSpinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.pixelSizeSpinbox.setDecimals(5)
        self.pixelSizeSpinbox.setMaximum(99999.000000000000000)
        self.pixelSizeSpinbox.setValue(0.000000000000000)

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.pixelSizeSpinbox)

        self.pixelDensityLabel = QLabel(self.parametersGroupBox)
        self.pixelDensityLabel.setObjectName(u"pixelDensityLabel")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.pixelDensityLabel)

        self.pixelDensitySpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.pixelDensitySpinBox.setObjectName(u"pixelDensitySpinBox")
        self.pixelDensitySpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.pixelDensitySpinBox.setDecimals(3)
        self.pixelDensitySpinBox.setMaximum(999999.000000000000000)
        self.pixelDensitySpinBox.setValue(0.000000000000000)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.pixelDensitySpinBox)


        self.verticalLayout_3.addLayout(self.formLayout_3)

        self.parametersLine = QFrame(self.parametersGroupBox)
        self.parametersLine.setObjectName(u"parametersLine")
        self.parametersLine.setFrameShape(QFrame.Shape.HLine)
        self.parametersLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_3.addWidget(self.parametersLine)

        self.parametersSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.parametersSpacer)

        self.autoGuessPushButton = QPushButton(self.parametersGroupBox)
        self.autoGuessPushButton.setObjectName(u"autoGuessPushButton")
        self.autoGuessPushButton.setEnabled(False)

        self.verticalLayout_3.addWidget(self.autoGuessPushButton)

        self.parametersGridLayout = QGridLayout()
        self.parametersGridLayout.setObjectName(u"parametersGridLayout")
        self.caplengthLabel = QLabel(self.parametersGroupBox)
        self.caplengthLabel.setObjectName(u"caplengthLabel")

        self.parametersGridLayout.addWidget(self.caplengthLabel, 4, 0, 1, 1)

        self.tipxCheckBox = QCheckBox(self.parametersGroupBox)
        self.tipxCheckBox.setObjectName(u"tipxCheckBox")
        self.tipxCheckBox.setChecked(True)

        self.parametersGridLayout.addWidget(self.tipxCheckBox, 1, 2, 1, 1)

        self.tipxLabel = QLabel(self.parametersGroupBox)
        self.tipxLabel.setObjectName(u"tipxLabel")

        self.parametersGridLayout.addWidget(self.tipxLabel, 1, 0, 1, 1)

        self.anglegSpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.anglegSpinBox.setObjectName(u"anglegSpinBox")
        self.anglegSpinBox.setDecimals(3)
        self.anglegSpinBox.setMinimum(-180.000000000000000)
        self.anglegSpinBox.setMaximum(180.000000000000000)
        self.anglegSpinBox.setSingleStep(0.500000000000000)
        self.anglegSpinBox.setValue(-180.000000000000000)

        self.parametersGridLayout.addWidget(self.anglegSpinBox, 0, 1, 1, 1)

        self.tipyLabel = QLabel(self.parametersGroupBox)
        self.tipyLabel.setObjectName(u"tipyLabel")

        self.parametersGridLayout.addWidget(self.tipyLabel, 2, 0, 1, 1)

        self.anglegCheckBox = QCheckBox(self.parametersGroupBox)
        self.anglegCheckBox.setObjectName(u"anglegCheckBox")
        self.anglegCheckBox.setChecked(True)

        self.parametersGridLayout.addWidget(self.anglegCheckBox, 0, 2, 1, 1)

        self.tipxSpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.tipxSpinBox.setObjectName(u"tipxSpinBox")
        self.tipxSpinBox.setDecimals(3)
        self.tipxSpinBox.setMaximum(99999.000000000000000)

        self.parametersGridLayout.addWidget(self.tipxSpinBox, 1, 1, 1, 1)

        self.tipySpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.tipySpinBox.setObjectName(u"tipySpinBox")
        self.tipySpinBox.setDecimals(3)
        self.tipySpinBox.setMaximum(99999.000000000000000)

        self.parametersGridLayout.addWidget(self.tipySpinBox, 2, 1, 1, 1)

        self.anglegLabel = QLabel(self.parametersGroupBox)
        self.anglegLabel.setObjectName(u"anglegLabel")

        self.parametersGridLayout.addWidget(self.anglegLabel, 0, 0, 1, 1)

        self.tipyCheckBox = QCheckBox(self.parametersGroupBox)
        self.tipyCheckBox.setObjectName(u"tipyCheckBox")
        self.tipyCheckBox.setChecked(True)

        self.parametersGridLayout.addWidget(self.tipyCheckBox, 2, 2, 1, 1)

        self.r0Label = QLabel(self.parametersGroupBox)
        self.r0Label.setObjectName(u"r0Label")

        self.parametersGridLayout.addWidget(self.r0Label, 3, 0, 1, 1)

        self.r0SpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.r0SpinBox.setObjectName(u"r0SpinBox")
        self.r0SpinBox.setDecimals(5)
        self.r0SpinBox.setMaximum(99999.000000000000000)
        self.r0SpinBox.setSingleStep(0.100000000000000)
        self.r0SpinBox.setValue(0.000000000000000)

        self.parametersGridLayout.addWidget(self.r0SpinBox, 3, 1, 1, 1)

        self.caplengthSpinBox = QDoubleSpinBox(self.parametersGroupBox)
        self.caplengthSpinBox.setObjectName(u"caplengthSpinBox")
        self.caplengthSpinBox.setDecimals(5)
        self.caplengthSpinBox.setMaximum(99999.000000000000000)
        self.caplengthSpinBox.setSingleStep(0.100000000000000)
        self.caplengthSpinBox.setValue(0.000000000000000)

        self.parametersGridLayout.addWidget(self.caplengthSpinBox, 4, 1, 1, 1)

        self.r0CheckBox = QCheckBox(self.parametersGroupBox)
        self.r0CheckBox.setObjectName(u"r0CheckBox")
        self.r0CheckBox.setChecked(True)

        self.parametersGridLayout.addWidget(self.r0CheckBox, 3, 2, 1, 1)

        self.caplengthCheckBox = QCheckBox(self.parametersGroupBox)
        self.caplengthCheckBox.setObjectName(u"caplengthCheckBox")
        self.caplengthCheckBox.setChecked(True)

        self.parametersGridLayout.addWidget(self.caplengthCheckBox, 4, 2, 1, 1)


        self.verticalLayout_3.addLayout(self.parametersGridLayout)

        self.optimizePushButton = QPushButton(self.parametersGroupBox)
        self.optimizePushButton.setObjectName(u"optimizePushButton")
        self.optimizePushButton.setEnabled(False)

        self.verticalLayout_3.addWidget(self.optimizePushButton)


        self.verticalLayout.addWidget(self.parametersGroupBox)

        self.physicsGroupBox = QGroupBox(self.tab_measure)
        self.physicsGroupBox.setObjectName(u"physicsGroupBox")
        self.verticalLayout_2 = QVBoxLayout(self.physicsGroupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.rhogHLayout = QHBoxLayout()
        self.rhogHLayout.setObjectName(u"rhogHLayout")
        self.rhogLabel = QLabel(self.physicsGroupBox)
        self.rhogLabel.setObjectName(u"rhogLabel")

        self.rhogHLayout.addWidget(self.rhogLabel)

        self.rhogSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.rhogSpinBox.setObjectName(u"rhogSpinBox")
        self.rhogSpinBox.setBaseSize(QSize(200, 30))
        self.rhogSpinBox.setReadOnly(False)
        self.rhogSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rhogSpinBox.setDecimals(3)
        self.rhogSpinBox.setMaximum(99999.000000000000000)
        self.rhogSpinBox.setValue(9.805999999999999)

        self.rhogHLayout.addWidget(self.rhogSpinBox)


        self.verticalLayout_2.addLayout(self.rhogHLayout)

        self.measureLine = QFrame(self.physicsGroupBox)
        self.measureLine.setObjectName(u"measureLine")
        self.measureLine.setFrameShape(QFrame.Shape.HLine)
        self.measureLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_2.addWidget(self.measureLine)

        self.measureSpacer = QSpacerItem(20, 139, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.measureSpacer)

        self.variablesFormLayout = QFormLayout()
        self.variablesFormLayout.setObjectName(u"variablesFormLayout")
        self.variablesFormLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.variablesFormLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.variablesFormLayout.setFormAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.gammaLabel = QLabel(self.physicsGroupBox)
        self.gammaLabel.setObjectName(u"gammaLabel")

        self.variablesFormLayout.setWidget(0, QFormLayout.LabelRole, self.gammaLabel)

        self.gammaSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.gammaSpinBox.setObjectName(u"gammaSpinBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.gammaSpinBox.sizePolicy().hasHeightForWidth())
        self.gammaSpinBox.setSizePolicy(sizePolicy2)
        self.gammaSpinBox.setBaseSize(QSize(200, 30))
        self.gammaSpinBox.setReadOnly(True)
        self.gammaSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.gammaSpinBox.setDecimals(3)
        self.gammaSpinBox.setMinimum(0.000000000000000)
        self.gammaSpinBox.setMaximum(99999.990000000005239)
        self.gammaSpinBox.setValue(0.000000000000000)

        self.variablesFormLayout.setWidget(0, QFormLayout.FieldRole, self.gammaSpinBox)

        self.bondLabel = QLabel(self.physicsGroupBox)
        self.bondLabel.setObjectName(u"bondLabel")

        self.variablesFormLayout.setWidget(1, QFormLayout.LabelRole, self.bondLabel)

        self.bondSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.bondSpinBox.setObjectName(u"bondSpinBox")
        self.bondSpinBox.setBaseSize(QSize(200, 30))
        self.bondSpinBox.setReadOnly(True)
        self.bondSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.bondSpinBox.setDecimals(3)
        self.bondSpinBox.setMaximum(99999.990000000005239)
        self.bondSpinBox.setValue(0.000000000000000)

        self.variablesFormLayout.setWidget(1, QFormLayout.FieldRole, self.bondSpinBox)


        self.verticalLayout_2.addLayout(self.variablesFormLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.formLayout_4 = QFormLayout()
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.worthingtonLabel = QLabel(self.physicsGroupBox)
        self.worthingtonLabel.setObjectName(u"worthingtonLabel")

        self.formLayout_4.setWidget(3, QFormLayout.LabelRole, self.worthingtonLabel)

        self.worthingtonSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.worthingtonSpinBox.setObjectName(u"worthingtonSpinBox")
        self.worthingtonSpinBox.setReadOnly(True)
        self.worthingtonSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.worthingtonSpinBox.setDecimals(3)
        self.worthingtonSpinBox.setMaximum(99999.990000000005239)
        self.worthingtonSpinBox.setValue(0.000000000000000)

        self.formLayout_4.setWidget(3, QFormLayout.FieldRole, self.worthingtonSpinBox)

        self.dropVolumeLabel = QLabel(self.physicsGroupBox)
        self.dropVolumeLabel.setObjectName(u"dropVolumeLabel")

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.dropVolumeLabel)

        self.dropVolumeSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.dropVolumeSpinBox.setObjectName(u"dropVolumeSpinBox")
        self.dropVolumeSpinBox.setReadOnly(True)
        self.dropVolumeSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.dropVolumeSpinBox.setDecimals(3)
        self.dropVolumeSpinBox.setMaximum(1000000.000000000000000)

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.dropVolumeSpinBox)

        self.needleDiameterSpinBox = QDoubleSpinBox(self.physicsGroupBox)
        self.needleDiameterSpinBox.setObjectName(u"needleDiameterSpinBox")
        self.needleDiameterSpinBox.setReadOnly(True)
        self.needleDiameterSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.needleDiameterSpinBox.setDecimals(3)
        self.needleDiameterSpinBox.setMaximum(1000000.000000000000000)

        self.formLayout_4.setWidget(2, QFormLayout.FieldRole, self.needleDiameterSpinBox)

        self.needleDiameterLabel = QLabel(self.physicsGroupBox)
        self.needleDiameterLabel.setObjectName(u"needleDiameterLabel")

        self.formLayout_4.setWidget(2, QFormLayout.LabelRole, self.needleDiameterLabel)


        self.verticalLayout_2.addLayout(self.formLayout_4)


        self.verticalLayout.addWidget(self.physicsGroupBox)

        self.analysisTabs.addTab(self.tab_measure, "")

        self.horizontalLayout.addWidget(self.analysisTabs)

        PPD_MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(PPD_MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1619, 23))
        PPD_MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(PPD_MainWindow)
        self.autoThresholdCheckBox.toggled.connect(self.customThresholdLabel.setDisabled)
        self.smoothingCheckBox.toggled.connect(self.smoothingDistanceSpinBox.setEnabled)
        self.smoothingCheckBox.toggled.connect(self.SmoothingDistanceLabel.setEnabled)
        self.autoThresholdCheckBox.toggled.connect(self.customThresholdSpinBox.setDisabled)

        self.analysisTabs.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(PPD_MainWindow)
    # setupUi

    def retranslateUi(self, PPD_MainWindow):
        PPD_MainWindow.setWindowTitle(QCoreApplication.translate("PPD_MainWindow", u"MainWindow", None))
        self.welcomeIcon.setText(QCoreApplication.translate("PPD_MainWindow", u"Py Pendent Drop", None))
        self.imageSelectionGroupBox.setTitle(QCoreApplication.translate("PPD_MainWindow", u"Image selection", None))
        self.imageFileLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Image file", None))
        self.imageFileBrowsePushButton.setText(QCoreApplication.translate("PPD_MainWindow", u"Browse", None))
        self.subregionLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Subregion of interest", None))
        self.descr1ROITL.setText(QCoreApplication.translate("PPD_MainWindow", u"TL corner: (", None))
        self.descr2ROITL.setText(QCoreApplication.translate("PPD_MainWindow", u", ", None))
        self.descr3ROITL.setText(QCoreApplication.translate("PPD_MainWindow", u")", None))
        self.descr1ROIBR.setText(QCoreApplication.translate("PPD_MainWindow", u"BR corner: (", None))
        self.descr2ROIBR.setText(QCoreApplication.translate("PPD_MainWindow", u", ", None))
        self.descr3ROIBR.setText(QCoreApplication.translate("PPD_MainWindow", u")", None))
        self.contourGroupBox.setTitle(QCoreApplication.translate("PPD_MainWindow", u"Drop contour detection", None))
#if QT_CONFIG(tooltip)
        self.autoThresholdCheckBox.setToolTip(QCoreApplication.translate("PPD_MainWindow", u"Choose automatically the binarization threshold, using Otsu' s method.", None))
#endif // QT_CONFIG(tooltip)
        self.autoThresholdCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Auto threshold", None))
#if QT_CONFIG(tooltip)
        self.customThresholdLabel.setToolTip(QCoreApplication.translate("PPD_MainWindow", u"The binarization threshold.", None))
#endif // QT_CONFIG(tooltip)
        self.customThresholdLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Threshold", None))
#if QT_CONFIG(tooltip)
        self.subpixelCheckBox.setToolTip(QCoreApplication.translate("PPD_MainWindow", u"Subpixel edge extraction, adapted from R. Ngiam.", None))
#endif // QT_CONFIG(tooltip)
        self.subpixelCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Use subpixel refinement", None))
#if QT_CONFIG(tooltip)
        self.smoothingCheckBox.setToolTip(QCoreApplication.translate("PPD_MainWindow", u"Smooth the obtained profile to remove nonphysical high frequency spikes.", None))
#endif // QT_CONFIG(tooltip)
        self.smoothingCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Smooth profile", None))
#if QT_CONFIG(tooltip)
        self.SmoothingDistanceLabel.setToolTip(QCoreApplication.translate("PPD_MainWindow", u"The typical distance of which the contour is smoothed.", None))
#endif // QT_CONFIG(tooltip)
        self.SmoothingDistanceLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Smoothing distance", None))
        self.smoothingDistanceSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" px", None))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.tab_imagecontour), QCoreApplication.translate("PPD_MainWindow", u"Drop shape characterization", None))
        self.parametersGroupBox.setTitle(QCoreApplication.translate("PPD_MainWindow", u"Parameters finding", None))
        self.pixelSizeLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Pixel spacing", None))
        self.pixelSizeSpinbox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"Click to provide pixel spacing (in mm)", None))
        self.pixelSizeSpinbox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mm/px", None))
        self.pixelDensityLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Pixel density", None))
        self.pixelDensitySpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"Click to provide pixel density (in mm\u207b\u00b9) ", None))
        self.pixelDensitySpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" px/mm", None))
        self.autoGuessPushButton.setText(QCoreApplication.translate("PPD_MainWindow", u"Auto-guess parameters", None))
        self.caplengthLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Capillary length", None))
        self.tipxCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Fit this parameter", None))
        self.tipxLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Tip x position", None))
        self.anglegSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.anglegSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" deg", None))
        self.tipyLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Tip y position", None))
        self.anglegCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Fit this parameter", None))
        self.tipxSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.tipxSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" px", None))
        self.tipySpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.tipySpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" px", None))
        self.anglegLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Angle of gravity", None))
        self.tipyCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Fit this parameter", None))
        self.r0Label.setText(QCoreApplication.translate("PPD_MainWindow", u"Drop tip radius", None))
        self.r0SpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.r0SpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mm", None))
        self.caplengthSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.caplengthSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mm", None))
        self.r0CheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Fit this parameter", None))
        self.caplengthCheckBox.setText(QCoreApplication.translate("PPD_MainWindow", u"Fit this parameter", None))
        self.optimizePushButton.setText(QCoreApplication.translate("PPD_MainWindow", u"Optimize (fit the checked parameters)", None))
        self.physicsGroupBox.setTitle(QCoreApplication.translate("PPD_MainWindow", u"Physical variables", None))
        self.rhogLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"density x gravity (g x Delta rho)", None))
        self.gammaLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Surface tension gamma", None))
        self.gammaSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.gammaSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mN/m", None))
        self.bondLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Bond number", None))
        self.bondSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"-", None))
        self.worthingtonLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Worthington number", None))
        self.worthingtonSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"Not implemented yet", None))
        self.dropVolumeLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Drop volume", None))
        self.dropVolumeSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"Not implemented yet", None))
        self.dropVolumeSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mm\u00b3", None))
        self.needleDiameterSpinBox.setSpecialValueText(QCoreApplication.translate("PPD_MainWindow", u"Not implemented yet", None))
        self.needleDiameterSpinBox.setSuffix(QCoreApplication.translate("PPD_MainWindow", u" mm", None))
        self.needleDiameterLabel.setText(QCoreApplication.translate("PPD_MainWindow", u"Needle diameter", None))
        self.analysisTabs.setTabText(self.analysisTabs.indexOf(self.tab_measure), QCoreApplication.translate("PPD_MainWindow", u"Measurement", None))
    # retranslateUi

