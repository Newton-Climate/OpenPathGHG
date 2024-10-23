import sys
import numpy as np
import pyqtgraph as pg
import time
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from random import randint

from PyQt6 import QtCore, QtWidgets

import threading
from threading import Thread

import beam_align


class Worker(QObject):
    def __init__(self, beamaligner):
        super(Worker, self).__init__()
        self.beamaligner = beamaligner

    finished = pyqtSignal()

    def keepCentered(self):
        self.beamaligner.keepCentered()
        self.finished.emit()

    def plotField(self):
        self.beamaligner.plotField(200)
        self.finished.emit()

    def calibrateSampleBin(self):
        self.beamaligner.calibrateSampleBin()
        self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, beamaligner):
        super().__init__()

        self.beamaligner = beamaligner

        self.setWindowTitle("Laser Feedback Control")

        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()

        self.keepAligned = QPushButton("Keep Beam Aligned")
        self.mapField = QPushButton("Map Beam's Physical Field")
        self.calibrateDevice = QPushButton("Calibrate Device")
        layout2.addWidget(self.keepAligned)
        layout2.addWidget(self.mapField)
        layout2.addWidget(self.calibrateDevice)
        layout1.addLayout(layout2)

        self.pg_layout = pg.GraphicsLayoutWidget()

        # Add subplots
        self.graph1 = self.pg_layout.addPlot(
            row=0, col=0, title="Observed Modulated Signal Amplitude"
        )
        self.graph2 = self.pg_layout.addPlot(row=0, col=1, title="Motor Position")
        self.graph1.setLabel("left", "Amplitude (Vrms)")
        self.graph1.setLabel("bottom", "Frequency")
        self.graph2.setLabel("left", "Pitch Offset (degrees)")
        self.graph2.setLabel("bottom", "Yaw Offset (degrees)")

        self.pg_layout.setBackground("w")

        # Show our layout holding multiple subplots
        layout1.addWidget(self.pg_layout)
        container = QWidget()
        container.setLayout(layout1)
        self.setCentralWidget(container)

        self.keepAligned.clicked.connect(self.keepAlignedClicked)
        self.mapField.clicked.connect(self.mapFieldClicked)
        self.calibrateDevice.clicked.connect(self.calibrateDeviceClicked)

        self.update_plot1()
        self.update_plot2()
        print(threading.current_thread().name)

        time.sleep(5)
        # # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_plot1)
        self.timer.timeout.connect(self.update_plot2)
        self.timer.start()

    def keepAlignedClicked(self):
        if self.keepAligned.text() == "Keep Beam Aligned":

            self.thread = QThread()
            self.worker = Worker(self.beamaligner)
            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.keepCentered)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.restoreButtons)

            self.thread.start()

            self.keepAligned.setText("Keeping Aligned")
            self.mapField.setText("Cannot Map Field While Keeping Aligned")
            self.calibrateDevice.setText(
                "Cannot Calibrate Device While Keeping Aligned"
            )
            self.mapField.setEnabled(False)
            self.calibrateDevice.setEnabled(False)
        else:
            self.keepAligned.setText("Shutting Down Alignment Procedure")
            self.keepAligned.setEnabled(False)
            self.beamaligner.keepingCentered = False

    def mapFieldClicked(self):

        self.thread = QThread()
        self.worker = Worker(self.beamaligner)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.plotField)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.keepAligned.setText("Cannot Keep Aligned While Mapping Field")
        self.mapField.setText("Mapping Field")
        self.calibrateDevice.setText("Cannot Calibrate Device While Mapping Field")
        self.keepAligned.setEnabled(False)
        self.mapField.setEnabled(False)
        self.calibrateDevice.setEnabled(False)

        self.thread.finished.connect(self.restoreButtons)

    def calibrateDeviceClicked(self):

        self.thread = QThread()
        self.worker = Worker(self.beamaligner)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.calibrateSampleBin)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.keepAligned.setText("Cannot Keep Aligned While Calibrating Device")
        self.mapField.setText("Cannot Map Field While Calibrating Device")
        self.calibrateDevice.setText("Calibrating Device")
        self.keepAligned.setEnabled(False)
        self.mapField.setEnabled(False)
        self.calibrateDevice.setEnabled(False)

        self.thread.finished.connect(self.restoreButtons)

    def restoreButtons(self):
        self.keepAligned.setEnabled(True)
        self.mapField.setEnabled(True)
        self.calibrateDevice.setEnabled(True)
        self.keepAligned.setText("Keep Beam Aligned")
        self.mapField.setText("Map Beam's Physical Field")
        self.calibrateDevice.setText("Calibrate Device")

    def update_plot1(self):
        spectrum = self.beamaligner.export_spectrum
        try:
            self.line1.setData(self.beamaligner.freq_range, spectrum)

        except:
            self.line1 = self.graph1.plot(
                self.beamaligner.freq_range,
                spectrum,
            )

    def update_plot2(self):
        yaw_locs = self.beamaligner.yaw_locs
        pitch_locs = self.beamaligner.pitch_locs
        try:
            self.line2.setData(yaw_locs, pitch_locs)
            self.dot.setData([yaw_locs[-1]], [pitch_locs[-1]], symbol="o")
        except:
            self.line2 = self.graph2.plot(
                yaw_locs,
                pitch_locs,
            )
            self.dot = pg.ScatterPlotItem()
            self.dot.addPoints([yaw_locs[-1]], [pitch_locs[-1]], symbol="o")
            self.graph2.addItem(self.dot)


if __name__ == "__main__":
    # serialNoYaw = "27006315"
    # serialNoPitch = "27006283"
    # start_yaw = -40867
    # start_pitch = -97862
    serialNoYaw = "27250209"
    serialNoPitch = "27250140"
    start_yaw = 218523  # (Lab)
    start_pitch = 241626  # (Lab)
    # start_yaw = 21111
    # start_pitch = -92993
    yawBoundary = 2000
    pitchBoundary = 1000
    deviationVal = 0.1
    binFactor = 200
    currPeaked = False
    with open("data/loc_coords.txt", "a", encoding="utf-8") as loc_log:
        with beam_align.BeamAligner(
            currPeaked=currPeaked,
            serialNoYaw=serialNoYaw,
            serialNoPitch=serialNoPitch,
            yawBoundary=yawBoundary,
            pitchBoundary=pitchBoundary,
            deviationVal=deviationVal,
            startYaw=start_yaw,
            startPitch=start_pitch,
            loc_log=loc_log,
        ) as newbeamaligner:
            app = QtWidgets.QApplication([])
            main = MainWindow(beamaligner=newbeamaligner)
            main.show()
            newbeamaligner.windowRevealed(main)
            app.exec()
