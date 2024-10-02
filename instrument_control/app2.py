import sys
import kdc_control
import numpy as np
import pyqtgraph as pg
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from random import randint

from PyQt6 import QtCore, QtWidgets

import threading
from threading import Thread


    # time.sleep(5)
    # line1.setData(x*2, y/2)

    # status = app.exec()
    # sys.exit(status)

class Worker(QObject):
    def __init__(self, motorapp):
        super(Worker, self).__init__()
        self.motorapp = motorapp
    
    finished = pyqtSignal()

    def keepCentered(self):
        self.motorapp.keepCentered(50)
        finished.emit()
    
    def plotField(self):
        self.motorapp.plotField(200)
        finished.emit()
        

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, motorapp):
        super().__init__()

        self.motorapp = motorapp

        self.setWindowTitle("Laser Feedback Control")

        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()

        self.keepAligned = QPushButton("Keep Beam Aligned")
        self.mapField = QPushButton("Map Beam's Physical Field")
        layout2.addWidget(self.keepAligned)
        layout2.addWidget(self.mapField)
        layout1.addLayout(layout2)

        self.pg_layout = pg.GraphicsLayoutWidget()

        # Add subplots
        self.graph1 = self.pg_layout.addPlot(row=0, col=0, title="Observed Modulated Signal Amplitude")
        self.graph2 = self.pg_layout.addPlot(row=0, col=1, title="Motor Position")
        self.graph1.setLabel("left", "Amplitude (Vrms)")
        self.graph1.setLabel("bottom", "Frequency")
        self.graph2.setLabel("left", "Pitch Offset (degrees)")
        self.graph2.setLabel("bottom", "Yaw Offset (degrees)")

        self.pg_layout.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))

        # self.time = list(range(10))
        # self.temperature = [randint(20, 40) for _ in range(10)]

        # Show our layout holding multiple subplots
        layout1.addWidget(self.pg_layout)
        container = QWidget()
        container.setLayout(layout1)
        self.setCentralWidget(container)

        self.thread = QThread()
        self.worker = Worker(self.motorapp)
        self.worker.moveToThread(self.thread)

        self.keepAligned.clicked.connect(self.keepAlignedClicked)
        self.mapField.clicked.connect(self.mapFieldClicked)

        # display_spectrum = motorapp.getSpectrum()[motorapp.start_index:motorapp.end_index]
        # print(len(display_spectrum))
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

            self.thread.started.connect(self.worker.keepCentered)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.thread.start()

            self.keepAligned.setText("Keeping Aligned")
            self.mapField.setText("Cannot Map Field While Keeping Aligned")
            self.mapField.setEnabled(False)
        else:
            self.restoreButtons()
            self.motorapp.keepingCentered = False
        
    
    def mapFieldClicked(self):

        self.thread.started.connect(self.worker.plotField)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        
        self.keepAligned.setText("Cannot Keep Aligned While Mapping Field")
        self.mapField.setText("Mapping Field")
        self.keepAligned.setEnabled(False)
        self.mapField.setEnabled(False)

        self.thread.finished.connect(
                self.restoreButtons
            )
        
    def restoreButtons(self):
        self.keepAligned.setEnabled(True)
        self.mapField.setEnabled(True)
        self.keepAligned.setText("Keep Beam Aligned")
        self.mapField.setText("Map Beam's Physical Field")

    def update_plot1(self):
        # print("update1")
        # print(threading.current_thread().name)
        spectrum = self.motorapp.export_spectrum
        try:
            self.line1.setData(self.motorapp.freq_range, spectrum)

        except:
            # print("plotted")
            self.line1 = self.graph1.plot(self.motorapp.freq_range, spectrum,
                # pen=pen,
            )

    def update_plot2(self):
        # print("update2")
        yaw_locs = self.motorapp.yaw_locs
        pitch_locs = self.motorapp.pitch_locs
        try:
            self.line2.setData(yaw_locs, pitch_locs)
            self.graph2.setXRange(start_yaw - yawBoundary, start_yaw + yawBoundary)
            self.graph2.setYRange(start_pitch - pitchBoundary, start_pitch + pitchBoundary)

        except:
            self.line2 = self.graph2.plot(yaw_locs, pitch_locs,
                # pen=pen,
            )

    # def update_plot(self):
    #     self.time = self.time[1:]
    #     self.time.append(self.time[-1] + 1)
    #     self.temperature = self.temperature[1:]
    #     self.temperature.append(randint(20, 40))
    #     self.line1.setData(self.time, self.temperature)

def revealWindow(currPeaked):
    with open("loc_coords.txt", "a") as loc_log:
        with kdc_control.MotorApplication(currPeaked=currPeaked, serialNoYaw=serialNoYaw, serialNoPitch=serialNoPitch, yawBoundary=yawBoundary, pitchBoundary=pitchBoundary, deviationVal=deviationVal, startYaw=start_yaw, startPitch=start_pitch, loc_log=loc_log) as motorapp:
            app = QtWidgets.QApplication([])
            main = MainWindow(motorapp=motorapp)
            main.show()
            motorapp.windowRevealed(main)
            app.exec()

if __name__ == '__main__':
    # serialNoYaw = "27006315"
    # serialNoPitch = "27006283"
    # start_yaw = -40867
    # start_pitch = -97862
    serialNoYaw = "27250209"
    serialNoPitch = "27250140"
    start_yaw = 218523
    start_pitch = 241626
    yawBoundary = 2000
    pitchBoundary = 1000
    deviationVal = 0.1
    binFactor = 200
    revealWindow(currPeaked=False)