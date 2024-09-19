import sys
import random
import matplotlib
import time
import kdc_control
import cProfile
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, motorapp, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Laser Feedback Control")

        layout = QHBoxLayout()
        layout2 = QVBoxLayout()
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        print("test")
        layout.addWidget(self.canvas1)
        self.keepAligned = QPushButton("Keep Beam Aligned")
        self.mapField = QPushButton("Map Beam's Physical Field")
        layout2.addWidget(self.keepAligned)
        layout2.addWidget(self.mapField)
        layout.addLayout(layout2)
        layout.addWidget(self.canvas2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.keepAligned.clicked.connect(self.keepAlignedClicked)
        self.mapField.clicked.connect(self.mapFieldClicked)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref1 = None
        self._plot_ref2 = None
        self.canvas1.axes.set_title("Observed Modulated Signal Amplitude")
        self.canvas1.axes.set_xlabel("Frequency")
        self.canvas1.axes.set_ylabel("Amplitude (Vrms)")
        self.canvas2.axes.set_title("Motor Position")
        self.canvas2.axes.set_xlabel("Yaw Offset (degrees)")
        self.canvas2.axes.set_ylabel("Pitch Offset (degrees)")

        self.motorapp = motorapp

        self.show()

        # # Setup a timer to trigger the redraw by calling update_plot.
        # self.timer = QtCore.QTimer()
        # self.timer.setInterval(100)
        print(motorapp.start_index)
        print(motorapp.end_index)
        display_spectrum = motorapp.getSpectrum()[motorapp.start_index:motorapp.end_index]
        print(len(display_spectrum))
        self.update_plot1(display_spectrum)
        self.update_plot2(motorapp.yaw_locs, motorapp.pitch_locs)

    def keepAlignedClicked(self):
        if self.keepAligned.text() == "Keep Beam Aligned":
            self.keepAligned.setText("Keeping Aligned")
            self.mapField.setText("Cannot Map Field While Keeping Aligned")
            self.setEnabled(False)
            # motorapp.keepCentered()
        else:
            self.keepAligned.setText("Keep Beam Aligned")
            self.mapField.setText("Map Beam's Physical Field")
            self.setEnabled(True)
            # motorapp.keepingCentered = False
    
    def mapFieldClicked(self):
        self.keepAligned.setText("Cannot Keep Aligned While Mapping Field")
        self.mapField.setText("Mapping Field")
        self.keepAligned.setEnabled(False)
        self.mapField.setEnabled(False)
        self.motorapp.plotField(200)
        self.keepAligned.setEnabled(True)
        self.mapField.setEnabled(True)
        self.keepAligned.setText("Keep Beam Aligned")
        self.mapField.setText("Map Beam's Physical Field")

        
    def update_plot1(self, spectrum):
        # self.spectrum = m.getSpectrum()[m.start_index:m.end_index]

        # Note: we no longer need to clear the axis.
        if self._plot_ref1 is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            
            plot_ref1s = self.canvas1.axes.plot(self.motorapp.freq_range, spectrum, 'r')
            # print(len(spectrum))
            # print(len(self.motorapp.freq_range))
            self._plot_ref1 = plot_ref1s[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref1.set_ydata(spectrum)
            self._plot_ref1.set_ydata([0]*180)

        # Trigger the canvas1 to update and redraw.
        self.canvas1.draw()

    def update_plot2(self, yaw_locs, pitch_locs):
        # self.yaw_loc_data = m.yaw_locs
        # self.pitch_loc_data = m.pitch_locs

        # Note: we no longer need to clear the axis.
        if self._plot_ref2 is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_ref2s = self.canvas2.axes.plot(yaw_locs, pitch_locs, 'r')
            self.canvas2.axes.set_xlim([start_yaw - yawBoundary, start_yaw + yawBoundary])
            self.canvas2.axes.set_ylim([start_pitch - pitchBoundary, start_pitch + pitchBoundary])
            self._plot_ref2 = plot_ref2s[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref2.set_data(yaw_locs, pitch_locs)

        # Trigger the canvas1 to update and redraw.
        self.canvas2.draw()
    
    # def quitApp(self):
    #     QApplication().quit()


def revealWindow():

    with kdc_control.MotorApplication(serialNoYaw, serialNoPitch, yawBoundary=yawBoundary, pitchBoundary=pitchBoundary, deviationVal=deviationVal, startYaw=start_yaw, startPitch=start_pitch) as m:
        app = QtWidgets.QApplication(sys.argv)
        print("created")
        w = MainWindow(m)
        print("made main window")
        # motorapp = m
        # m.window_revealed = True
        m.windowRevealed(w)
        app.exec()
        print("executed")


if __name__ == '__main__':

    # serialNoYaw = "27006315"
    # serialNoPitch = "27006283"
    serialNoYaw = "27250209"
    serialNoPitch = "27250140"
    yawBoundary = 2000
    pitchBoundary = 1000
    start_yaw = 218523
    start_pitch = 241626
    deviationVal = 0.5
    binFactor = 200

    # motorapp = None
    # cProfile.run('revealWindow()')
    revealWindow()