import sys
import random
import matplotlib
import time
import kdc_control 
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

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(motorapp, *args, **kwargs)

        self.setWindowTitle("Laser Feedback Control")

        layout = QHBoxLayout()
        layout2 = QVBoxLayout()
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)

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
        

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref1 = None
        self._plot_ref2 = None
        self.update_plot1()
        self.update_plot2()
        self.canvas1.axes.set_title("Observed Modulated Signal Amplitude")
        self.canvas1.axes.set_xlabel("Time")
        self.canvas1.axes.set_ylabel("Amplitude (dB)")
        self.canvas2.axes.set_title("Motor Position")
        self.canvas2.axes.set_xlabel("Yaw Offset (degrees)")
        self.canvas2.axes.set_ylabel("Pitch Offset (degrees)")

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot1)
        self.timer.timeout.connect(self.update_plot2)
        self.timer.start()

    def keepAlignedClicked(self):
        if self.keepAligned.text() == "Keep Beam Aligned":
            self.keepAligned.setText("Keeping Aligned")
            self.mapField.setText("Cannot Map Field While Keeping Aligned")
            self.setEnabled(False)
            motorapp.keepCentered()
        else:
            self.keepAligned.setText("Keep Beam Aligned")
            self.mapField.setText("Map Beam's Physical Field")
            self.setEnabled(True)
            motorapp.keepingCentered = False
    
    def mapFieldClicked(self):
        self.keepAligned.setText("Cannot Keep Aligned While Mapping Field")
        self.mapField.setText("Mapping Field")
        self.keepAligned.setEnabled(False)
        self.mapField.setEnabled(False)
        motorapp.plotField(1)
        self.keepAligned.setEnabled(True)
        self.mapField.setEnabled(True)
        self.keepAligned.setText("Keep Beam Aligned")
        self.mapField.setText("Map Beam's Physical Field")

        
    def update_plot1(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref1 is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_ref1s = self.canvas1.axes.plot(self.xdata, self.ydata, 'r')
            self._plot_ref1 = plot_ref1s[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref1.set_ydata(self.ydata)

        # Trigger the canvas1 to update and redraw.
        self.canvas1.draw()

    def update_plot2(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref2 is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_ref2s = self.canvas2.axes.plot(self.xdata, self.ydata, 'r')
            self._plot_ref2 = plot_ref2s[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref2.set_ydata(self.ydata)

        # Trigger the canvas1 to update and redraw.
        self.canvas2.draw()



if __name__ == '__main__':
    try:
        serialNoYaw = "27006315"
        serialNoTilt = "27006283"
        deviationVal = 0.1

        with kdc_control.MotorApplication(serialNoYaw, serialNoTilt, deviationVal) as m:
            app = QtWidgets.QApplication(m, sys.argv)
            w = MainWindow()
            app.exec()
            # m.plotField(1)

    except Exception as e:
        print("ERROR! ALERT!")
        print(e)