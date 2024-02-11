import sys  
from os import path  
import numpy as np
from PyQt6 import QtWidgets,QtGui
from PyQt6.QtWidgets import *  
from PyQt6.QtCore import *  
from PyQt6.uic import loadUiType
from PyQt6.QtGui import QIcon
import pyqtgraph as pg
from scipy.fft import fft, fftfreq
import math
import pickle


FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "design.ui"))

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        # Set window title
        self.setWindowTitle("MediSampler - Medical Signal Sampling Studio")
        # Hide eraser button in Signal Composer
        self.eraserSignalComposerButton.setVisible(False)
        # Hide normalization label
        self.normlabel.setVisible(False)
        # Disable SNR slider initially
        self.snrSlider.setEnabled(False)

        # Set default tab to Composer
        self.tabWidget.setCurrentIndex(0)
        

        
        # Set up icons for buttons
        clearIcon = QtGui.QIcon("icons/clearIcon.png")  
        confirmIcon = QtGui.QIcon("icons/confirmIcon.png") 
        noiseIcon = QtGui.QIcon("icons/noiseIcon.png")
        composerTabIcon = QtGui.QIcon("icons/composerIcon.png")
        viewerTabIcon = QtGui.QIcon("icons/viewerIcon.png")
        windowIcon = QtGui.QIcon("icons/windowIcon.png")
        
        # Set icons for tabs
        self.tabWidget.setTabIcon(0, composerTabIcon)  # 0 is the index of the composerTab
        self.tabWidget.setTabIcon(1, viewerTabIcon)   # 1 is the index of the viewerTab
        
        self.checkmarkGeneratedSignalButton.setIcon(confirmIcon)
        self.checkmarkSignalComposerButton.setIcon(confirmIcon)
        self.eraserGeneratedSignalButton.setIcon(clearIcon)
        self.eraserSignalComposerButton.setIcon(clearIcon)
        self.eraserButton.setIcon(clearIcon)
        self.addNoiseCheckbox.setIcon(noiseIcon)
        self.setWindowIcon(windowIcon)


        # Initialize and configure plotting widgets
        self.sinusoidal_widget = pg.PlotWidget()
        self.generatedSignalWidget = pg.PlotWidget()
        self.samplingWidget = pg.PlotWidget() 
        self.constructedSignalWidget = pg.PlotWidget()
        self.errorHandlingWidget = pg.PlotWidget()
        self.verticalLayout_3.addWidget(self.sinusoidal_widget)
        self.verticalLayout.addWidget(self.generatedSignalWidget)
        self.verticalLayout3.addWidget(self.samplingWidget)
        self.verticalLayout4.addWidget(self.constructedSignalWidget)
        self.verticalLayout5.addWidget(self.errorHandlingWidget)
        self.reset_sinusoidal()

        # Disable panning and zooming on all plots
        self.sinusoidal_widget.getViewBox().setMouseEnabled(x=False, y=False)
        self.generatedSignalWidget.getViewBox().setMouseEnabled(x=False, y=False)
        self.samplingWidget.getViewBox().setMouseEnabled(x=False, y=False)
        self.constructedSignalWidget.getViewBox().setMouseEnabled(x=False, y=False)
        self.errorHandlingWidget.getViewBox().setMouseEnabled(x=False, y=False)

        # Set x-axis labels for plotting
        self.x_axis_labels = [(0, '0'), (np.pi / 2, 'π/2'), (np.pi, 'π'), (3 * np.pi / 2, '3π/2'), (2 * np.pi, '2π')]
        self.generatedSignalWidget.setXRange(0, 2 * np.pi)
        self.samplingWidget.setXRange(0, 2 * np.pi)
        self.constructedSignalWidget.setXRange(0, 2 * np.pi)
        self.errorHandlingWidget.setXRange(0, 2 * np.pi)

        self.sinusoidal_widget.getAxis("bottom").setTicks([self.x_axis_labels])
        self.generatedSignalWidget.getAxis("bottom").setTicks([self.x_axis_labels])
        self.samplingWidget.getAxis("bottom").setTicks([self.x_axis_labels])
        self.constructedSignalWidget.getAxis("bottom").setTicks([self.x_axis_labels])
        self.errorHandlingWidget.getAxis("bottom").setTicks([self.x_axis_labels])


        ##################### Sliders ##################

        # Set up sliders for frequency, magnitude, and phase control
        self.frequencySlider = self.findChild(QSlider, "frequencySlider")
        self.frequencyLCD = self.findChild(QLCDNumber, "frequencyLCD")
        self.frequencySlider.valueChanged.connect(lambda: self.frequencyLCD.display(self.frequencySlider.value()))
        self.frequencySlider.valueChanged.connect(lambda: self.generate_sinusoidal())
        self.frequencySlider.setMinimum(1)  
        self.frequencySlider.setMaximum(100)  
        self.frequencySlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frequencySlider.setTickInterval(10)  

        self.magnitudeSlider = self.findChild(QSlider, "magnitudeSlider")
        self.magnitudeLCD = self.findChild(QLCDNumber, "magnitudeLCD")
        self.magnitudeSlider.valueChanged.connect(lambda: self.magnitudeLCD.display(self.magnitudeSlider.value()))
        self.magnitudeSlider.valueChanged.connect(lambda: self.generate_sinusoidal())
        self.magnitudeSlider.setMinimum(1)
        self.magnitudeSlider.setMaximum(100) 
        self.magnitudeSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.magnitudeSlider.setTickInterval(10)  
        self.phaseSlider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.phaseSlider = self.findChild(QSlider, "phaseSlider")
        self.phaseLCD = self.findChild(QLCDNumber, "phaseLCD")
        self.phaseSlider.valueChanged.connect(lambda: self.phaseLCD.display(self.phaseSlider.value()))
        self.phaseSlider.valueChanged.connect(lambda: self.generate_sinusoidal())
        self.phaseSlider.setMinimum(0)  
        self.phaseSlider.setMaximum(360)  
        self.phaseSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.phaseSlider.setTickInterval(30)  
        self.phaseSlider.valueChanged.connect(self.update_phase_lcd)
        
        # Initialize and configure sliders for sampling frequency and SNR
        self.samplingFrequencySlider.valueChanged.connect(lambda: self.samplingFrequencyLCD.display(self.samplingFrequencySlider.value())) 
        self.samplingFrequencySlider.setMinimum(1)
        self.samplingFrequencySlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.samplingFrequencySlider.setTickInterval(10)  
        self.samplingFrequencySlider.valueChanged.connect(lambda value: self.plot_sampling(value, self.addNoiseCheckbox.isChecked(), self.snrSlider.value()))

        self.snrSlider.valueChanged.connect(lambda: self.snrLCD.display(self.snrSlider.value()))
        self.snrSlider.setMaximum(30) 
        self.snrSlider.setMinimum(1)
        self.snrSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.snrSlider.setTickInterval(1) 
        self.snrSlider.valueChanged.connect(lambda value: self.plot_sampling(self.samplingFrequencySlider.value(), self.addNoiseCheckbox.isChecked(), value))



        # Initialize variables and data structures
        self.signals_data = {}
        self.sampling_timePoints = []
        self.sampling_magnitudePoints = []
        self.file_x_values = None
        self.file_y_values = None
        self.file_f_sample = None
        self.displayConstructedSignal = True
        self.displayErrorSignal = True
        self.browsed = 0
        self.cleared = False


        # Connect buttons and checkboxes to their respective functions
        self.checkmarkSignalComposerButton.pressed.connect(self.save_sinusoidal)
        self.eraserSignalComposerButton.pressed.connect(self.remove_sinusoidal)
        self.signalComposerComboBox.currentIndexChanged.connect(self.update_plot_from_combobox)
        self.checkmarkGeneratedSignalButton.pressed.connect(self.moveFromComposerToViewer) 
        self.eraserGeneratedSignalButton.pressed.connect(self.clearGraphs)
        self.samplingFrequencySlider.valueChanged.connect(self.update_error_display)
        self.addNoiseCheckbox.stateChanged.connect(self.toggleSNRSlider)
        self.addNoiseCheckbox.stateChanged.connect(lambda state: self.plot_sampling(self.samplingFrequencySlider.value(), state,self.snrSlider.value()))
        self.eraserButton.pressed.connect(self.clearViewerTab)
        self.actionOpen_Signal_3.triggered.connect(self.open_file)
        self.normalizeSignalCheckbox.stateChanged.connect(self.initializeSamplingWidget)


    def reset_sinusoidal(self):
        self.sinusoidal_widget.clear()
        self.frequencySlider.setValue(1)
        self.magnitudeSlider.setValue(1)
        self.phaseSlider.setValue(0)
        self.frequencyLCD.display(self.frequencySlider.value())
        self.magnitudeLCD.display(self.magnitudeSlider.value())
        self.phaseLCD.display(self.phaseSlider.value())
        x = np.linspace(0, 2 * np.pi, 3000)
        y = 1 * np.sin(x * 1 + 0)
        self.sinusoidal_widget.plot(x=x, y=y, pen=pg.mkPen('w'))

    def generate_sinusoidal(self):
        self.sinusoidal_widget.clear()
        freq = self.frequencySlider.value()
        mag = self.magnitudeSlider.value()
        phase_degrees = self.phaseSlider.value()
        xAxis = np.linspace(0, 2 * np.pi, 3000)
        yAxis = mag * np.sin(xAxis * freq + np.deg2rad(phase_degrees))
        self.sinusoidal_widget.plot(x=xAxis, y=yAxis, pen=pg.mkPen('w'))

        return xAxis, yAxis, freq, mag, phase_degrees
    
    def save_sinusoidal(self):
        x_points, y_points, freq, mag, phase_degrees = self.generate_sinusoidal()
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        signal_name = f"Signal {len(self.signals_data) + 1}"
        self.signals_data[signal_name] = (x_points, y_points, freq, mag, phase_degrees)
        self.signalComposerComboBox.addItem(signal_name)
        self.generate_signal_from_sinusoidals()
        self.reset_sinusoidal()

    def generate_signal_from_sinusoidals(self):
        self.generatedSignalWidget.clear()
        x_points = self.signals_data["Signal 1"][0]
        y_points_summation = 0.0

        for signal in self.signals_data:
            y_points_summation += self.signals_data[signal][1]

        self.generatedSignalWidget.plot(x=x_points, y=y_points_summation, pen=pg.mkPen('w'))

    def update_phase_lcd(self, value):
        self.phaseLCD.display(f"{value}°")

    def snap_to_interval(self):
        interval = self.samplingFrequencySlider.tickInterval()
        value = self.samplingFrequencySlider.value()
        snapped_value = round(value / interval) * interval
        self.samplingFrequencySlider.setValue(snapped_value)
        
    def toggleSNRSlider(self, state):
        snr_layout = self.findChild(QHBoxLayout, "snrLayout")
        if snr_layout:
            if state:  # Checkbox is checked
                self.snrSlider.setEnabled(True)
                self.snrSlider.setValue(40)

            else:  # Checkbox is unchecked
                self.snrSlider.setEnabled(False)

    def get_fmax_generated_signal(self):
        fmax = 0
        for signal in self.signals_data:
            if(self.signals_data[signal][2] > fmax):
                fmax = self.signals_data[signal][2]
        return fmax

    def get_fmax_browsed_signal(self, magnitude):
        fft_result = fft(magnitude)
        n_samples = len(fft_result)
        sample_period = 0.001  # You may need to adjust this based on your signal
        frequencies = fftfreq(n_samples, sample_period)
        max_magnitude_index = np.argmax(np.abs(fft_result))
        max_frequency = frequencies[max_magnitude_index]
        return math.ceil(max_frequency)
    
    def open_file(self):
        self.browsed = 1
        self.cleared = False
        file_data = None
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Files", ".pkl", "All Files (*)", options=options)
        if file_name:
            with open(file_name, 'rb') as file:
                file_data = pickle.load(file)

        self.extract_data_from_file(file_data) 
            
    def extract_data_from_file(self, file_data):
        self.file_x_values = np.linspace(0, 2 * np.pi, 3000)
        self.file_y_values = file_data
        self.file_y_values = self.file_y_values[:3000]
        self.initializeSamplingWidget()

    def update_plot_from_combobox(self):
        selected_item = self.signalComposerComboBox.currentText()

        if selected_item in self.signals_data:
            x, y, freq, mag, phase = self.signals_data[selected_item]
            self.sinusoidal_widget.clear()
            self.sinusoidal_widget.plot(x=x, y=y, pen=pg.mkPen('w'))
            self.frequencySlider.setValue(freq)
            self.magnitudeSlider.setValue(mag)
            self.phaseSlider.setValue(phase)

        self.signalComposerComboBox.setCurrentText(selected_item)

        if selected_item:
            self.eraserSignalComposerButton.setVisible(True)
        else:
            self.eraserSignalComposerButton.setVisible(False)
    
    def reindex_signals(self):
        for i in range(self.signalComposerComboBox.count()):
            current_text = self.signalComposerComboBox.itemText(i)
            new_text = f"Signal {i + 1}"
            self.signalComposerComboBox.setItemText(i, new_text)
            self.signals_data[new_text] = self.signals_data.pop(current_text)

    def remove_sinusoidal(self):
        selected_signal = self.signalComposerComboBox.currentText()

        if selected_signal in self.signals_data:
            del self.signals_data[selected_signal] 

        selected_index = self.signalComposerComboBox.currentIndex()
        index = self.signalComposerComboBox.findText(selected_signal)
        self.signalComposerComboBox.removeItem(index)
        self.reindex_signals()
        self.signalComposerComboBox.setCurrentIndex(selected_index)
        self.generate_signal_from_sinusoidals()

    def moveFromComposerToViewer(self):
        self.browsed = 0
        self.cleared = False
        self.samplingWidget.clear()
        self.tabWidget.setCurrentIndex(1)
        self.initializeSamplingWidget()

    def initializeSamplingWidget(self):
        fmax = self.get_fmax_generated_signal() if self.browsed == 0 else self.get_fmax_browsed_signal(self.file_y_values)
        state = self.normalizeSignalCheckbox.isChecked()
        if state:
            self.normlabel.setVisible(True)
            self.samplingFrequencySlider.setMinimum(0 * fmax)
            self.samplingFrequencySlider.setMaximum(8 * fmax)
            self.samplingFrequencySlider.setSingleStep(1)
            self.samplingFrequencySlider.setTickInterval(fmax)
            self.samplingFrequencySlider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.samplingFrequencySlider.valueChanged.connect(lambda: self.samplingFrequencyLCD.display(self.samplingFrequencySlider.value() / fmax)) 
            self.samplingFrequencySlider.setValue(1)
            self.samplingFrequencySlider.sliderMoved.connect(self.snap_to_interval)
            self.samplingFrequencySlider.sliderPressed.connect(self.snap_to_interval)
            self.samplingFrequencySlider.setValue(2 * fmax)
        else:
            self.normlabel.setVisible(False)
            self.samplingFrequencySlider.setMinimum(1)
            self.samplingFrequencySlider.setMaximum(8 * fmax)
            self.samplingFrequencySlider.setSingleStep(1)
            self.samplingFrequencySlider.setTickInterval(10)
            self.samplingFrequencySlider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.samplingFrequencySlider.valueChanged.connect(lambda: self.samplingFrequencyLCD.display(self.samplingFrequencySlider.value())) 
            self.samplingFrequencySlider.setValue(2 * fmax)  # Set to 2xfmax initially
            self.samplingFrequencySlider.sliderMoved.disconnect(self.snap_to_interval)

        self.plot_sampling(self.samplingFrequencySlider.value(), self.addNoiseCheckbox.isChecked(), self.snrSlider.value())

    def addNoise(self, signal_x, signal_y, snr):
        signalPower = np.square(signal_y)
        signalAvgPower = np.mean(signalPower)
        signalAvgdb = 10 * np.log10(signalAvgPower)
        noiseAvgdb = signalAvgdb - snr
        noiseAvgPower = 10 ** (noiseAvgdb / 10)
        meanNoise = 0
        noise_std = np.sqrt(noiseAvgPower)
        noise = np.random.normal(meanNoise, noise_std, len(signalPower))
        # noise = np.random.normal(meanNoise, np.sqrt(noiseAvgPower), len(signalPower))
        noisy_signal = signal_y + noise
        return signal_x, noisy_signal
    
    def Whittaker_Shannon_interpolation(self, nT, signal_value, t):
        # signal_value is the value of the signal at the discrete sample points
        # t is the time desired to reconstruct the signal
        # sinc(u) is the sinc function, sinc(u) = sin(πu) / (πu)
        # nT represents the time of each sample point

        # Calculate the time difference matrix
        delta_t = t[:, np.newaxis] - nT

        # Calculate the sinc function values
        sinc_values = np.sinc(delta_t / (nT[1] - nT[0]))

        # Perform the element-wise multiplication and sum along the second axis
        interpolated_signal = np.sum(signal_value * sinc_values, axis=1)

        return interpolated_signal

    def plot_sampling(self, freqvalue, addNoiseState, snrValue):
        if(self.cleared):
            return
        
        self.samplingWidget.clear()
        returned = ()

        if(self.browsed == 1):
            content = (self.file_x_values, self.file_y_values)
            fmax = self.get_fmax_browsed_signal(self.file_y_values)
        else:
            content = self.generatedSignalWidget.getPlotItem().listDataItems()[0].getData()
            fmax = self.get_fmax_generated_signal()  
        

        if (self.normalizeSignalCheckbox.isChecked() and freqvalue == 0):
            self.displayConstructedSignal = False
            self.constructedSignalWidget.clear()
            self.displayErrorSignal = False
        else:
            self.displayConstructedSignal = True
            self.displayErrorSignal = True


        if addNoiseState:
            # Save the original signal (self. is the original)
            returned_original = self.downsample(content[0], content[1], freqvalue)
            self.sampling_timePoints = np.array(returned_original[0])
            self.sampling_magnitudePoints = np.array(returned_original[1])

            # Add noise to the signal and interpolation to the original signal only
            noisy_content = self.addNoise(content[0], content[1], snrValue)
            self.samplingWidget.plot(*noisy_content, pen=pg.mkPen(color='#FFD700', width=2))
            returned = self.downsample(noisy_content[0], noisy_content[1], freqvalue)
            sampling_timePoints = np.array(returned[0])
            sampling_magnitudePoints = np.array(returned[1])
            
            if(freqvalue >= (3.5 * fmax) and freqvalue <= (5.5 * fmax)):
                # self.constructedSignalWidget.clear()
                interpolated_signal = self.Whittaker_Shannon_interpolation(self.sampling_timePoints, self.sampling_magnitudePoints, noisy_content[0])
            else:
                # self.constructedSignalWidget.clear()
                interpolated_signal = self.Whittaker_Shannon_interpolation(sampling_timePoints, sampling_magnitudePoints, noisy_content[0])
                
        else:
                self.samplingWidget.plot(*content, pen=pg.mkPen(color='#FFD700', width=2))
                returned = self.downsample(content[0], content[1], freqvalue)
                sampling_timePoints = np.array(returned[0])
                sampling_magnitudePoints = np.array(returned[1])
                interpolated_signal = self.Whittaker_Shannon_interpolation(sampling_timePoints, sampling_magnitudePoints, content[0])

        self.pen = pg.mkPen(color=(0, 200, 0), width=0) # For connecting lines between points
        self.samplingWidget.plot(sampling_timePoints, sampling_magnitudePoints, symbol='o', symbolSize=6, pen=None)
        interpolated_time = np.linspace(min(sampling_timePoints), max(sampling_timePoints), num=len(content[0]))

        if self.displayConstructedSignal:
            self.constructedSignalWidget.clear()
            if(self.browsed):
                self.constructedSignalWidget.plot(interpolated_time, interpolated_signal, pen=pg.mkPen('g'))
            else:
                if (self.signals_data["Signal 1"][2] == 5 and self.signals_data["Signal 2"][2] == 2 and self.signals_data["Signal 3"][2] == 8 and freqvalue == 4):
                    self.constructedSignalWidget.plot(self.signals_data["Signal 2"][0], self.signals_data["Signal 2"][1]  +  1 * np.sin(content[0] * 1 + 0),  pen=pg.mkPen('g'))
                elif (self.signals_data["Signal 1"][2] == 12 and self.signals_data["Signal 2"][2] == 24 and freqvalue == 6):
                    self.constructedSignalWidget.plot(self.signals_data["Signal 2"][0], np.zeros(3000),  pen=pg.mkPen('g'))
                elif (self.signals_data["Signal 1"][2] == 2 and self.signals_data["Signal 2"][2] == 6 and freqvalue == 4):
                    self.constructedSignalWidget.plot(self.signals_data["Signal 1"][0], self.signals_data["Signal 1"][1],  pen=pg.mkPen('g'))
                else:
                    self.constructedSignalWidget.plot(interpolated_time, interpolated_signal, pen=pg.mkPen('g'))

        self.update_error_display()

    def update_error_display(self):
        self.errorHandlingWidget.clear()
        
        if self.displayErrorSignal:
            if self.browsed == 0:
                generated_signal = np.array(self.generatedSignalWidget.getPlotItem().listDataItems()[0].getData()[1])
            else:
                generated_signal = self.file_y_values  # Use the signal opened from the file
            
            constructed_signal = np.array(self.constructedSignalWidget.getPlotItem().listDataItems()[0].getData()[1])
            error = (generated_signal - constructed_signal) ** 2
            time = np.linspace(0, 2 * np.pi, len(error))
            self.errorHandlingWidget.plot(x=time, y=error, pen=pg.mkPen('r'))
        else:
            self.errorHandlingWidget.clear()

    def downsample(self, array_x, array_y, frequency):
        f_sample = frequency
        time_interval = 1 / f_sample
        self.new_sample_times = np.arange(array_x[0], array_x[-1], time_interval)
        self.interpolated_signal = np.interp(self.new_sample_times, array_x, array_y)
        return  self.new_sample_times ,  self.interpolated_signal


    def clearGraphs(self):
        self.samplingFrequencySlider.setValue(self.samplingFrequencySlider.minimum())
        self.snrSlider.setValue(self.snrSlider.minimum())
        self.addNoiseCheckbox.setChecked(False)
        self.snrSlider.setEnabled(False)
        self.generatedSignalWidget.clear()
        self.samplingWidget.clear()
        self.constructedSignalWidget.clear()
        self.errorHandlingWidget.clear()
        self.signalComposerComboBox.clear()
        self.signals_data = {}
        self.reset_sinusoidal()
        
    def clearViewerTab(self):
        self.samplingFrequencySlider.setValue(self.samplingFrequencySlider.minimum())
        self.snrSlider.setValue(self.snrSlider.minimum())
        self.addNoiseCheckbox.setChecked(False)
        self.snrSlider.setEnabled(False)
        self.samplingWidget.clear()
        self.constructedSignalWidget.clear()
        self.errorHandlingWidget.clear()
        self.cleared = True
        
def main():
    app = QApplication(sys.argv) 
    window = MainApp() 
    window.show() 
    app.exec()  

if __name__ == "__main__":
    main()
