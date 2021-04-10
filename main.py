import sys

from PyQt5.QtWidgets import QApplication
from GUI import MainWindow

from training.prioritizedDQN import CnnDQN # FIXME

task = sys.argv[1]

if task == 'play':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

elif task == 'train':
    algo = sys.argv[2] if len(sys.argv) > 2 else None

    if algo.lower() == 'prioritizedDQN'.lower():
        import training.trainPriorDQN   # TODO: convert to funciton to allow parameters