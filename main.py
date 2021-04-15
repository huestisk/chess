import sys

from models.models import ChessNN

task = sys.argv[1]
if task == 'play':
    from GUI import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

elif task == 'train':
    import training.train   # TODO: convert to funciton to allow parameters