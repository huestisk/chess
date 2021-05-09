import sys
from models.models import ChessNN

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = 'play'

if task.startswith('train'):
    if task == 'trainDQN':
        from training.trainDQN import DQN
        trainer = DQN()
    elif task == 'trainPriorDQN':
        from training.trainPriorDQN import PriorDQN
        trainer = PriorDQN()
    elif task == 'trainRainbow':
        from training.trainRainbow import Rainbow
        trainer = Rainbow()

    trainer.train()

elif task == 'play':
    from GUI import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()