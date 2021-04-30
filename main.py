import sys
from models.models import ChessNN

task = sys.argv[1]

if task.startswith('train'):
    if task == 'trainDQN':
        from training.trainDQN import DQN
        trainer = DQN()

    elif task == 'trainPriorDQN':
        from training.trainPriorDQN import PriorDQN
        trainer = PriorDQN()

    trainer.train()

if task == 'play':
    from GUI import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()