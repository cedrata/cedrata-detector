# ###########################################################################
# Author: Luca Greggio
# Github: ...
# Email: luca01.greggio@edu.unife.it, greggius9891@gmail.com
# This is a machine learning project able to detect where notes are played 
# in audio tracks. 
# If you are interested in it please please beofre using this code contact me
# ###########################################################################

# Pyton modules
import sys

# PyQt modules
from PyQt5.QtWidgets import *

# Weka modules
import weka.core.jvm as jvm

# Local modules
from note_detection import *


if __name__ == "__main__":
    # Starting weka jvm
    jvm.start()

    # Building window
    app = QApplication(sys.argv)
    window = MainWindow()

    # Apply style
    with open("Diffnes.qss", "r") as stylesheet:
        qss = stylesheet.read()
        app.setStyleSheet(qss)

    # Show Window
    window.show()

    # exit application, to stop jvm checkout "def closeEvent"
    sys.exit(app.exec_())
