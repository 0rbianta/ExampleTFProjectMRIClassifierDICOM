from cProfile import label
from statistics import mode


class neuralRunner:
    def __init__(self, modelPath, labelPath):
        self.modelpath = modelPath
        self.labelPath = labelPath
    