# bootstrp zoidberg

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import signal, time, readchar
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class DatasetsMatrix:
    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = mnist.load_data()
        self.stat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.train_X_single_line = []
        self.test_X_single_line = []
        self.scaler = StandardScaler()
        self.classifier = KNeighborsClassifier(n_neighbors = 8)
        self.class_report = None
        self.conf_matrix = None
        self.accu_score = None

    def print_raw_data(self):
        print('train_X: ' + str(self.train_X.shape))
        print('train_Y: ' + str(self.train_Y.shape))
        print('test_X:  '  + str(self.test_X.shape))
        print('test_Y:  '  + str(self.test_Y.shape))

    ### Prompt loop handling ###
    def handler(self, signum, frame):
        msg = "Do you really want to exit? y/n "
        print(msg, end="", flush=True)
        res = readchar.readchar()
        if res == 'y':
            print("")
            exit(1)
        else:
            print("", end="\r", flush=True)
            print(" " * len(msg), end="", flush=True) # clear the printed line
            print("    ", end="\r", flush=True)

    def start_prompt_loop(self):
        signal.signal(signal.SIGINT, self.handler)
        while True:
            cmd = input("> ")
            if cmd == "raw":
                self.print_raw_data()
            elif cmd == "stat":
                self.compute_stat()
                self.display_stat()
            elif cmd == "rand":
                self.data_process()
                self.print_processed_data()
            elif cmd == "moy":
                self.display_average()
            elif cmd == "compute":
                print("Please wait, this may take a while...")
                self.compute_accuracy()
            elif cmd.split()[0] == "input" and len(cmd.split()) > 1:
                self.test_data(int(cmd.split()[1]))
            else:
                print("Available commands :\n\traw\tprint raw values")
                print("\tstat\tdisplay statistics")
                print("\trand\tshow random numbers example")
                print("\tmoy\tdisplay average number form")
                print("\tcompute\tcompute the train dataset and test it with the test one")

    ### Compute functions ###
    def compute_stat(self):
        for i in range(len(self.train_Y)):
            self.stat[self.train_Y[i]] += 1
        for elem in self.stat:
            print('\t', self.stat.index(elem), '\t|\t', elem)

    def display_stat(self):
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.bar(self.label, self.stat)
        fig.suptitle('How many occurrences of each number are there?')
        plt.show()

    def data_process(self):
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.train_X[i], cmap=plt.get_cmap('gray'))

    def print_processed_data(self):
        plt.show()

    def average(self, digit):
        tab_result = []
        nbr = 0
        for i in range(28):
            tmp = []
            for _ in range(28):
                tmp.append(0)
            tab_result.append(tmp)
        for i in range(len(self.train_Y)):
            if (self.train_Y[i] == digit):
                tab_result += self.train_X[i]
                nbr += 1
        return(np.round(tab_result / nbr))

    def display_average(self):
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.average(i), cmap=plt.get_cmap('gray'))
        plt.show()

    def make_one_line(self, dataset, dataset_one_line):
        for i in range(len(dataset)):
            line = []
            for _ in range(784):
                line.append(0)
            for x in range(len(dataset[i])):
                for y in range(len(dataset[i][x])):
                    line[x * 28 + y] = dataset[i][x][y]
            dataset_one_line.append(line)

    def scale_data(self):
        self.scaler.fit(self.train_X_single_line)
        self.train_X_single_line = self.scaler.transform(self.train_X_single_line)
        self.test_X_single_line = self.scaler.transform(self.test_X_single_line)
        self.classifier.fit(self.train_X_single_line, self.train_Y)
        predict = self.classifier.predict(self.test_X_single_line)
        class_report = classification_report(self.test_Y, predict)
        conf_matrix = confusion_matrix(self.test_Y, predict)
        accu_score = accuracy_score(self.test_Y, predict)
        self.class_report = classification_report(self.test_Y, predict)
        self.conf_matrix = confusion_matrix(self.test_Y, predict)
        self.accu_score = accuracy_score(self.test_Y, predict)

    def print_accuracy(self):
        print(self.class_report)
        print(self.conf_matrix)
        print(self.accu_score)

    def compute_accuracy(self):
        self.make_one_line(self.train_X, self.train_X_single_line)
        self.make_one_line(self.test_X, self.test_X_single_line)
        self.scale_data()

    def predict_input(self, data):
        predict = self.classifier.predict([data])
        return predict

    def test_data(self, index):
        print(datasets.predict_input(datasets.test_X_single_line[index]))
        plt.subplot(330 + 1 + 0)
        plt.imshow(datasets.test_X[index], cmap=plt.get_cmap('gray'))
        plt.show()

    def find_wrong_data(self):
        for i in range(len(self.test_X_single_line)):
            print(datasets.predict_input(datasets.test_X_single_line[i])[0], ' =?= ',  datasets.test_Y[i])
            if (datasets.predict_input(datasets.test_X_single_line[i])[0] != datasets.test_Y[i]):
                plt.subplot(330 + 1 + 0)
                plt.imshow(datasets.test_X[i], cmap=plt.get_cmap('gray'))
                print(datasets.predict_input(datasets.test_X_single_line[i]))
                plt.show()

datasets = DatasetsMatrix()
# print(datasets.train_X)
datasets.make_one_line(datasets.train_X, datasets.train_X_single_line)
datasets.make_one_line(datasets.test_X, datasets.test_X_single_line)
datasets.scale_data()
# datasets.find_wrong_data()


datasets.start_prompt_loop()