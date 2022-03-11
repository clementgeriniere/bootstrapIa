from keras.datasets import mnist
from matplotlib import pyplot
from keras import backend as K
import numpy

#loading
class zoidberg:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    tab_dataset = [train_X, train_y, test_X, test_y]

    def print_from_dataset(self, dataset):
        return self.tab_dataset[dataset]

    def data_proc(self):
        for i in range(9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(self.train_X[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    def show_digits_moy(self):
        for i in range(9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(self.moy(i), cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    def show_digits(self):
        for i in range(9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(self.digit(i), cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    def flat_list(self):
        self.test_X = (self.train_X * 10) / 255
        self.test_X *= 10
        self.test_X = numpy.round(self.test_X)
        print(self.test_X[0])

    def distrib(self):
        tab_distrib = []
        for i in range(10):
            tab_distrib.append(0)
        for i in range(len(self.train_y)):
            tab_distrib[self.train_y[i]] += 1
        for i in range(len(tab_distrib)):
            print(i, ':' , tab_distrib[i], end='| ')
        print()
    
    def digit(self, digit):
        for i in range(len(self.train_y)):
            if (self.train_y[i] == digit):
                # pyplot.subplot(330 + 1 + i)
                pyplot.imshow(self.train_X[i], cmap=pyplot.get_cmap('gray'))
                return (self.train_X[i])

    def moy(self, digit):
        tab_result = []
        nbr = 0
        for i in range(28):
            tab_result_j = []
            for j in range(28):
                tab_result_j.append(0)
            tab_result.append(tab_result_j)
        for i in range(len(self.train_y)):
            if (self.train_y[i] == digit):
                tab_result += self.train_X[i]
                nbr += 1
        return(numpy.round(tab_result / nbr))



    def make_one_line(self):
        for i in range(len(self.train_X) - 1):
            line = []
            for _ in range(784):
                line.append(0)
            print(len(self.train_X[i]), len(self.train_X[i]))
            for x in range(len(self.train_X[i])):
                for y in range(len(self.train_X[i][x])):
                    line[x * 28 + y] = self.train_X[i][x][y]
                    print(line)
            self.train_X[i] = line
            print(line)
            

    def reshape(self):
        img_size_x, img_size_y = 28, 28
        batch_size = 128
        num_classes = 10
        epochs = 12
        if K.image_data_format() == 'channels_first':
            print()
            # test.train_X = test.train_X.reshape(test.train_X.shape[0], 1, img_size_x, img_size_y) 
            # test.test_X = test.train_X.reshape(test.test_X.shape[0], 1, img_size_x, img_size_y)
            # input_shape = (1, img_size_x, img_size_y)
        else: 
            test.train_X = test.train_X.reshape(test.train_X.shape, img_size_x, img_size_y, 1) 
            test.test_X = test.test_X.reshape(test.test_X.shape, img_size_x, img_size_y, 1)
            input_shape = (img_size_x, img_size_y, 1)


test = zoidberg()
# test.data_proc()
test.distrib()
# print(test.digit(0))
# pyplot.show()
# test.show_digits()
# test.show_digits_moy()
test.make_one_line()

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(test.digit(i), cmap=pyplot.get_cmap('gray'))

# test.flat_list()

# test.train_X = test.train_X.astype('float32')
# test.test_X = test.test_X.astype('float32')
# test.train_X = test.train_X / 255
# test.test_X = test.test_X / 255
# print(test.train_X[0])
# testoutput = []
# for i in range(len(test.train_X[0])
#     for y in range(len(test.train_X[0][i]))
#         testoutput.append(test.train_X[0][i][y])
# print(testoutput)
# for i in range( test.train_X)
# input_shape = (1, img_size_x, img_size_y)
# print('label': test.train_X.head[0])

tab_count_train = [0,0,0,0,0,0,0,0,0,0,0]
tab_count_test = [0,0,0,0,0,0,0,0,0,0,0]

