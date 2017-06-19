
import os,csv
import random
from sklearn.model_selection import train_test_split
import re


from model.vgg19_trainable import Train_Flags

def generate_train(dataset_path):

    test_num = 0
    with open('train.csv', 'w') as output:
        with open('train_labels.csv', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                output.write("%s.jpg,%s" % (os.path.join(dataset_path, row[0]), row[1]))
                output.write("\n")



def generate_test(dataset_path, test_batch):

    with open('test.csv', 'w') as output:

        test_path = os.path.join(dataset_path, 'test')

        test_num = 0

        list_test_imgs = []


        for lists in os.listdir(test_path):
            int_list = lists.replace('.jpg', '')
            list_test_imgs.append(int(int_list))

        list_test_imgs.sort()

        for test_img in list_test_imgs:

            path = os.path.join(test_path, str(test_img) + '.jpg')

            output.write("%s,%s" % (path,'none'))
            output.write("\n")

            test_num = test_num + 1
        print test_num, test_batch
        while(test_num % test_batch != 0):
            output.write("%s,%s" % (path,'none'))
            output.write("\n")
            test_num = test_num + 1


# using the parameter in train.py: train/test batch_size
train_flags = Train_Flags()

generate_train(train_flags.dataset_path + '/train')

generate_test(train_flags.dataset_path, train_flags.batch_size)

