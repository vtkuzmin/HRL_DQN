import os
import sys
import datetime
import pathlib

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Done")
    except OSError:
        print('Error: Creating directory. ' + directory)


def main():
    os.chdir('../experiments/plain DQN/')
    path = os.getcwd()
    print("The current working directory is %s" % path)

    file_path = "experiment " + str(datetime.datetime.now())[:-10]
    createFolder(file_path)

    os.chdir('../../DQN/')

    print("The current working directory is %s" % os.getcwd())




if __name__ == "__main__":
    main()