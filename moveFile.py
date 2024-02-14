import os
import shutil


def move_file(old_path, new_path):
    shutil.move(old_path, new_path)


def findAllFile(base, des):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            # print(fullname)
            # print(f)
            # print(f.split("_")[0])
            path = f.split("_")[2].split(".")[0]
            print(path)
            new_path = des + "\\" + path
            # print(new_path)
            move_file(fullname, new_path)


source = "C:\\Users\\Lawaiian\\PycharmProjects\\dlmi\\images"
dest = "C:\\Users\\Lawaiian\\PycharmProjects\\dlmi\\data"

findAllFile(source, dest)
