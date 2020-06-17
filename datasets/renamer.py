import os
from shutil import copyfile
train_dir = 'KITTI/train_raw'
for dir in os.listdir('KITTI/train_raw'):
    folder_dir = os.path.join(train_dir, dir)
    for file in os.listdir(folder_dir):
        copyfile(os.path.join(folder_dir, file), os.path.join('KITTI/train', file))
        os.rename(os.path.join('KITTI/train', file), os.path.join('KITTI/train', str(10000*int(dir[2:]) + int(file[:6])) + '.png'))
    print('Done: ', folder_dir)