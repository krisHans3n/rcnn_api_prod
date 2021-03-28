"""
# NIST2016 Dataset Preprocessing
<font size=3>Generate the coordinates of the bounding box from the Groundtruth mask.<br><br>
    We obtained the NIST dataset by participating in the Media Forensics Challenge 2019. [Link](https://www.nist.gov/itl/iad/mig/media-forensics-challenge-2019-0)<br>Since we signed a agreement, we are unable to provide the NIST 2016 dataset.<br><br>
    In addition, we divide all images into training and test sets according to the ratio of manipulation techniques.<br>
    The number of training and test sets is the same as RGB-N.<br>
    We provide text files for the training and test sets. See Readme.md for more details.<font>
"""

import cv2
import csv
import os
from glob import glob
from sklearn.model_selection import train_test_split

"""
<font size=4， color=#DC143C> We save tampered images and mask in new folders. After the script running, please delete the origenal probe folfer and rename probe_save folder as probe. </font>
"""

nist_path = '../dataset/NC2016_test/'
probe_path = nist_path + 'probe_save/'  # Tampered images save path, not origenal probe folder
mask_path = nist_path + 'mask/'  # Maks save path
if not os.path.exists(probe_path):
    os.makedirs(probe_path)

if not os.path.exists(mask_path):
    os.makedirs(mask_path)


def bounding_box(image, mask, row_data, mani_type):
    box_list = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("contours number：", len(contours))
    image_copy = image.copy()
    contours = sorted(contours, key=lambda i: len(i), reverse=True)

    for i in range(0, len(contours)):
        # We use 175 as a threshold to eliminate noise in the ground truth mask.
        if (len(contours[i])) > 175:
            x, y, w, h = cv2.boundingRect(contours[i])
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            box_list.append(str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + mani_type)
    if len(box_list) == 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        box_list.append(str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + mani_type)
    return box_list


def load_image(image_path):
    img = cv2.imread(nist_path + image_path)
    return img


if __name__ == '__main__':
    # reference file
    csv_reader = csv.reader(open(nist_path + 'reference/manipulation/NC2016-manipulation-ref.csv', encoding='utf-8'))
    for row in csv_reader:
        row_data = row[0].split('|')
        name_str = '_'
        if row_data[0] == 'TaskID':
            continue
        if row_data[7] == 'N':
            mani_tpye = 'authentic'
            print('========', mani_tpye, '============')
            continue
        else:
            mask = load_image(row_data[3])

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
            image = load_image(row_data[2])
            if row_data[11] == 'Y':
                mani_tpye = 'removal'
                box_list = bounding_box(image, mask, row_data, mani_tpye)
            elif row_data[12] == 'Y':
                mani_tpye = 'splice'
                box_list = bounding_box(image, mask, row_data, mani_tpye)
            elif row_data[13] == 'Y':
                mani_tpye = 'copyclone'
                box_list = bounding_box(image, mask, row_data, mani_tpye)
            for i in range(0, len(box_list)):
                name_str = name_str + box_list[i] + '_'

            print(row_data[1])
            cv2.imwrite(probe_path + row_data[1] + name_str.rstrip('_') + ".png", image)
            cv2.imwrite(mask_path + row_data[1] + name_str.rstrip('_') + ".png", mask)
            print('========', mani_tpye, '============')

print('\n\n=======================Done==============================!')

# %%
data_dir = probe_path  # FIXME
ext = 'NC2016*'
cls = ['removal', 'splice', 'copyclone']
filenames = glob(os.path.join(data_dir, ext))
print(len(filenames))
pic_name = []
mani_type = []
for file in filenames:
    content = os.path.splitext(os.path.basename(file))[0].split("_")
    if content[-1] in cls:
        pic_name.append(os.path.splitext(os.path.basename(file))[0])
        mani_type.append(content[-1])
    else:
        print(content[-1])
print("len pic_name: %d" % len(pic_name))
print("len mani_type: %d \n" % len(mani_type))
print('=======Split train and test set========')
pic_name_train, pic_name_test, mani_type_train, mani_type_test = train_test_split(pic_name, mani_type, test_size=0.282,
                                                                                  random_state=0)

print("train set number: %d" % len(pic_name_train))
print("test set number: %d" % len(pic_name_test))

with open(nist_path + '/NIST_train_new_2.txt', 'w') as f:
    for pic in pic_name_train:
        content = pic.split("_")
        if content[-1] in cls:
            content2 = [str(i) for i in content[2:]]
            content3 = ' '.join(content2)
            f.write('%s %s\n' % (pic, content3))

with open(nist_path + '/NIST_test_new_2.txt', 'w') as f:
    for pic in pic_name_test:
        content = pic.split("_")
        if content[-1] in cls:
            content2 = [str(i) for i in content[2:]]
            content3 = ' '.join(content2)
            f.write('%s %s\n' % (pic, content3))

print('=============Split over==============')
