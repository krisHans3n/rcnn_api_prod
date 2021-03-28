import os


a = {'Image_Spring_17_4.gif': [['DBSCAN_CPY_MOVE', 11], ['img_result', 11], ['img_result', 'onfvbotvbsyutbvpdgubiufe4f6g8e8fbevo8ceb4']], 'img_mountains.jpg': [['DBSCAN_CPY_MOVE', 33], ['img_result', '/9j/4AAQSkZJ']]}

b = {'Image_Spring_17_4.gif': [['DBSCAN_CPY_MOVE_2', 22], ['img_result', 22]], 'img_mountains.jpg': [['DBSCAN_CPY_MOVE_2', 166], ['img_result', '/9j/4AAQSkrererbaerbae']]}

c = {'Image_Spring_17_4.gif': 'onfvbotvbsyutbvpdgubiufe4f6g8eevo8ceb4'}

file_keys = ['Image_Spring_17_4.gif', 'img_mountains.jpg']

dicts = [a, b, c]

# Check if response will need some adjusting - some apis may fail to gather the same image
# all_imgs_used = True
# for i in range(0, 3):
#     print()
#     if all(key in dicts[i] for key in file_keys):
#         print("keys are present")
#     else:
#         print("keys are not present")
#         all_imgs_used = False

# TODO: If not all keys are present in the dictionaries update the shared files

new_dict = {}
#  if all_imgs_used:
for f in file_keys:
    new_dict[f] = []
    for j in range(0, 3):
        if f in new_dict.keys():
            for v_ in dicts[j][f]:
                new_dict[f].append(v_)

    print(new_dict)












