import cv2
import os

def face_coordinates(path):
    '''
    returns the coordinates of a detected face'''
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    if face.empty():
        raise IOError('Unable to load the face cascade classifier xml file')
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    return face_rects[0]

def crop_face(path, output):
    '''
    returns the cropped image when a face is detected'''
    img = cv2.imread(path)
    x,y,w,h = face_coordinates(path)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(output, crop_img)


if __name__ == '__main__':


    '''
    There has to be a directory of raw_data
    with a dir called dataset_new
    with train and test dirs
    with pictures of faces for class yawning ann no_yawning
    '''
    start = os.getcwd()
    dataset_train_yawn_path = start + '/raw_data/dataset_new/train/yawn'
    dataset_train_no_yawn_path = start + '/raw_data/dataset_new/train/no_yawn'

    dataset_test_yawn_path = start + '/raw_data/dataset_new/test/yawn'
    dataset_test_no_yawn_path = start + '/raw_data/dataset_new/test/no_yawn'

    list_of_path = [dataset_train_yawn_path, dataset_train_no_yawn_path, dataset_test_yawn_path, dataset_test_no_yawn_path]

    count = 0
    for list in list_of_path:

        for (root,dirs,files) in os.walk(list):

            for file in files:
                path_departure = root+'/'+file
                path_arrival = path_departure.replace('dataset_new', 'datset_extracted_face')

                try:
                    crop_face(path_departure, path_arrival)
                    count+=1
                except:
                    pass
            print(count, root.split('/')[-1])
            count = 0
