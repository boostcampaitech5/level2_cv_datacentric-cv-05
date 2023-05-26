
import json
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
from PIL import Image

FILE_NAMES = [
    "drp.en_ko.in_house.deepnatural_003442.jpg", 
    "drp.en_ko.in_house.deepnatural_003707.jpg", 
    "drp.en_ko.in_house.deepnatural_003871.jpg", 
    "drp.en_ko.in_house.deepnatural_003661.jpg", 
    "drp.en_ko.in_house.deepnatural_003388.jpg", 
    "drp.en_ko.in_house.deepnatural_003021.jpg", 
    "drp.en_ko.in_house.deepnatural_003955.jpg", 

    "drp.en_ko.in_house.deepnatural_003031.jpg", 
    "drp.en_ko.in_house.deepnatural_003417.jpg", 
    "drp.en_ko.in_house.deepnatural_003293.jpg",

    "drp.en_ko.in_house.deepnatural_003535.jpg", 
    "drp.en_ko.in_house.deepnatural_001892.jpg", 
    "drp.en_ko.in_house.deepnatural_002415.jpg", 
    "drp.en_ko.in_house.deepnatural_004006.jpg",
    ]

#JSON_FILE_NAME_LIST = ['output1_0.88_0.93.json', 'output2_0.92_0.79.json']

def save_eda_image(
        JSON_FILE_NAME,
        IMAGE_NUM,
        IMAGE_FILE_NAME,
        IMAGE_OUTPUT_NAME_HEAD = 'output1_',
        USE_FILE_NAME = True,
    ) :
    """
    JSON_FILE_NAME : json 파일 경로
    IMAGE_NUM : json 파일 안에서 몇변째 이미지를 뽑아보고 싶을 때 쓰는거 (파일 이름 모를 때)
    IMAGE_FILE_NAME : 이미지 파일 이름을 알 때
    IMAGE_OUTPUT_NAME_HEAD : 박스 친 이미지 파일 저장 경로 (head + 파일 원본 이름) 이렇게 저장됨.
    USE_FILE_NAME : 파일 이름을 알아서 파일 이름으로 출력할 때 이거 True 로 설정할 것
    """

    # Open the JSON file for reading
    with open(JSON_FILE_NAME, 'r') as file:
        # Load the JSON data into a dictionary
        data_dict = json.load(file)

    data_dict = data_dict['images']

    image_names = list(data_dict.keys())

    if USE_FILE_NAME :
        image_name = IMAGE_FILE_NAME
    else :
        image_name = image_names[IMAGE_NUM]
        
    image_name

    bbox_names = list(data_dict[image_name]['words'].keys())

    bbox_list = []
    for bbox_name in bbox_names :
        bbox_list.append(data_dict[image_name]['words'][bbox_name]['points'])

    import cv2
    import numpy as np

    image_path = '/opt/ml/input/data/medical/img/test/' + image_name

    # 이미지 로드
    image = cv2.imread(image_path)

    # plt.axis(False)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    points = np.array(bbox_list)

    # 다각형 그리기
    cv2.polylines(image, np.int32(points), isClosed=True, color=(0,0,200), thickness=1)

    # 이미지 출력
    plt.axis(False)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # 이미지 저장할 경로와 파일명
    output_path = IMAGE_OUTPUT_NAME_HEAD + image_name
    print(output_path)

    # 이미지 저장
    cv2.imwrite(output_path, image)

    print("이미지 저장이 완료되었습니다.")

for fn in FILE_NAMES :
    save_eda_image('output1_0.88_0.93.json',0,fn,'./output2/',True)

