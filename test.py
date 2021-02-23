import argparse
import yaml
import cv2
import numpy as np
from core.PaddleOCR import PaddleOCR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='data/params.yaml')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    

    with open(args.params) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    ocr_engine = PaddleOCR(data_dict)
    img = cv2.imread(data_dict['image_dir'])
    result = ocr_engine.ocr(img,
                            det=data_dict['det'],
                            rec=data_dict['rec'],
                            cls=data_dict['use_angle_cls'])
    for box in result:
        pts=np.array(box, np.int32)
        pts=pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0),2)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)