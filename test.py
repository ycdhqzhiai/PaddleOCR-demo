import argparse
import yaml
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from core.PaddleOCR import PaddleOCR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='data/params.yaml')
    return parser.parse_args()

def draw_result(img, result):
    img_rec = np.ones_like(img, np.uint8)*255
    img_pil = Image.fromarray(img_rec)
    draw = ImageDraw.Draw(img_pil)
    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 16)
    for info in result:
        bbox, rec_info = info
        pts=np.array(bbox, np.int32)
        pts=pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0),2)
        txt = rec_info[0] + str(rec_info[1])
        draw.text(tuple(pts[0][0]), txt, font=font, fill =(0,255,0))
    bk_img = np.array(img_pil)
    draw_img  = np.hstack([img,bk_img])
    return draw_img

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
    draw_img = draw_result(img, result)
    cv2.imshow("img", draw_img)
    cv2.waitKey(0)