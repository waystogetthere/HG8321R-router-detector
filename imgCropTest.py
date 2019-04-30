import cv2
import CropImg
import os

# 图片剪切
toCutPath = './data/trainData/'       # 设置待剪切图片文件夹
toSavePath = './data/tempData/'      # 设置剪切后的图片文件夹
imgList = os.listdir(toCutPath)
for imgName in imgList:
    img = cv2.imread(toCutPath + imgName)
    CropImg.cropImg2File(img, imgName.replace(".jpg", ""), toSavePath)
    # input("Press enter key to continue...")


# 批量修改文件名
movie_name = os.listdir('./data/positive/')
for temp in movie_name:
    new_name = temp.replace(" 副本", "")
    os.rename('./data/positive/' + temp,
              './data/positive/' + new_name)


