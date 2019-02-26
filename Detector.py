# HOG检测人
import cv2
import numpy as np

def is_inside(o, i):
    # 判断矩形o是不是在i矩形中，其中 o：矩形o  (x,y,w,h)，i：矩形i  (x,y,w,h)
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(img, person):
    # 在img图像上绘制矩形框person，参数person表示目标的box信息
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


def detect_test():
    # 检测代码，调用python库函数
    img = cv2.imread('/home/yhq/PycharmProjects/Hog_Svm/image/people.jpg')
    '''
    创建HOG描述符对象 
    winSize = (64,128)，blockSize = (16,16)，blockStride = (8,8)，cellSize = (8,8)，nbins = 9  
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)  
    默认9个bin，一个block 2*2 cell,共((64-16)/8 +1)×((128-16)/8+1)个block
    所以特征维度：9×4×((64-16)/8 +1)×((128-16)/8+1)=3780  
    '''
    hog = cv2.HOGDescriptor()  #3780维，多一维是一维偏移，便于后期分类器计算
    # hist = hog.compute(img[0:128,0:64])   计算一个检测窗口的维度
    # print(hist.shape)
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    # print('detector', type(detector), detector.shape)
    hog.setSVMDetector(detector)
    # 多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
    found, w = hog.detectMultiScale(img)
    # 过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            # r在q内？
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)
    for person in found_filtered:
        draw_person(img, person)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detect_test()