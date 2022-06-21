#coding:utf-8
from PIL import Image
import numpy as np
import math
import time
import os
import sys
from skimage.metrics import structural_similarity as ssim


ImageWidth = 256
ImageHeight = 256
FILE_PATH = r"F:\1-code\MPA-master\Result\%d_%d\Output" % (ImageWidth,ImageHeight)
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

def SaveResult(str):
    try:
        fname = time.strftime("%Y%m%d", time.localtime()) 
        f2 = open(FILE_PATH + '\\0_result' + fname + '.txt','a+')
        f2.read()
        f2.write('\n')
        f2.write(str)
        f2.write('\n')
    finally:
        if f2:
            f2.close()
    return 0

def PSNR(image_array1,image_array2):
    assert(np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert(n > 0)
    MSE = 0.0
    for i in range(0,n):
        MSE+=math.pow(int(image_array1[i]) - int(image_array2[i]),2)
    MSE = MSE / n 
    if MSE > 0:
        rtnPSNR = 10 * math.log10(255 * 255 / MSE)
    else:
        rtnPSNR = 100
    return rtnPSNR

def dec2bin_higher_ahead(x,n):
    b_array1 = np.zeros(n)
    for i in range(0,n ,1):
        b_array1[i] = int(x % 2)
        x = x // 2
    b_array2 = np.zeros(n)
    for i in range(0,n ,1):
        b_array2[i] = b_array1[n - i] # n-1-i ？
    return b_array2

def dec2bin_lower_ahead(y,n):
    x = y
    b_array1 = np.zeros(n)
    for i in range(0,n ,1):
        b_array1[i] = int(x % 2)
        x = x // 2

    return b_array1



# # ALGORITHM: EMD06
def EMD06(image_array,secret_string,n=2,k=3,image_file_name=''):
    assert(n == 2 or n == 4 or n == 8 or n == 16 or n == 32 or n == 64)
    moshu = 2 * n + 1
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    fG_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * pixels_group[i,j]
        fG_array[i] = fG % moshu
    m = int(math.log((2 * n + 1),2))
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,m):
            if(i * m + j < s_data.size):
                 secret_group[i,j] = s_data[i * m + j]
        i = i + 1
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
            d_array[i] = d
    embedded_pixels_group = pixels_group.copy()
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fG = fG_array[i]
        j = int(d - fG) % moshu
        if (j > 0): #如果为0的话，则不进行修改
            if (j <= n) :
                embedded_pixels_group[i , j - 1]+=1
            else:
                embedded_pixels_group[i ,(2 * n + 1 - j) - 1]+=-1
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * embedded_pixels_group[i,j]
        recover_d_array[i] = fG % moshu
    assert(int((recover_d_array - d_array).sum()) == 0)
    num_pixels_changed = num_secret_groups * n
    #-----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1,imgpsnr2)

    def SSIM(image_array1, image_array2):
        assert (np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert (n > 0)
        ux = 0.0
        uy = 0.0
        cx2 = 0.0
        cy2 = 0.0
        cxy = 0.0
        c1 = 0.0
        c2 = 0.0
        L = 255
        k1 = 0.01
        k2 = 0.03
        ssim = 0
        for i in range(0, n):
            ux += int(image_array1[i])
        ux = ux / n
        for i in range(0, n):
            uy += int(image_array2[i])
        uy = uy / n
        for i in range(0, n):
            cx2 += math.pow(int(image_array1[i]) - ux, 2)
        cx2 = cx2 / (n - 1)
        for i in range(0, n):
            cy2 += math.pow(int(image_array2[i]) - uy, 2)
        cy2 = cy2 / (n - 1)
        for i in range(0, n):
            cxy += (int(image_array1[i]) - ux) * (int(image_array2[i]) - uy)
        cxy = cxy / (n - 1)
        c1 = k1 * L * L
        c2 = k2 * L * L
        ssim = ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux * ux + uy * uy + c1) * (cx2 + cy2 + c2))
        return ssim
    ssim = SSIM(imgpsnr1, imgpsnr2)

    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)

    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)

    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q

    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth , ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath,tempfilename) = os.path.split(image_file_name)
    (originfilename,extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(k) + ".png"
    img_out.save(new_file,'png')
    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (originfilename,sys._getframe().f_code.co_name,n,k,num_pixels_changed,psnr,ssim,QI)
    print(str1)
    SaveResult('\n' + str1)

    return 0


# ALGORITHM: IEMD
def IEMD(image_array, secret_string, n=2, k=3, image_file_name=''):
    n = 2
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n):
            if (i * n + j < image_array.size):
                pixels_group[i, j] = image_array[i * n + j]
        i = i + 1
    num_secret_groups = math.ceil(secret_string.size / 3)
    secret_group = np.zeros((num_secret_groups, 3))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, 3):
            if (i * 3 + j < s_data.size):
                secret_group[i, j] = s_data[i * 3 + j]
        i = i + 1
    assert (np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)
    d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        d = 0
        for j in range(0, 3):
            d += secret_group[i, j] * (2 ** (3 - 1 - j))
        d_array[i] = d
    embedded_pixels_group = pixels_group.copy()
    for i in range(0, num_secret_groups):
        fe = 1 * pixels_group[i, 0] + 3 * pixels_group[i, 1]
        fe = fe % 8
        if (int(fe) == int(d_array[i])):
            embedded_pixels_group[i, 0] = pixels_group[i, 0]
            embedded_pixels_group[i, 1] = pixels_group[i, 1]
        else:
            fe = 1 * (pixels_group[i, 0] + 1) + 3 * pixels_group[i, 1]
            fe = fe % 8
            if (int(fe) == int(d_array[i])):
                embedded_pixels_group[i, 0] += 1
            else:
                fe = 1 * (pixels_group[i, 0] - 1) + 3 * pixels_group[i, 1]
                fe = fe % 8
                if (int(fe) == int(d_array[i])):
                    embedded_pixels_group[i, 0] += -1
                else:
                    fe = 1 * pixels_group[i, 0] + 3 * (pixels_group[i, 1] + 1)
                    fe = fe % 8
                    if (int(fe) == int(d_array[i])):
                        embedded_pixels_group[i, 1] += 1
                    else:
                        fe = 1 * pixels_group[i, 0] + 3 * (pixels_group[i, 1] - 1)
                        fe = fe % 8
                        if (int(fe) == int(d_array[i])):
                            embedded_pixels_group[i, 1] += -1
                        else:
                            fe = 1 * (pixels_group[i, 0] + 1) + 3 * (pixels_group[i, 1] + 1)
                            fe = fe % 8
                            if (int(fe) == int(d_array[i])):
                                embedded_pixels_group[i, 0] += 1
                                embedded_pixels_group[i, 1] += 1
                            else:
                                fe = 1 * (pixels_group[i, 0] + 1) + 3 * (pixels_group[i, 1] - 1)
                                fe = fe % 8
                                if (int(fe) == int(d_array[i])):
                                    embedded_pixels_group[i, 0] += 1
                                    embedded_pixels_group[i, 1] += -1
                                else:
                                    fe = 1 * (pixels_group[i, 0] - 1) + 3 * (pixels_group[i, 1] + 1)
                                    fe = fe % 8
                                    if (int(fe) == int(d_array[i])):
                                        embedded_pixels_group[i, 0] += -1
                                        embedded_pixels_group[i, 1] += 1
    # -----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        fe = 1 * embedded_pixels_group[i, 0] + 3 * embedded_pixels_group[i, 1]
        recover_d_array[i] = fe % 8
    assert (int((recover_d_array - d_array).sum()) == 0)
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)

    def SSIM(image_array1, image_array2):
        assert (np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert (n > 0)
        ux = 0.0
        uy = 0.0
        cx2 = 0.0
        cy2 = 0.0
        cxy = 0.0
        c1 = 0.0
        c2 = 0.0
        L = 255
        k1 = 0.01
        k2 = 0.03
        ssim = 0
        for i in range(0, n):
            ux += int(image_array1[i])
        ux = ux / n
        for i in range(0, n):
            uy += int(image_array2[i])
        uy = uy / n
        for i in range(0, n):
            cx2 += math.pow(int(image_array1[i]) - ux, 2)
        cx2 = cx2 / (n - 1)
        for i in range(0, n):
            cy2 += math.pow(int(image_array2[i]) - uy, 2)
        cy2 = cy2 / (n - 1)
        for i in range(0, n):
            cxy += (int(image_array1[i]) - ux) * (int(image_array2[i]) - uy)
        cxy = cxy / (n - 1)
        c1 = k1 * L * L
        c2 = k2 * L * L
        ssim = ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux * ux + uy * uy + c1) * (cx2 + cy2 + c2))
        return ssim

    ssim = SSIM(imgpsnr1, imgpsnr2)
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2 = 0
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q

    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr,ssim,QI)
    print(str1)
    SaveResult('\n' + str1)

    return 0

# ALGORITHM: JY_2009
def JY(image_array, secret_string, n=1, k=2, image_file_name=''):
    num_pixel_groups = image_array.size
    assert (int(k) <= 2)
    moshu = 2 * k + 1
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups, k))
    for i in range(0, num_secret_groups, 1):
        for j in range(0, k, 1):
            if (i * k + j < secret_string.size):
                secret_group[i, j] = s_data[i * k + j]
    assert (num_pixel_groups > num_secret_groups)
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        d = 0
        for j in range(0, k, 1):
            d += secret_group[i, j] * (2 ** j)  # 将secret视为低位在前
        secret_d_array[i] = d
    embedded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0, num_secret_groups):
        x = 0
        if (pixels_group[i] >= 0) and (pixels_group[i] <= moshu):
            for x in range(0, moshu, 1):
                fg = (pixels_group[i] + x) % moshu
                if int(fg) == int(secret_d_array[i]):
                    embedded_pixels_group[i] = pixels_group[i] + x
                    break
        else:
            if (pixels_group[i] >= 255 - moshu) and (pixels_group[i] <= 255):
                for x in range(-1 * moshu + 1, 1, 1):
                    fg = (pixels_group[i] + x) % moshu
                    if int(fg) == int(secret_d_array[i]):
                        embedded_pixels_group[i] = pixels_group[i] + x
                        break
            else:
                for x in range(-1 * moshu, moshu + 1, 1):
                    fg = (pixels_group[i] + x) % moshu
                    if int(fg) == int(secret_d_array[i]):
                        embedded_pixels_group[i] = pixels_group[i] + x
                        break
        tmp1 = embedded_pixels_group[i] % moshu
        tmp2 = int(secret_d_array[i])
        assert (tmp1 == tmp2)
    # -----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        recover_d_array[i] = embedded_pixels_group[i] % moshu
    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    num_pixels_changed = num_secret_groups
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]  # 取前面的pixel
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    def SSIM(image_array1, image_array2):
        assert (np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert (n > 0)
        ux = 0.0
        uy = 0.0
        cx2 = 0.0
        cy2 = 0.0
        cxy = 0.0
        c1 = 0.0
        c2 = 0.0
        L = 255
        k1 = 0.01
        k2 = 0.03
        ssim = 0
        for i in range(0, n):
            ux += int(image_array1[i])
        ux = ux / n
        for i in range(0, n):
            uy += int(image_array2[i])
        uy = uy / n
        for i in range(0, n):
            cx2 += math.pow(int(image_array1[i]) - ux, 2)
        cx2 = cx2 / (n - 1)
        for i in range(0, n):
            cy2 += math.pow(int(image_array2[i]) - uy, 2)
        cy2 = cy2 / (n - 1)
        for i in range(0, n):
            cxy += (int(image_array1[i]) - ux) * (int(image_array2[i]) - uy)
        cxy = cxy / (n - 1)
        c1 = k1 * L * L
        c2 = k2 * L * L
        ssim = ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux * ux + uy * uy + c1) * (cx2 + cy2 + c2))
        return ssim
    ssim = SSIM(imgpsnr1, imgpsnr2)

    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2 = 0
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q
    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr, ssim, QI)
    print(str1)
    SaveResult('\n' + str1)
    return 0

# ALGORITHM: KKWW
def KKWW16(image_array,secret_string,n=2,k=3,image_file_name=''):
    def dec_2k_lower_ahead(x,n):
        b_array1 = np.zeros(n)
        for i in range(0,n,1):
            b_array1[i] = int(x) % (2 ** k)
            x = x // (2 ** k)
        return b_array1

    moshu = 2 ** (n * k + 1)
    c_array = np.zeros(n)
    c_array[0] = 1
    for i in range(1,n):
        c_array[i] = (2 ** k) * c_array[i - 1] + 1
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
                 if pixels_group[i,j] <= 2 ** k - 2:
                     pixels_group[i,j] = 2 ** k - 1
                 if pixels_group[i,j] >= 255 - (2 ** k - 2):
                     pixels_group[i,j] = 255 - (2 ** k - 1)
        i = i + 1
    fG_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fG = 0
        for j in range(0,n):
            fG += c_array[j] * pixels_group[i,j]
        fG_array[i] = int(fG) % moshu
    m = n * k + 1
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    for i in range(0,num_secret_groups):
        for j in range(0,m):
            if(i * m + j < secret_string.size):
                 secret_group[i,j] = secret_string[i * m + j]
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
        d_array[i] = d
    embedded_pixels_group = pixels_group.copy()
    diff_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fG = fG_array[i]
        diff_array[i] = int(d - fG) % moshu

    for i in range(0,num_secret_groups):
        diff = diff_array[i]
        if int(diff) > 0:
            if int(diff) == 2 ** (n * k):
                embedded_pixels_group[i,n - 1] = pixels_group[i,n - 1] + (2 ** k - 1)
                embedded_pixels_group[i,0] = pixels_group[i,0] + 1
            else:
                if int(diff) < 2 ** (n * k):
                   d_transfromed = dec_2k_lower_ahead(diff,n)
                   for j in range(n - 1,-1,-1):
                       embedded_pixels_group[i,j] = embedded_pixels_group[i,j] + d_transfromed[j]
                       if j > 0:
                            embedded_pixels_group[i,j - 1] = embedded_pixels_group[i,j - 1] - d_transfromed[j]
                else:
                    if int(diff) > 2 ** (n * k):
                       d_transfromed = dec_2k_lower_ahead((2 ** (n * k + 1)) - diff,n)
                       for j in range(n - 1,-1,-1):
                           embedded_pixels_group[i,j] = embedded_pixels_group[i,j] - d_transfromed[j]
                           if j > 0:
                               embedded_pixels_group[i,j - 1] = embedded_pixels_group[i,j - 1] + d_transfromed[j]
    #-----------------------------------------------------------------------------------
    num_pixels_changed = num_secret_groups * n
    #-----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fG = 0
        for j in range(0,n):
            fG += c_array[j] * embedded_pixels_group[i,j]
        recover_d_array[i] = int(fG) % moshu

    assert(int((recover_d_array - d_array).sum()) == 0)
    #-----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1,imgpsnr2)

    def SSIM(image_array1, image_array2):
        assert (np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert (n > 0)
        ux = 0.0
        uy = 0.0
        cx2 = 0.0
        cy2 = 0.0
        cxy = 0.0
        c1 = 0.0
        c2 = 0.0
        L = 255
        k1 = 0.01
        k2 = 0.03
        ssim = 0
        for i in range(0, n):
            ux += int(image_array1[i])
        ux = ux / n
        for i in range(0, n):
            uy += int(image_array2[i])
        uy = uy / n
        for i in range(0, n):
            cx2 += math.pow(int(image_array1[i]) - ux, 2)
        cx2 = cx2 / (n - 1)
        for i in range(0, n):
            cy2 += math.pow(int(image_array2[i]) - uy, 2)
        cy2 = cy2 / (n - 1)
        for i in range(0, n):
            cxy += (int(image_array1[i]) - ux) * (int(image_array2[i]) - uy)
        cxy = cxy / (n - 1)
        c1 = k1 * L * L
        c2 = k2 * L * L
        ssim = ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux * ux + uy * uy + c1) * (cx2 + cy2 + c2))
        return ssim
    ssim = SSIM(imgpsnr1, imgpsnr2)

    #----------------------------------QI------------------------------------
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q
    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------

    img_out = img_out.reshape(ImageWidth , ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath,tempfilename) = os.path.split(image_file_name)
    (originfilename,extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(k) + ".png"
    img_out.save(new_file,'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (originfilename,sys._getframe().f_code.co_name,n,k,num_pixels_changed,psnr,ssim,QI)
    print(str1)
    SaveResult('\n' + str1)
    return 0

# ALGORITHM: GEMD13
def GEMD13(image_array, secret_string, n=2, k=3, image_file_name=''):
    def dec2bin_lower_ahead(x, n):
        b_array1 = np.zeros(n + 1)
        for i in range(0, n + 1, 1):
            b_array1[i] = int(x % 2)
            x = x // 2
        return b_array1

    moshu = 2 ** (n + 1)
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n):
            if (i * n + j < image_array.size):
                pixels_group[i, j] = image_array[i * n + j]
        i = i + 1
    fGEMD_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        fGEMD = 0
        for j in range(0, n):
            fGEMD += (2 ** (j + 1) - 1) * pixels_group[i, j]
        fGEMD_array[i] = fGEMD % moshu
    # -----------------------------------------------------------------------------------
    m = n + 1
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups, m))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, m):
            if (i * m + j < s_data.size):
                secret_group[i, j] = s_data[i * m + j]
        i = i + 1
    # -----------------------------------------------------------------------------------
    assert (np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)
    d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        d = 0
        for j in range(0, m):
            d += secret_group[i, j] * (2 ** (m - 1 - j))
        d_array[i] = d
    # -----------------------------------------------------------------------------------
    embedded_pixels_group = pixels_group.copy()
    diff_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        d = d_array[i]
        fGEMD = fGEMD_array[i]
        assert (fGEMD < 33)
        diff_array[i] = int(d - fGEMD) % moshu

    for i in range(0, num_secret_groups):
        diff = int(diff_array[i])
        if (diff == 2 ** n):
            embedded_pixels_group[i, 0] = pixels_group[i, 0] + 1
            embedded_pixels_group[i, n - 1] = pixels_group[i, n - 1] + 1
        if (diff > 0) and (diff < 2 ** n):
            b_array = dec2bin_lower_ahead(diff, n)
            for j in range(n, 0, -1):  # 倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] + 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] - 1
        if (diff > 2 ** n) and (diff < 2 ** (n + 1)):
            b_array = dec2bin_lower_ahead(2 ** (n + 1) - diff, n)
            for j in range(n, 0, -1):  # 倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] - 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] + 1

    # -----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        fGEMD = 0
        for j in range(0, n):
            fGEMD += (2 ** (j + 1) - 1) * embedded_pixels_group[i, j]
        recover_d_array[i] = fGEMD % moshu

    assert (int((recover_d_array - d_array).sum()) == 0)
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    #----------------------------------QI------------------------------------
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q

    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    img1 = image_array.reshape(ImageWidth, ImageHeight)
    img2 = np.array(Image.open(new_file))

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr,ssim(img1, img2),QI)
    print(str1)
    SaveResult('\n' + str1)
    return 0

# ALGORITHM: SB19
def SB19(image_array, secret_string, n=1, k=3, image_file_name=''):
    n = 1
    num_pixel_groups = image_array.size
    # -----------------------------------------------------------------------------------
    moshu = k * k
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups, k))
    for i in range(0, num_secret_groups, 1):
        for j in range(0, k, 1):
            if (i * k + j < secret_string.size):
                secret_group[i, j] = s_data[i * k + j]

    assert (num_pixel_groups > num_secret_groups)
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups, 1):
        d = 0
        for j in range(0, k, 1):
            d += secret_group[i, j] * (2 ** j)
        secret_d_array[i] = d
    # -----------------------------------------------------------------------------------
    embedded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0, num_secret_groups):
        x = 0
        for x in range(-1 * math.floor(moshu / 2), math.floor(moshu / 2) + 1, 1):
            f = (pixels_group[i] + x) % moshu
            if int(f) == int(secret_d_array[i]):
                if pixels_group[i] + x < 0:
                    embedded_pixels_group[i] = pixels_group[i] + x + moshu
                else:
                    embedded_pixels_group[i] = pixels_group[i] + x

                break
        tmp1 = embedded_pixels_group[i] % moshu
        tmp2 = int(secret_d_array[i])
        assert (tmp1 == tmp2)
    # -----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        recover_d_array[i] = embedded_pixels_group[i] % moshu

    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]

    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    #----------------------------------QI------------------------------------
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q

    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')
    img1 = image_array.reshape(ImageWidth, ImageHeight)
    img2 = np.array(Image.open(new_file))

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr, ssim(img1,img2),QI)
    print(str1)
    SaveResult('\n' + str1)
    return 0

# ALGORITHM: MOPNA
def MOPNA(image_array, secret_string, n=2, k=3, image_file_name=''):
    n = 2
    moshu = 2 ** (n * k + 1)
    c0 = 3
    c1 = 11

    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))

    for i in range(0, num_pixel_groups, 1):
        for j in range(0, n, 1):
            if i * n + j < image_array.size:
                pixels_group[i, j] = image_array[i * n + j]
    fG_array = np.zeros((num_pixel_groups))
    for i in range(0, num_pixel_groups, 1):
        fG_array[i] = (c0 * pixels_group[i, 0] + c1 * pixels_group[i, 1]) % moshu

    num_BitsPerPixelsGoup = n * k + 1
    num_secret_groups = math.ceil(secret_string.size / num_BitsPerPixelsGoup)
    secret_group = np.zeros((num_secret_groups, num_BitsPerPixelsGoup))
    secret_string_copy = secret_string.copy()
    for i in range(0, num_secret_groups, 1):
        for j in range(0, num_BitsPerPixelsGoup, 1):
            if i * num_BitsPerPixelsGoup + j < secret_string.size:
                secret_group[i, j] = secret_string_copy[i * num_BitsPerPixelsGoup + j]

    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups, 1):
        for j in range(0, num_BitsPerPixelsGoup, 1):
            secret_d_array[i] += (2 ** j) * secret_group[i, j]

    # -----------------------------------------------------------------------------------
    def CPV(image_array1, image_array2):
        assert (np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert (n > 0)
        MSE = 0
        P2 = 0
        for i in range(0, n):
            MSE += math.pow(image_array1[i] - image_array2[i], 2)
            P2 += math.pow(image_array1[i], 2)
        if (MSE > 0) and (int(P2) > 0):
            rtnCSNR = 10 * math.log10(P2 / MSE)
        else:
            rtnCSNR = 100
        return rtnCSNR
    # -----------------------------------------------------------------------------------
    assert (num_pixel_groups > num_secret_groups)
    embedded_pixels_group = pixels_group.copy()
    for i in range(0, num_secret_groups, 1):
        tmp_MaxPsnr = -120000
        tmp_SlectedIndex0 = -129
        tmp_SlectedIndex1 = -129
        tmp_P = np.zeros(2)
        for j0 in range(-1 * moshu, moshu, 1):
            for j1 in range(-1 * moshu, moshu, 1):
                tmp_P[0] = (pixels_group[i, 0] + j0)
                tmp_P[1] = (pixels_group[i, 1] + j1)
                if (int(tmp_P[0]) >= 0 and int(tmp_P[1]) >= 0):
                    tmp = (c0 * tmp_P[0] + c1 * tmp_P[1]) % moshu
                    if (int(secret_d_array[i]) == int(tmp)):
                        tmp1 = CPV(pixels_group[i], tmp_P)
                        if tmp1 > tmp_MaxPsnr:
                            tmp_MaxPsnr = tmp1
                            tmp_SlectedIndex0 = j0
                            tmp_SlectedIndex1 = j1
        assert (tmp_SlectedIndex0 > -129)
        assert (tmp_SlectedIndex1 > -129)
        embedded_pixels_group[i, 0] = pixels_group[i, 0] + tmp_SlectedIndex0
        embedded_pixels_group[i, 1] = pixels_group[i, 1] + tmp_SlectedIndex1

        tmp = 0
        for j in range(0, num_BitsPerPixelsGoup, 1):
            tmp = (c0 * (embedded_pixels_group[i, 0]) + c1 * (embedded_pixels_group[i, 1])) % moshu
        assert (int((tmp - secret_d_array[i]).sum()) == 0)

    num_pixels_changed = num_secret_groups * 2
    # -----------------------------------------------------------------------------------
    recover_d_array = np.zeros(num_secret_groups)  # 待嵌入的secret值
    for i in range(0, num_secret_groups, 1):
        for j in range(0, num_BitsPerPixelsGoup, 1):
            tmp = (c0 * (embedded_pixels_group[i, 0]) + c1 * (embedded_pixels_group[i, 1])) % moshu
            recover_d_array[i] = tmp
    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)

    #----------------------------------QI------------------------------------
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q

    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')
    img1 = image_array.reshape(ImageWidth, ImageHeight)
    img2 = np.array(Image.open(new_file))

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr, ssim(img1,img2),QI)
    print(str1)
    SaveResult('\n' + str1)
    return

# ALGORITHM: MPA
def MPA22(image_array, secret_string, n=2, k=3, image_file_name=''):
    def dec_2k_lower_ahead(x, n):
        b_array1 = np.zeros(n)
        for i in range(0, n, 1):
            b_array1[i] = int(x) % (2 ** k)
            x = x // (2 ** k)
        return b_array1
    n1 = int(n / 2)
    n2 = int(n / 2)
    moshu = 2 ** (n1 * k + 1)
    c_array = np.zeros(n1)
    c_array[0] = 1
    for i in range(1, n1):
        c_array[i] = (2 ** k) * c_array[i - 1] + 1

    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n):
            if (i * n + j < image_array.size):
                pixels_group[i, j] = image_array[i * n + j]
                if pixels_group[i, j] <= 2 ** k - 2:
                    pixels_group[i, j] = 2 ** k - 1
                if pixels_group[i, j] >= 255 - (2 ** k - 2):
                    pixels_group[i, j] = 255 - (2 ** k - 1)
        i = i + 1
    n1_pixels_group = np.zeros((num_pixel_groups, n1))
    n2_pixels_group = np.zeros((num_pixel_groups, n2))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n1):
            n1_pixels_group[i] = pixels_group[i][0:n1]
        i = i + 1
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n2):
            n2_pixels_group[i] = pixels_group[i][-(n2):]
        i = i + 1

    n1_fG_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        fG = 0
        for j in range(0, n1):
            fG += c_array[j] * n1_pixels_group[i, j]
        n1_fG_array[i] = int(fG) % moshu

    n2_fG_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        fG = 0
        for j in range(0, n2):
            fG += c_array[j] * n2_pixels_group[i, j]
        n2_fG_array[i] = int(fG) % moshu
    # -----------------------------------------------------------------------------------
    m = n * k + 2
    m1 = int(m / 2)
    m2 = int(m / 2)

    num_secret_groups = math.ceil(s_data.size / m)
    secret_group = np.zeros((num_secret_groups, m))
    for i in range(0, num_secret_groups):
        for j in range(0, m):
            if (i * m + j < s_data.size):
                secret_group[i, j] = s_data[i * m + j]
    n1_secret_group = np.zeros((num_secret_groups, m1))
    n2_secret_group = np.zeros((num_secret_groups, m2))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, m1):
            n1_secret_group[i] = secret_group[i][0:m1]
        i = i + 1
    i = 0
    while (i < num_secret_groups):
        for j in range(0, m2):
            n2_secret_group[i] = secret_group[i][-(m2):]
        i = i + 1

    n1_d_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        d = 0
        for j in range(0, m1):
            d += n1_secret_group[i, j] * (2 ** (m1 - 1 - j))
        n1_d_array[i] = d

    n2_d_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        d = 0
        for j in range(0, m2):
            d += n2_secret_group[i, j] * (2 ** (m2 - 1 - j))
        n2_d_array[i] = d
    # -----------------------------------------------------------------------------------
    n1_embedded_pixels_group = n1_pixels_group.copy()
    n2_embedded_pixels_group = n2_pixels_group.copy()
    n1_diff_array = np.zeros(num_secret_groups)
    for i in range(num_pixel_groups):
        d = n1_d_array[i]
        fG = n1_fG_array[i]
        n1_diff_array[i] = int(d - fG) % moshu

    for i in range(num_secret_groups):
        diff = n1_diff_array[i]
        if int(diff) > 0:
            if int(diff) == 2 ** (n1 * k):
                n1_embedded_pixels_group[i, n1 - 1] = n1_pixels_group[i, n1 - 1] + (2 ** k - 1)
                n1_embedded_pixels_group[i, 0] = n1_pixels_group[i, 0] + 1
            else:
                if int(diff) < 2 ** (n1 * k):
                    d_transfromed = dec_2k_lower_ahead(diff, n1)
                    for j in range(n1 - 1, -1, -1):
                        n1_embedded_pixels_group[i, j] = n1_embedded_pixels_group[i, j] + d_transfromed[j]
                        if j > 0:
                            n1_embedded_pixels_group[i, j - 1] = n1_embedded_pixels_group[i, j - 1] - d_transfromed[j]
                else:
                    if int(diff) > 2 ** (n1 * k):
                        d_transfromed = dec_2k_lower_ahead((2 ** (n1 * k + 1)) - diff, n1)
                        for j in range(n1 - 1, -1, -1):
                            n1_embedded_pixels_group[i, j] = n1_embedded_pixels_group[i, j] - d_transfromed[j]
                            if j > 0:
                                n1_embedded_pixels_group[i, j - 1] = n1_embedded_pixels_group[i, j - 1] + d_transfromed[j]

    n2_diff_array = np.zeros(num_secret_groups)
    for i in range(0, num_pixel_groups):
        d = n2_d_array[i]
        fG = n2_fG_array[i]
        n2_diff_array[i] = int(d - fG) % moshu

    for i in range(num_secret_groups):
        diff = n2_diff_array[i]
        if int(diff) > 0:
            if int(diff) == 2 ** (n2 * k):
                n2_embedded_pixels_group[i, n2 - 1] = n2_pixels_group[i, n2 - 1] + (2 ** k - 1)
                n2_embedded_pixels_group[i, 0] = n2_pixels_group[i, 0] + 1
            else:
                if int(diff) < 2 ** (n2 * k):
                    d_transfromed = dec_2k_lower_ahead(diff, n2)
                    for j in range(n2 - 1, -1, -1):
                        n2_embedded_pixels_group[i, j] = n2_embedded_pixels_group[i, j] + d_transfromed[j]
                        if j > 0:
                            n2_embedded_pixels_group[i, j - 1] = n2_embedded_pixels_group[i, j - 1] - d_transfromed[j]
                else:
                    if int(diff) > 2 ** (n2 * k):
                        d_transfromed = dec_2k_lower_ahead((2 ** (n2 * k + 1)) - diff, n2)
                        for j in range(n2 - 1, -1, -1):
                            n2_embedded_pixels_group[i, j] = n2_embedded_pixels_group[i, j] - d_transfromed[j]
                            if j > 0:
                                n2_embedded_pixels_group[i, j - 1] = n2_embedded_pixels_group[i, j - 1] + d_transfromed[j]
    embedded_pixels_group = np.concatenate((n1_embedded_pixels_group,n2_embedded_pixels_group), axis=1)
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    img_out1 = embedded_pixels_group.flatten()
    img_out = img_out1[:ImageWidth * ImageHeight]  # 取前面的pixel
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    #----------------------------------QI------------------------------------
    sum_P = 0
    for i in range(ImageWidth * ImageHeight):
        sum_P += image_array[i]
    Average_P = sum_P / (ImageWidth * ImageHeight)
    sum_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum_Q += img_out1[i]
    Average_Q = sum_Q / (ImageWidth * ImageHeight)
    sum1 = 0
    for i in range(ImageWidth * ImageHeight):
        sum1_P1 = image_array[i] - Average_P
        sum1_Q1 = img_out1[i] - Average_Q
        sum1 += sum1_P1 * sum1_Q1
    sum2_P = 0
    sum2_Q = 0
    for i in range(ImageWidth * ImageHeight):
        sum2_P += (image_array[i] - Average_P) * (image_array[i] - Average_P)
        sum2_Q += (img_out1[i] - Average_Q) * (img_out1[i] - Average_Q)
    sum2 = sum2_P + sum2_Q
    denominator = sum2 * ((Average_P * Average_P) + (Average_Q * Average_Q))
    numerator = 4 * Average_P * Average_Q * sum1
    QI = numerator / denominator
    #------------------------------------------------------------------------
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    img1 = image_array.reshape(ImageWidth, ImageHeight)
    img2 = np.array(Image.open(new_file))

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.4f,SSIM: %.4f,QI: %.4f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr, ssim(img1, img2),QI)
    print(str1)
    SaveResult('\n' + str1)
    return 0

def proof():
    n = 2
    k = 5
    moshu = 2 ** (n * k + 1)
    c0 = 3
    c1 = 11
    outlist = []
    for g0 in range(0,256):
        for g1 in range(0,256):
            d = (c0 * g0 + c1 * g1) % moshu
            if d not in outlist:
                outlist.append(d)
    outlist.sort()
    assert(len(outlist) == moshu)
    return

np.random.seed(1203)
s_data = np.random.randint(0,2,262144)
path = os.getcwd()
path = path + r"\OriginalPictures\%d_%d" % (ImageWidth,ImageHeight)
SaveResult('start')
for file in os.listdir(path):
    file_path = os.path.join(path, file)  
    if os.path.isfile(file_path):
        print(file_path)
        img = Image.open(file_path,"r")
        img = img.convert('L')
        img_array1 = np.array(img)
        img_array2 = img_array1.reshape(img_array1.shape[0] * img_array1.shape[1])
        img_array3 = img_array1.flatten()

        EMD06(img_array3,s_data,2,3,file_path)
        EMD06(img_array3,s_data,4,3,file_path)
        IEMD(img_array3,s_data,2,3,file_path)
        JY(img_array3,s_data,1,2,file_path)
        GEMD13(img_array3, s_data, 2, 3, file_path)
        KKWW16(img_array3,s_data,2,2,file_path)
        SB19(img_array3,s_data,1,2,file_path)
        MOPNA(img_array3,s_data,2,1,file_path)
        MPA22(img_array3,s_data,2,2,file_path)
SaveResult('end')
time.sleep(10)
