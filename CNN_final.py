import cv2
import numpy as np
import math
def apply_padding(image, padding):
    height, width, channels = image.shape
    if padding == 0:
        # 外面多一層白色
        padded_image = np.zeros((height + 2, width + 2, channels), dtype=image.dtype)
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    padded_image[i + 1, j + 1, c] = image[i, j, c]
    elif padding == 1:
        # 外面多一層黑色
        padded_image = np.full((height + 2, width + 2, channels), 255, dtype=image.dtype)
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    padded_image[i + 1, j + 1, c] = image[i, j, c]
    elif padding == -1:
        # 不做padding
        padded_image = image
    return padded_image

def avg_conv(image, n, m, kernel, bias, padding):
    image = apply_padding(image, padding)
    # get input size
    height, width, channels = image.shape
    # define output
    new_image = np.zeros((height - m + 1, width - n + 1, channels), dtype=np.float32)
    # cnn
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            for k in range(channels):
                sum = 0
                for ke_y in range(m):
                    for ke_x in range(n):
                        sum += image[i + ke_y, j + ke_x, k] * kernel[ke_y, ke_x, k]
                new_image[i, j, k] = sum + bias
    return new_image

def sobel_conv(image, n, m, kernel_y, kernel_x ,bias, padding):
    image = apply_padding(image, padding)
    # get input size
    height, width, channels = image.shape
    # turn to grayscale
    if channels == 3:
        grayscale_image = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                r, g, b = image[i, j]
                grayscale_image[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
        image = grayscale_image
        image.shape = image.shape + (1,)
    # define output
    new_image = np.zeros((height - m + 1, width - n + 1, 1), dtype=np.float32)
    # cnn for x
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            for k in range(channels):
                sum = 0
                for ke_y in range(m):
                    for ke_x in range(n):
                        sum += image[i + ke_y, j + ke_x, 0] * kernel_x[ke_y, ke_x]
                new_image[i, j, 0] = abs(sum) + bias
    # cnn for y
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            for k in range(channels):
                sum = 0
                for ke_y in range(m):
                    for ke_x in range(n):
                        sum += image[i + ke_y, j + ke_x, 0] * kernel_y[ke_y, ke_x]
                new_image[i, j, 0] = abs(sum) + bias
    return new_image

def gaussian_conv(image, n, m, kernel, bias, padding):
    image = apply_padding(image, padding)
    # get input size
    height, width, channels = image.shape
    # define output
    new_image = np.zeros((height - m + 1, width - n + 1, channels), dtype=np.float32)
    # cnn
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            for k in range(channels):
                sum = 0
                for ke_y in range(m):
                    for ke_x in range(n):
                        sum += image[i + ke_y, j + ke_x, k] * kernel[ke_y, ke_x]
                new_image[i, j, k] = sum + bias
    return new_image
    
def pool(input, size, stride, type):
    # get input size
    height, width, channels = input.shape
    # count output size
    out_height = (height - size) // stride + 1
    out_width = (width - size) // stride + 1
    # define output
    output = np.zeros((out_height, out_width, channels), dtype=np.float32)
    # cnn
    for i in range(out_height):
        for j in range(out_width):
            for k in range(channels):
                ma = -100000000
                # max pooling
                if(type == 0):
                    for y in range(size):
                        for x in range(size):
                            ma = max(ma, input[i * stride + y, j * stride + x, k])
                    output[i, j, k] = ma
                # avg pooling
                elif(type == 1):
                    ma = 0
                    for y in range(size):
                        for x in range(size):
                            ma  += input[i * stride + y, j * stride + x, k]
                    output[i, j, k] =  ma / (size * size)
    return output

def avg(n, m, rgb):
    # 創造一個 n*m*rgb 的 kernel
    kernel = np.ones((n, m, rgb), dtype=np.float32) / (n * m)
    return kernel

def sobel_x(n,m):
    # 定義卷積核
    #[y][x][z]
    kernel = np.array([[1, 0 , -1],
                    [2 , 0 , -2],
                    [1 , 0 , -1]], dtype=np.float32)
    return kernel

def sobel_y(n,m):
    # 定義卷積核
    #[y][x][z]
    kernel = np.array([[1, 2 , 1],
                    [0 , 0 , 0],
                    [-1 , -2 , -1]], dtype=np.float32)
    return kernel

def gaussian_filter(n,m,sigma):
    #定義gaussian filter, sigma為1
    kernel = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            x = i - n // 2
            y = j - m // 2
            kernel[i, j] = math.e ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    return kernel

def add_signature(a, b):
    if b.shape[2] == 1:
        return add_signature_black(a, b)
    if b.shape[2] == 3:
        return add_signature_rgb(a, b)

#將a圖片簽名檔放到b圖片右下角
def add_signature_rgb(a, b):
    # 獲取圖片的尺寸
    height_a, width_a, channels_a = a.shape
    height_b, width_b, channels_b = b.shape
    # 簽名檔的位置(右下角)
    x = width_b - width_a
    y = height_b - height_a
    # 將簽名檔放到b圖片右下角
    for i in range(height_a):
        for j in range(width_a):
            for k in range(3):
                if a[i, j, k] < 254:  # 將黑色簽名檔部分放到b圖片(簽名檔為白底黑字)
                    b[y + i, x + j, k] = a[i, j, k]
    return b
#將a圖片簽名檔放到b圖片右下角
def add_signature_black(a, b):
    # 獲取圖片的尺寸
    height_a, width_a, channels_a = a.shape
    height_b, width_b, channels_b = b.shape
    # 簽名檔的位置(右下角)
    x = width_b - width_a
    y = height_b - height_a
    # 將簽名檔放到b圖片右下角
    for i in range(height_a):
        for j in range(width_a):
            if a[i, j, 0] < 254:  # 將黑色簽名檔部分放到b圖片(簽名檔為白底黑字)
                b[y + i, x + j, 0] = 255
    return b

if __name__ == "__main__":
    n = 3
    m = 3
    name = 'patrick'
    image = cv2.imread(f'{name}.jpg')
    if(len(image.shape) == 2):
        image.shape = image.shape + (1,)
    avg_kernel = avg(n,m, image.shape[2])
    sobel_x_kernel = sobel_x(n,m)
    sobel_y_kernel = sobel_y(n,m)
    gaussian_kernel = gaussian_filter(n,m,1)
    # cnn,bias = 0
    avg_filtered_image = avg_conv(image, n, m, avg_kernel, bias=0, padding=0)
    sobel_filtered_image = sobel_conv(image, n, m, sobel_x_kernel, sobel_y_kernel,bias=0, padding=1)
    gaussian_filtered_image = gaussian_conv(image, n, m, gaussian_kernel, bias=0, padding=1)
    pool_image = pool(gaussian_filtered_image, 2, 2, 1)
    # add signature
    signeture = cv2.imread('sign.png')
    result = add_signature(signeture, pool_image)
    cv2.imwrite(f'{name}_pool.png', result)
    result = add_signature(signeture, avg_filtered_image)
    cv2.imwrite(f'{name}_avg_filtered.png', result)
    result = add_signature(signeture, sobel_filtered_image)
    cv2.imwrite(f'{name}_sobel_filtered.png', result)
    result = add_signature(signeture, gaussian_filtered_image)
    cv2.imwrite(f'{name}_gaussian_filtered.png', result)