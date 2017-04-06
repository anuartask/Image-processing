import numpy as np
from skimage.draw import ellipse
from skimage.draw import line
from skimage.morphology import disk, watershed, convex_hull_image, erosion
from skimage.transform import rotate
from scipy.ndimage.morphology import distance_transform_bf
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage import io, filters, measure, feature
from skimage.filters import sobel
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from sklearn.ensemble import RandomForestClassifier
import copy

y_dozen = np.array([7, 4, 6, 8, 5, 3, 9, 10, 2, 1])

color = ['y', 'r', 'b']

c = np.array([[255, 255, 0],
              [255, 0, 0],
              [0, 0, 255]])
color_train = np.array([[194, 121, 6],
                        [224, 152, 7],
                        [215, 131, 6],
                        [224, 153, 7],
                        [223, 151, 7],
                        [199, 129, 6],
                        [126, 74, 14],
                        [127, 83, 2],
                        [186, 120, 15],
                        [164, 106, 22],
                        [200, 152, 61],
               
                        [176, 29, 19],
                        [146, 25, 12],
                        [175, 28, 20],
                        [168, 28, 20],
                        [167, 27, 20],
                        [178, 72, 38],
                        [199, 112, 87],
                        [180, 54, 51],
                        [175, 56, 46],
                        [200, 108, 83],
                        [152, 27, 15],
              
                        [44, 38, 38],
                        [51, 32, 48],
                        [55, 49, 57],
                        [47, 45, 53],
                        [179, 149, 127],
                        [183, 153, 126],
                        [149, 128, 116],
                        [62, 55, 63],
                        [25, 19, 13]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2, 2])

color_clf = RandomForestClassifier(n_estimators=100)
color_clf.fit(color_train, y_train)


def binary_image(image, show=False):
    
    """Возвращает бинаризованную картинку.
        
        image: входное изображение размера (M, N, 3).
        
        show: вывод на экран бинаризованного изображения
        
    """
    
    #grayscale преобразование
    gray_image = rgb2gray(image).astype('float64')
    
    #фильтр Sobel, выделение границ
    bin_image = sobel(gray_image)
    
    #бинаризация с порогом
    markers = np.zeros_like(bin_image)
    gray_image *= 255.
    markers[gray_image < 105] = 1
    markers[gray_image > 110] = 2
    
    #алгоритм водораздела
    bin_image = watershed(bin_image, markers)
    
    #запонение границ
    bin_image = ndi.binary_fill_holes(2 - bin_image)
    
    #удаление мелких объектов
    label_objects, nb_labels = ndi.label(bin_image)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 1000
    mask_sizes[0] = 0
    bin_image = mask_sizes[label_objects]
    
    if show:
        io.imshow(bin_image)
    
    return bin_image

def is_hexagon(seg_image, show=False):
    
    """Возвращает маску с шестиугольниками.
        
        seg_image: сегментированное изображение размера (M, N).
        
        show: вывод на экран бинаризованного изображения
        
    """
    
    
    mask = np.zeros(seg_image.shape).astype('bool')
    n_segments = np.unique(seg_image)[1: ]
    
    S = seg_image.shape[0] * seg_image.shape[1]
    
    for n_seg in n_segments:
        bin_image = np.zeros(seg_image.shape)
        bin_image[seg_image == n_seg] = 1
        
        #построение выпуклой оболочки
        bin_image = convex_hull_image(bin_image)
        
        #поиск центра масс
        S_hexagon = bin_image.sum()
        x_, y_ = np.where(bin_image)
        x_center = np.ceil((x_ * bin_image[x_, y_]).sum() / S_hexagon)
        y_center = np.ceil((y_ * bin_image[x_, y_]).sum() / S_hexagon)
        
        #алгоритм distance transform
        d_transform = distance_transform_bf(bin_image)
        x_d_center, y_d_center = np.where(d_transform == d_transform.max())
        x_d_center = x_d_center[0]
        y_d_center = y_d_center[0]
        
        #если центр, найденный через distance transform совпадает с центром масс, то это шестиугольник
        if np.sqrt((x_d_center - x_center) ** 2 + (y_d_center - y_center) ** 2) <= 9:
            if (S_hexagon / S > 0.04):
                bin_image = erosion(bin_image, disk(20))
            else:
                bin_image = erosion(bin_image, disk(5))           
            mask[bin_image == 1] = True
            
    
    if show:
        io.imshow(mask)
    return mask


def feature_hexagon(seg_image, show=False):
    """
        Возвращает словарь с признаками шестиугольников.
    """
    
    feature_image = np.zeros(seg_image.shape).astype('bool')    
    n_segments = np.unique(seg_image)[1: ]
    rad_min = []
    rad_max = []
    center = []
    extra_dot_1 = []
    extra_dot_2 = []
    angle = []
    
    strel = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
    x = np.tile(np.arange(feature_image.shape[1]), feature_image.shape[0]).reshape(feature_image.shape)
    y = np.tile(np.arange(feature_image.shape[0]), feature_image.shape[1]).reshape(feature_image.T.shape).T
        
    for n_seg in n_segments:
        bin_image = np.zeros(seg_image.shape).astype('bool')
        bin_image[seg_image == n_seg] = True
        
        #поиск центра масс шестиугольника
        S = bin_image.sum()
        x_, y_ = np.where(bin_image)
        x_center = np.ceil((x_ * bin_image[x_, y_]).sum() / S)
        y_center = np.ceil((y_ * bin_image[x_, y_]).sum() / S)
        center.append((x_center, y_center))
        
        #выделение границы шестиугольника
        eroded_image = ndi.binary_erosion(bin_image, strel, border_value=0)
        border_image = bin_image - eroded_image
        
        #периметр
        x_border, y_border = np.where(border_image)
        P = x_border.shape[0]
        
        #радиус вписанной и описанной окружности
        r_min = np.min(np.sqrt((x_border - x_center) ** 2 + (y_border - y_center) ** 2))
        r_max = np.max(np.sqrt((x_border - x_center) ** 2 + (y_border - y_center) ** 2))
        r_min = np.ceil(r_min)
        r_max = np.ceil(r_max)
        rad_min.append(r_min)
        rad_max.append(r_max)
        
        #максимально отдаленные точки на границе
        x_b = np.tile(x_border, P).reshape((P, P))
        y_b = np.tile(y_border, P).reshape((P, P))
        vert = np.argmax((x_b - x_border[np.newaxis, :].T) ** 2 + (y_b - y_border[np.newaxis, :].T) ** 2)
        i = vert // P
        j = vert % P
        
        feature_image[x_border, y_border] = True
        extra_dot_1.append((x_border[i], y_border[i]))
        extra_dot_2.append((x_border[j], y_border[j]))
        feature_image[line(x_border[i], y_border[i], x_border[j], y_border[j])] = True
        
        #угол наклона шестиугольника относительно оси X
        ang = np.arctan(-(x_border[i] - x_border[j]) / (y_border[i] - y_border[j])) * 180 / np.pi
        if ang > 0:
            ang = 120 - ang
        else:
            ang = -60 - ang
        angle.append(ang)
        
        
    if show:
        io.imshow(feature_image)
        
        
    return {'feature_image': feature_image,
            'r_min': rad_min,
            'r_max': rad_max,
            'center': center,
            'extra_dot_1': extra_dot_1,
            'extra_dot_2': extra_dot_2,
            'angle': angle}

def colors_order(image, seg_image, feat, filename, dozen=False):
    """
        Возвращает цвета на гранях в "нормализованном" виде.
        
        image: исходное изображение размера (M, N, 3)
        
        seg_image: сегментированное изображение размера (M, N)
        
        feat: словарь признаков
        
        filename: строка, указывающее на место сохранения картинки с распознанными цветами
        
        dozen: является ли изображение обучающим.
    
    """
    
    n_rotate = 6 #количество поворотов
    n_segments = np.unique(seg_image)[1: ]
    hexagons_type = []
    rec_len = 4 #длина квадрата
    
    for n_seg in n_segments:
        im = copy.deepcopy(image)
        im[seg_image != n_seg] = 0
        if dozen:
            io.imsave("d_" + str(n_seg) + '.png', im)
        x_center, y_center = feat['center'][n_seg - 1]
        r_min, r_max = feat['r_min'][n_seg - 1], feat['r_max'][n_seg - 1]
        x_max = x_center + r_max - 1
        x_min = x_center - r_max + 1
        y_max = y_center + r_max - 1
        y_min = y_center - r_max + 1
        
        #выделяем подматрицу с изображением конкретной фишки
        im_seg = im[x_min: x_max, y_min: y_max]
        hex_colors = []
        angle = feat['angle'][n_seg - 1]
        rows, cols, _ = im_seg.shape
        
        #повернем изображение так, чтобы изображение стояло на грани
        im_seg = rotate(im_seg, angle)
        for i in range(n_rotate):
            x_center, y_center = rows / 2 - 0.5, cols / 2 - 0.5
            x_r, y_r = x_center - r_min + 5, y_center
            
            #находим средний цвет в квадрате, находящейся на грани
            x_min = int(x_r - (rec_len / 2))
            x_max = int(x_r + (rec_len / 2))
            y_min = int(y_r - (rec_len / 2))
            y_max = int(y_r + (rec_len / 2))
            mean_color = (im_seg[x_min: x_max, y_min: y_max, :] * 255.).mean(axis=(0, 1))
            
            #предсказываем цвет
            hex_colors.append(color_clf.predict(mean_color)[0])
            rr, cc = ellipse(x_r, y_r, 5, 5, im_seg.shape)
            im_seg[rr, cc] = c[color_clf.predict(mean_color)[0]] / 255.
            
            #поворот шестиугольника на 60 градусов
            im_seg = rotate(im_seg, 60)

        io.imsave(filename + "_im_seg_" + str(n_seg) + ".png", im_seg)
        
        #цвета представляем в нормализованном виде
        hex_color_1 = hex_colors
        if dozen:
            while hex_color_1[0] != 0:
                hex_color_1 = np.roll(hex_color_1, 1)
            hexagons_type.append(hex_color_1)
        else:
            while hex_color_1[0] != 0:
                hex_color_1 = np.roll(hex_color_1, 1)
            hex_color_2 = np.roll(hex_color_1, 1)
            while hex_color_2[0] != 0:
                hex_color_2 = np.roll(hex_color_2, 1)
            hexagons_type.append((hex_color_1, hex_color_2))
        
    return hexagons_type

def predict(fit_hex, image, seg_image, feat, colors, filename):
    """
        Сохраняет изображение с предсказанным номером фишки.
    """
    
    n_segments = np.unique(seg_image)[1: ]
    
    font = ImageFont.truetype(font="/usr/share/fonts/truetype/unfonts-core/UnDinaruBold.ttf", size=15)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    
    for n_seg in n_segments:
        x_center, y_center = feat['center'][n_seg - 1]
        
        for i in range(len(fit_hex)):
            if ((np.sum(colors[n_seg - 1][0] != fit_hex[i]) == 0) or
               (np.sum(colors[n_seg - 1][1] != fit_hex[i]) == 0)):
                
                draw.text((y_center, x_center), str(y_dozen[i]), (255, 255, 255), font=font)
                break
    image.save(filename + "_predict.png")