import random
import numpy as np
from skimage.measure import label as lbl
from skimage.measure import regionprops
from skimage.transform import rotate
# from scipy import misc


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img
    
def adjust_img(source_img, reference_img):
    ''' Histogram matching'''
    matched_img = np.zeros_like(source_img, dtype=np.float32)

    for channel in range(3):  # R, G, B 각 채널 처리
        # 1. 소스 및 참조 히스토그램 계산
        src_hist, src_bins = np.histogram(source_img[..., channel].flatten(), bins=256, range=(0, 255))
        ref_hist, ref_bins = np.histogram(reference_img[..., channel].flatten(), bins=256, range=(0, 255))

        # 2. 누적 분포 함수(CDF) 계산
        src_cdf = np.cumsum(src_hist).astype(np.float32)
        src_cdf /= src_cdf[-1]  # Normalize to [0, 1]
        
        ref_cdf = np.cumsum(ref_hist).astype(np.float32)
        ref_cdf /= ref_cdf[-1]  # Normalize to [0, 1]

        # 3. 매핑 테이블 생성
        mapping = np.zeros(256, dtype=np.uint8)
        for src_value in range(256):
            closest_idx = np.argmin(np.abs(ref_cdf - src_cdf[src_value]))
            mapping[src_value] = closest_idx

        # 4. 매핑 적용
        matched_img[..., channel] = mapping[source_img[..., channel].astype(np.uint8)]
        
    matched_img = np.clip(matched_img, 0, 255).astype(np.uint8)
    
    ''' Brightness matching'''
    img = matched_img.astype(np.float32)
    current_mean = matched_img.mean()
    current_std = matched_img.std()
    
    target_mean = reference_img.mean()
    target_std = reference_img.std()

    # 밝기 및 대비 조정
    adjusted_img = (img - current_mean) / (current_std + 1e-6) * target_std + target_mean
    brightness_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    
    ''' Colour matching'''
    source_img = brightness_img.astype(np.float32)
    reference_img = reference_img.astype(np.float32)

    for channel in range(3):  # RGB 채널별로 처리
        source_mean, source_std = source_img[..., channel].mean(), source_img[..., channel].std()
        ref_mean, ref_std = reference_img[..., channel].mean(), reference_img[..., channel].std()

        # 색상 조정
        source_img[..., channel] = (source_img[..., channel] - source_mean) / (source_std + 1e-6) * ref_std + ref_mean

    return np.clip(source_img, 0, 255).astype(np.uint8)

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()

    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2

def random_rot_new(pre_img, post_img, label, crop_size, target_class=[1, 2], ignore_index=0):
    crop_size = int(np.ceil(np.sqrt(2* crop_size ** 2)))
    pre_img, post_img, label = random_crop_obj(pre_img, post_img, label, crop_size, target_class, [0,0,0], ignore_index)
    angle = np.random.uniform(-180,180)

    pre_img = rotate(pre_img, angle, order=0, preserve_range=True).copy()
    post_img = rotate(post_img, angle, order=0, preserve_range=True).copy()
    label = rotate(label, angle, order=0, preserve_range=True).copy()

    return pre_img, post_img, label

def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pad_image

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_image[H_start:H_end, W_start:W_end, 0]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = object.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    H_start = random.randrange(0, H - crop_size + 1, 1)
    H_end = H_start + crop_size
    W_start = random.randrange(0, W - crop_size + 1, 1)
    W_end = W_start + crop_size

    # H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    # post_img = post_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(GT))
    return pre_img, object


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.10):

        while(1):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.sum(cnt) / crop_size ** 2 >= cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_2 = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2
  
  
def random_crop_obj(pre_img, post_img, label, crop_size, target_class=None, mean_rgb=[0,0,0], ignore_index=255):
    h, w = label.shape
    
    H = max(crop_size, h)
    W = max(crop_size, w)
    
    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
    
    pad_pre_image[:,:,0] = mean_rgb[0]
    pad_pre_image[:,:,1] = mean_rgb[1]
    pad_pre_image[:,:,2] = mean_rgb[2]
    
    pad_post_image[:,:,0] = mean_rgb[0]
    pad_post_image[:,:,1] = mean_rgb[1]
    pad_post_image[:,:,2] = mean_rgb[2]
    
    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label
    
    def extract_patch_with_object_center():
        shift_x, shift_y = 0, 0

        labeled_array = lbl(label > 0, connectivity=1)
        regions = regionprops(labeled_array)
        
        object_centers = [(int(region.centroid[0]), int(region.centroid[1])) for region in regions]
        
        if not object_centers:
            return None
        if target_class:
            target_centers = list(filter(lambda x: label[x] in target_class, object_centers))
            if target_centers:
                selected_center = random.choice(target_centers)
            else:
                selected_center = random.choice(object_centers)
        else:
            selected_center = random.choice(object_centers)

        if random.random() > 0.5:
            shift_x = np.random.randint(-64,64)
            shift_y = np.random.randint(-64,64)

        H_start = max(0, selected_center[0] + shift_x - crop_size // 2)
        H_end = min(h, H_start + crop_size)
        W_start = max(0, selected_center[1] + shift_y - crop_size // 2)
        W_end = min(w, W_start + crop_size)
        
        H_start = H_end - crop_size
        W_start = W_end - crop_size
        
        return H_start, H_end, W_start, W_end,
        
    # 패치 좌표 계산
    patch_coords = extract_patch_with_object_center()

    # 객체가 없으면 기본 랜덤 크롭 수행
    if patch_coords is None:
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < 0.75:
                break
    else:
        H_start, H_end, W_start, W_end = patch_coords
    
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    
    return pre_img, post_img, label
       
    
def random_brightness_contrast(pre_img, post_img, label, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), prob=0.5):
    if random.random() < prob:
        brightness_factor = random.uniform(*brightness_range)
        contrast_factor = random.uniform(*contrast_range)

        pre_img = np.clip(pre_img * brightness_factor, 0, 255)
        post_img = np.clip(post_img * brightness_factor, 0, 255)
        
        pre_img = np.clip((pre_img - pre_img.mean()) * contrast_factor + pre_img.mean(), 0, 255)
        post_img = np.clip((post_img - post_img.mean()) * contrast_factor + post_img.mean(), 0, 255)

    return pre_img, post_img, label

def random_noise(pre_img, post_img, label, noise_factor=0.05, prob=0.5):
    if random.random() < prob:
        noise = np.random.normal(0, noise_factor, pre_img.shape)
        pre_img = np.clip(pre_img + noise * 255, 0, 255)
        post_img = np.clip(post_img + noise * 255, 0, 255)

    return pre_img, post_img, label

def random_cutout(pre_img, post_img, label, max_hole_size=64, prob=0.5):
    if random.random() < prob:
        h, w = (np.array(pre_img.shape[:2]) / 2).astype(int)
        hole_h = random.randint(0, max_hole_size)
        hole_w = random.randint(0, max_hole_size)

        # 랜덤한 위치에 사각형 구멍을 만듬
        top = random.randint(0, h - hole_h)
        left = random.randint(0, w - hole_w)

        # 구멍 부분을 0으로 설정 (배경값을 이용)
        pre_img[top:top + hole_h, left:left + hole_w] = 0
        post_img[top:top + hole_h, left:left + hole_w] = 0
        label[top:top + hole_h, left:left + hole_w] = 0

    return pre_img, post_img, label
