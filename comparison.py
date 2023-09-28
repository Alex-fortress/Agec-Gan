import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 加载生成图像和真实图像
generated_image_path = "generated photos/generated photo 3.jpg"
real_image_path = "UTKFace/UTKFace/40_1_1_20170110153441199.jpg.chip.jpg"

# 调整图像尺寸为相同的宽度和高度
generated_image = cv2.imread(generated_image_path)
real_image = cv2.imread(real_image_path)
generated_image = cv2.resize(generated_image, (real_image.shape[1], real_image.shape[0]))

# 将图像转换为灰度图像
generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
real_image_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)

# 计算PSNR值
psnr = compare_psnr(real_image, generated_image)

# 计算SSIM值
ssim = compare_ssim(real_image_gray, generated_image_gray, data_range=generated_image_gray.max() - generated_image_gray.min())

# 打印PSNR和SSIM值
print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.4f}")
