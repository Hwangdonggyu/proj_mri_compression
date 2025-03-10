import os
import pydicom
import numpy as np
import cv2

# ✅ 입력 DICOM 디렉토리
dicom_dir = "/nas-home/donggyu/adni_split/SIEMENS/051_S_4980/MPRAGE/2013-12-16_10_18_05.0/I401803"

# ✅ 출력 디렉토리 설정
output_png_dir = "./processed_dicom/png"  # PNG 저장 디렉토리
output_meta_dicom_dir = "./processed_dicom/meta_dicom"  # 메타데이터만 남긴 DICOM 저장 디렉토리
bounding_box_file = os.path.join("./processed_dicom/", "bounding_boxes.txt")  # 바운딩 박스 정보 저장 파일

# ✅ 출력 폴더 생성
os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_meta_dicom_dir, exist_ok=True)

# ✅ DICOM 파일 정렬 (InstanceNumber 기준)
def get_instance_number(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    return int(ds.InstanceNumber) if 'InstanceNumber' in ds else 0
    

dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
dicom_files.sort(key=lambda x: get_instance_number(os.path.join(dicom_dir, x)))

# ✅ 첫 번째 DICOM 파일에서 이미지 크기 가져오기
first_dicom = pydicom.dcmread(os.path.join(dicom_dir, dicom_files[0]))
image_height, image_width = first_dicom.pixel_array.shape

# ✅ Threshold 자동 설정 (10 이하 픽셀 제외 후 평균)
nonzero_pixel_values = []
for dicom_file in dicom_files:
    dicom_path = os.path.join(dicom_dir, dicom_file)
    ds = pydicom.dcmread(dicom_path)
    valid_pixels = ds.pixel_array[ds.pixel_array > 10]  # 10 이하 제거
    nonzero_pixel_values.extend(valid_pixels)

threshold = np.mean(nonzero_pixel_values) if len(nonzero_pixel_values) > 0 else 150
print(f"🎯 자동 설정된 Threshold 값: {threshold:.2f}")

# ✅ 바운딩 박스 정보 저장 파일 초기화
with open(bounding_box_file, "w") as f:
    f.write(f"{len(dicom_files)} {image_width} {image_height}\n")  # 첫 줄: 슬라이스 개수 + 원본 크기(W, H)

# ✅ DICOM 처리 (모든 DICOM 저장, PNG는 Bounding Box 있을 때만 저장)
for idx, dicom_file in enumerate(dicom_files):
    dicom_path = os.path.join(dicom_dir, dicom_file)
    ds = pydicom.dcmread(dicom_path)  # DICOM 파일 로드

    # ✅ 바운딩 박스 생성
    image_array = ds.pixel_array.astype(np.float32)
    mask = image_array > threshold
    y_indices, x_indices = np.where(mask)

    # 🚨 예외 처리: Bounding Box 감지 확인
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # ✅ 크롭 이미지 생성
        cropped_image = image_array[y_min:y_max, x_min:x_max]

        # 🚨 예외 처리: 빈 배열인지 확인
        if cropped_image.size == 0 or (x_max - x_min) < 5 or (y_max - y_min) < 5:
            if "PixelData" in ds:
                del ds.PixelData  # PixelData 제거
            meta_dicom_filename = os.path.join(output_meta_dicom_dir, dicom_file)
            ds.save_as(meta_dicom_filename)  # 메타데이터만 저장된 DICOM 저장
            continue
            
        else:
            # ✅ 정규화 (0~255) 후 8비트 변환
            cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min() + 1e-7) * 255
            cropped_image = cropped_image.astype(np.uint8)

            # ✅ PNG 저장
            png_filename = os.path.join(output_png_dir, f"cropped_{idx}.png")
            cv2.imwrite(png_filename, cropped_image)

            # ✅ 바운딩 박스 정보 저장
            with open(bounding_box_file, "a") as f:
                f.write(f"{idx} {x_min} {x_max} {y_min} {y_max}\n")

    # ✅ PixelData 제거 후 메타데이터만 남긴 DICOM 저장 (모든 DICOM 파일 저장)
    if "PixelData" in ds:
        del ds.PixelData  # PixelData 제거

    meta_dicom_filename = os.path.join(output_meta_dicom_dir, dicom_file)
    ds.save_as(meta_dicom_filename)  # 메타데이터만 저장된 DICOM 저장

print(f"📂 바운딩 박스 정보 저장 완료: {bounding_box_file}")
