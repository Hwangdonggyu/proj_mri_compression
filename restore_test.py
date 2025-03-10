import os
import cv2
import pydicom
import numpy as np
import argparse
from PIL import Image
import re

# ✅ 인자 설정
parser = argparse.ArgumentParser(description="PNG를 원래 크기로 복원 후 DICOM 변환")
parser.add_argument("--output", type=str, default="output", help="복원된 DICOM 저장 디렉토리")
args = parser.parse_args()

# ✅ 경로 설정
decoded_folder = "./decoded_images"  # PNG 해제된 이미지 폴더
error_images_folder = "./error_images"  # WebP 손상된 이미지 폴더
bounding_box_file = "./processed_dicom/bounding_boxes.txt"  # 바운딩 박스 정보 파일
meta_dicom_dir = "./processed_dicom/meta_dicom"  # PixelData 제거된 DICOM 경로
restored_dicom_dir = args.output  # 최종 DICOM 복원 경로

# ✅ 출력 폴더 생성
os.makedirs(restored_dicom_dir, exist_ok=True)

# ✅ "error_images" 폴더의 webp 파일을 png로 변환하여 decoded_images에 추가
for filename in os.listdir(error_images_folder):
    if filename.endswith(".webp"):
        webp_path = os.path.join(error_images_folder, filename)
        output_png_path = os.path.join(decoded_folder, filename.replace(".webp", ".png"))

        img = Image.open(webp_path).convert("L")  # WebP 파일 열기 및 Grayscale 변환
        img.save(output_png_path)  # PNG로 저장

        print(f"🖼️ WebP 변환 완료: {output_png_path}")

# ✅ 바운딩 박스 정보 읽기
with open(bounding_box_file, "r") as f:
    lines = f.readlines()

# ✅ 첫 줄에서 원본 이미지 크기(W, H) 가져오기
total_slices, original_width, original_height = map(int, lines[0].split())

# ✅ 바운딩 박스 정보 저장 (딕셔너리)
bounding_boxes = {}  # {슬라이스_번호: (x_min, x_max, y_min, y_max)}
for line in lines[1:]:
    parts = list(map(int, line.split()))
    slice_idx, x_min, x_max, y_min, y_max = parts
    bounding_boxes[slice_idx] = (x_min, x_max, y_min, y_max)

# ✅ meta_dicom 파일 이름에서 슬라이스 번호를 추출하여 매핑
dicom_map = {}  # {slice_idx: filename}
for file in os.listdir(meta_dicom_dir):
    if file.endswith(".dcm"):
        path = os.path.join(meta_dicom_dir, file)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            slice_idx = int(ds.InstanceNumber)  # 또는 다른 필드
            dicom_map[slice_idx] = file
        except:
            continue

# ✅ 정렬을 위한 숫자 추출 함수
def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

# ✅ PNG 파일 이름 리스트 준비 (디코딩된 이미지)
image_files = sorted(os.listdir(decoded_folder), key=extract_number)

# ✅ 복원 수행
for slice_idx in range(1, total_slices + 1):
    reconstructed_image = np.zeros((original_height, original_width), dtype=np.uint8)

    # ✅ 바운딩 박스 존재 시 이미지 복원
    if slice_idx in bounding_boxes:
        matched_files = [f for f in image_files if extract_number(f) == slice_idx]
        if matched_files:
            cropped_image_path = os.path.join(decoded_folder, matched_files[0])
            cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)

            if cropped_image is not None:
                x_min, x_max, y_min, y_max = bounding_boxes[slice_idx]
                reconstructed_image[y_min:y_max, x_min:x_max] = cropped_image

    # ✅ 해당 슬라이스의 DICOM 메타 파일이 있는 경우만 복원
    if slice_idx in dicom_map:
        dicom_path = os.path.join(meta_dicom_dir, dicom_map[slice_idx])
        ds = pydicom.dcmread(dicom_path)

        # ✅ PixelData 복원
        ds.PixelData = reconstructed_image.tobytes()
        ds.Rows, ds.Columns = reconstructed_image.shape
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.is_implicit_VR = False
        ds.is_little_endian = True
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds[0x7FE0, 0x0010].VR = 'OB'

        # ✅ 원래 파일명 그대로 저장
        restored_path = os.path.join(restored_dicom_dir, dicom_map[slice_idx])
        ds.save_as(restored_path)

        print(f"✅ 복원된 DICOM 저장 완료: {restored_path}")

    else:
        print(f"⚠️ 슬라이스 {slice_idx}에 대한 DICOM 메타데이터 파일 없음 — 저장 생략")

print(f"\n🎉 모든 DICOM 복원 완료! 결과 경로: {restored_dicom_dir}")
