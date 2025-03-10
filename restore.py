import os
import cv2
import pydicom
import numpy as np
import argparse
from PIL import Image

# ✅ 인자 설정
parser = argparse.ArgumentParser(description="PNG를 원래 크기로 복원 후 DICOM 변환")
parser.add_argument("--output", type=str, default="output", help="복원된 DICOM 저장 디렉토리")
args = parser.parse_args()

# ✅ 경로 설정
cropped_folder = "./decoded_images"  # 크롭된 이미지 폴더
bounding_box_file =  "./processed_dicom/bounding_boxes.txt"# 바운딩 박스 정보 파일
meta_dicom_dir = "./processed_dicom/meta_dicom"  # PixelData가 제거된 DICOM 경로
restored_dicom_dir = args.output  # 최종 DICOM 복원 경로

# ✅ 출력 폴더 생성
os.makedirs(restored_dicom_dir, exist_ok=True)

# ✅ 바운딩 박스 정보 읽기
with open(bounding_box_file, "r") as f:
    lines = f.readlines()

# ✅ 첫 줄에서 원본 이미지 크기(W, H) 가져오기
total_slices, original_width, original_height = map(int, lines[0].split())

# ✅ 바운딩 박스 정보 저장 (딕셔너리)
bounding_boxes = {}  # {슬라이스_번호: (x_min, x_max, y_min, y_max)}

for line in lines[1:]:  # 첫 줄 제외
    parts = list(map(int, line.split()))
    slice_idx, x_min, x_max, y_min, y_max = parts
    bounding_boxes[slice_idx] = (x_min, x_max, y_min, y_max)

# ✅ PNG 파일을 원래 크기로 복원 후 DICOM으로 변환
for slice_idx in range(total_slices):
    reconstructed_image = np.zeros((original_height, original_width), dtype=np.uint8)  # 검은색 배경

    if slice_idx in bounding_boxes:
        # ✅ 크롭된 이미지 불러오기
        cropped_image_path = os.path.join(cropped_folder, f"cropped_{slice_idx}.png")
        cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 로드
        
        if cropped_image is not None:
            # ✅ 바운딩 박스 좌표 불러오기
            x_min, x_max, y_min, y_max = bounding_boxes[slice_idx]

            # ✅ 원본 크기의 배열에 크롭된 이미지 삽입
            reconstructed_image[y_min:y_max, x_min:x_max] = cropped_image

    # ✅ 복원된 PNG 저장
    temp_png_path = os.path.join(restored_dicom_dir, f"reconstructed_{slice_idx}.png")
    cv2.imwrite(temp_png_path, reconstructed_image)

    # ✅ DICOM 복원
    dicom_filename = f"{slice_idx}"  # 원래 DICOM 파일명
    dicom_path = os.path.join(meta_dicom_dir, dicom_filename)

    if os.path.exists(dicom_path):  # 메타데이터 DICOM 파일이 있어야 함
        ds = pydicom.dcmread(dicom_path)  # DICOM 파일 읽기
        img = Image.open(temp_png_path).convert("L")  # PNG를 grayscale로 변환
        image_array = np.array(img, dtype=np.uint8)  # 🔹 8비트 변환 (uint8)

        # 🔹 PixelData 복원
        ds.PixelData = image_array.tobytes()
        ds.Rows, ds.Columns = image_array.shape  # 이미지 크기 설정

        # 🔹 VR 모호성 해결 (OB/OW 문제 해결)
        ds.is_implicit_VR = False  # 명시적 VR 사용
        ds.is_little_endian = True  # 리틀 엔디안 설정
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  # 명시적 VR 설정

        # 🔹 DICOM 이미지 속성 설정 (8비트 저장)
        ds.BitsAllocated = 8  # 8비트 픽셀 할당
        ds.BitsStored = 8  # 8비트 저장
        ds.HighBit = 7  # 8비트의 최상위 비트 (0~7)
        ds.SamplesPerPixel = 1  # 흑백 이미지 (그레이스케일)
        ds.PhotometricInterpretation = "MONOCHROME2"  # 의료용 그레이스케일 설정

        # 🔹 **VR 명확히 지정**
        ds[0x7FE0, 0x0010].VR = 'OB'  # PixelData VR을 명확히 'OB'로 설정

        # 🔹 원래 파일로 저장
        restored_dicom_path = os.path.join(restored_dicom_dir, dicom_filename)
        ds.save_as(restored_dicom_path)

        print(f"✅ 복원된 DICOM 저장 완료: {restored_dicom_path}")

print(f"📂 모든 슬라이스 복원 완료! (DICOM 경로: {restored_dicom_dir})")
