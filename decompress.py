import os
import pydicom
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="PNG를 DICOM으로 복원")
parser.add_argument("--output", type=str, default="output", help="복원된 DICOM 저장 디렉토리")
args = parser.parse_args()

# 복원할 파일 경로 설정
png_dir = "./decoded_images"  # 저장된 PNG 경로
meta_dicom_dir = "./processed_dicom/meta_dicom"  # PixelData가 제거된 DICOM 경로
restored_dicom_dir = args.output  # 복원된 DICOM 저장 경로

# 출력 폴더 생성
os.makedirs(restored_dicom_dir, exist_ok=True)

# PNG 파일을 DICOM으로 복원
for filename in os.listdir(png_dir):
    if filename.endswith(".png"):
        png_path = os.path.join(png_dir, filename)
        dicom_filename = filename.replace(".png", "")  # 원래 DICOM 파일명 복원
        dicom_path = os.path.join(meta_dicom_dir, dicom_filename)

        if os.path.exists(dicom_path):  # 메타데이터 DICOM 파일이 있어야 함
            ds = pydicom.dcmread(dicom_path)  # DICOM 파일 읽기
            img = Image.open(png_path).convert("L")  # PNG를 grayscale로 변환
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

print("✅ PNG에서 PixelData를 복원하여 8비트 DICOM 파일이 생성되었습니다!")
