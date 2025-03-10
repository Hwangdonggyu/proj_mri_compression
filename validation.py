import os
import pydicom
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")


# ✅ 폴더 경로 설정
original_dir = "/nas-home/donggyu/adni_split/SIEMENS/051_S_4980/MPRAGE/2013-12-16_10_18_05.0/I401803"
restored_dir = "/home/donggyu/cmc/mri_compress/a"

# ✅ InstanceNumber 기준으로 정렬된 파일 리스트 생성 함수
def get_sorted_file_list_by_instance_number(dicom_dir):
    file_list = []
    for file in os.listdir(dicom_dir):
        if file.endswith(".dcm"):
            path = os.path.join(dicom_dir, file)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                instance_number = int(ds.get(("InstanceNumber"), None))  # Tag: (0020,0013)
                if instance_number is not None:
                    file_list.append((int(instance_number), file))
            except Exception as e:
                print(f"⚠️ {file} 읽기 오류: {e}")
    return [f[1] for f in sorted(file_list, key=lambda x: x[0])]

# ✅ 정렬된 파일 리스트 불러오기
original_files = get_sorted_file_list_by_instance_number(original_dir)
restored_files = get_sorted_file_list_by_instance_number(restored_dir)

# ✅ 비교할 주요 DICOM 메타데이터 키워드 목록
important_keywords = [
    "PatientID", "PatientName", "StudyInstanceUID", "StudyDate", "StudyTime",
    "Modality", "SeriesInstanceUID", "SeriesNumber", "InstanceNumber",
    "SOPInstanceUID", "SOPClassUID", "Manufacturer", "Rows", "Columns",
    "SamplesPerPixel", "PhotometricInterpretation"
]

# ✅ 파일 개수 확인
if len(original_files) != len(restored_files):
    print(f"⚠️ 파일 수 불일치! 원본: {len(original_files)}개, 복원본: {len(restored_files)}개")

different = False
# ✅ 비교 시작
for orig_file, rest_file in zip(original_files, restored_files):
    orig_path = os.path.join(original_dir, orig_file)
    rest_path = os.path.join(restored_dir, rest_file)

    orig_ds = pydicom.dcmread(orig_path, stop_before_pixels=True)
    rest_ds = pydicom.dcmread(rest_path, stop_before_pixels=True)

    different_tags = []

    for keyword in important_keywords:
        if keyword in orig_ds:
            orig_val = orig_ds.data_element(keyword).value
            if keyword in rest_ds:
                rest_val = rest_ds.data_element(keyword).value
                if orig_val != rest_val:
                    print(f"  ❗ 다름 - Keyword: {keyword}")
                    print(f"    ▶ 원본:    {orig_val}")
                    print(f"    ▶ 복원본:  {rest_val}")
                    different_tags.append(keyword)
                    different = True
            else:
                print(f"  ❗ 복원본에 없음 - Keyword: {keyword}")
                print(f"    ▶ 원본 값: {orig_val}")
                different_tags.append(keyword)
                different = True
        else:
            if keyword in rest_ds:
                print(f"  ❗ 원본에 없음 - Keyword: {keyword}")
                print(f"    ▶ 복원본 값: {rest_ds.data_element(keyword).value}")
                different_tags.append(keyword)
                different = True

if not different:
    print("  ✅ 주요 메타데이터 완전 일치!")
