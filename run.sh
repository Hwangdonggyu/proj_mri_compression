#!/bin/bash

# ✅ 도움말 출력
usage() {
    echo "사용법: $0 --data_path <경로> --mode <comp/decomp> --output <결과물 이름> [--gpu <GPU번호>]"
    echo "예시: $0 --data_path /dicom/data/path --mode comp --output compressed.zpaq --gpu 0"
    exit 1
}

# ✅ 기본값 설정
GPU_ID=0  # 기본 GPU ID는 0

# ✅ 입력값 파싱
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_path) DATA_PATH="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --output) OUTPUT_NAME="$2"; shift 2;;
        --gpu) GPU_ID="$2"; shift 2;;
        *) usage;;
    esac
done

# ✅ 필수 인자 확인
if [ -z "$DATA_PATH" ] || [ -z "$MODE" ] || [ -z "$OUTPUT_NAME" ]; then
    usage
fi

# ✅ GPU 설정
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ✅ 실행 모드 확인
if [ "$MODE" == "comp" ]; then
    echo "🚀 압축 모드 실행 (comp)..."

    # ✅ 실행
    echo "실행 중: python compress.py --data_path $DATA_PATH"
    python "compress.py" --data_path "$DATA_PATH"

    # ✅ LIC_TCM 폴더로 이동
    LIC_TCM_PATH="./LIC_TCM"
    if [ -d "$LIC_TCM_PATH" ]; then
        cd "$LIC_TCM_PATH" || { echo "오류: $LIC_TCM_PATH 디렉터리로 이동 실패"; exit 1; }
    else
        echo "오류: $LIC_TCM_PATH 디렉터리가 존재하지 않습니다."
        exit 1
    fi

    python "compress.py" --data "../processed_dicom/png" --"$MODE" --checkpoint "./checkpoint/0.05checkpoint_best.pth.tar" --cuda --gpu_id "$GPU_ID"

    # ✅ zpaq 압축 실행
    OUTPUT_ZPAQ="../$OUTPUT_NAME"

    rm -rf "../processed_dicom/png"
    mv "../processed_dicom" "./" 

    echo "📦 zpaq 압축 시작..."
    zpaq add "$OUTPUT_ZPAQ" "./error_images" "./compressed_data" "./processed_dicom/" -m5 -force

    echo "✅ 압축 완료: $OUTPUT_ZPAQ"

    # ✅ 원본 폴더 삭제
    rm -rf "./processed_dicom"
    rm -rf "./compressed_data"
    rm -rf "./error_images"

elif [ "$MODE" == "decomp" ]; then
    echo "📦 압축 해제 모드 실행 (decomp)..."

    # ✅ zpaq 압축 해제
    zpaq x "$DATA_PATH" -force

    echo "압축 해제 완료: $DATA_PATH"

    python "./LIC_TCM/compress.py" --data "./compressed_data" --"$MODE" --checkpoint "./LIC_TCM/checkpoint/0.05checkpoint_best.pth.tar" --cuda --gpu_id "$GPU_ID"

    python "./restore_test.py" --output "$OUTPUT_NAME"

    # ✅ 원본 폴더 삭제
    rm -rf "./processed_dicom"
    rm -rf "./compressed_data"
    rm -rf "./error_images"
    rm -rf "./decoded_images"


else
    echo "오류: 지원되지 않는 모드입니다. 'comp' 또는 'decomp'를 선택하세요."
    exit 1
fi
