#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

sudo apt install nvidia-cuda-toolkit
echo -e "${YELLOW}Python CUDA 세미나 환경 설정을 시작합니다...${NC}"

# # Python 버전 확인
# if ! command --version python3 &> /dev/null; then
#     echo -e "${RED}Python3가 설치되어 있지 않습니다.${NC}"
#     exit 1
# fi

# # CUDA 설치 확인
# if ! command --version nvcc &> /dev/null; then
#     echo -e "${RED}CUDA가 설치되어 있지 않습니다.${NC}"
#     echo -e "${YELLOW}CUDA Toolkit을 설치해주세요: https://developer.nvidia.com/cuda-downloads${NC}"
#     exit 1
# fi

# # 가상환경 생성
# echo -e "${GREEN}가상환경을 생성합니다...${NC}"
# python3 -m venv .venv

# 가상환경 활성화
echo -e "${GREEN}가상환경을 활성화합니다...${NC}"
source .venv/bin/activate

# pip 업그레이드
echo -e "${GREEN}pip를 업그레이드합니다...${NC}"
pip install --upgrade pip

# 필요한 패키지 설치
echo -e "${GREEN}필요한 패키지들을 설치합니다...${NC}"
pip install -r requirements.txt

# 설치 확인
echo -e "${GREEN}설치된 패키지 목록:${NC}"
pip list

echo -e "${GREEN}환경 설정이 완료되었습니다!${NC}"
echo -e "${YELLOW}다음 명령어로 코드를 실행할 수 있습니다:${NC}"
echo -e "source .venv/bin/activate  # 가상환경 활성화"
echo -e "python code_1.py          # 벤치마크 실행" 