# Introduction

"Multispectral Pedestrian Detection with Sparsely Annotated Label"를 진행하기 위해 사용한 소스 코드입니다.

개발은 학교 세라프 환경에서 진행하였으며, 학습 및 inference를 위한 sh파일은 이에 맞춰져 있습니다.

따라서 소스코드의 진위 여부를 판별하시거나 동작 여부를 확인하시고자 한다면, 안정적인 학습을 위해 GPU를 2개 이상 할당 받을 수 있는 세라프 계정 사용을 권장합니다.

# Weights & Dataset

학습 진위 여부의 편의를 위해 제가 학습한 모델의 파일을 [구글 드라이브](https://drive.google.com/drive/folders/1ZoCP6xC78e7qo847QqaeevQmj8UeQwel?usp=sharing)로 공유해드립니다.

또한 데이터셋의 경우 [KAIST Multispectral Pedestrian Detection Benchmark Dataset](https://soonminhwang.github.io/rgbt-ped-detection/)에서 다운로드 받을 수 있습니다.

여기서 annoatation 파일로는 위 데이터셋 링크의 것이 아닌 다른 버전을 사용해 추가적으로 다운로드가 필요합니다. 

해당 [annotations-paired](https://drive.google.com/file/d/1FLkoJQOGt4PqRtr0j6namAaehrdt_A45/view) 파일을 다운받아주시기 바랍니다.

모든 파일을 다운받아 아래와 같이 위치시키면 아래 Guide 섹션에서 안내할 내용을 그대로 따라하실 수 있습니다.

이 때 폴더의 이름들이 아래와 같이 되어있는지 반드시 확인해주시기 바랍니다.

```bash
├── data
│   └── kaist-rgbt
│       ├── annotations_paired
│       │   └── set00
│       │       └── V000
│       │           ├── lwir
│       │           │   └── I00000.txt
│       │           └── visible
│       │               └── I00000.txt
│       └── images
│           └── set00
│               └── V000
│                   ├── lwir
│                   │   └── I00000.jpg
│                   └── visible
│                       └── I00000.jpg
├── src
│   ├─── train.py
│   └─── ...
├── final_weights
│   ├─── KAIST_30.pth.tar000
│   └─── ...
├── final_weights
│   ├─── KAIST_pretrained_30.pth.tar000
│   └─── ...
├── train.sh
└── inference.sh
``` 

# Guide

학습을 위해서는 train.sh 파일을 sbatch를 이용해 job을 올리시면 됩니다.

학습을 위해서는 세라프 계정에 접속한 후 터미널에서 다음과 같이 사용할 수 있습니다.

```
sbatch train.sh
```

또한 inference를 위해서는 inference.sh 파일을 사용하시면 됩니다. Inference의 경우 srun.sh을 이용해 debug node를 할당 받은 상태에서 진행하는 것을 권장드립니다.

Inference을 위해서는 세라프 계정에 접속한 후 터미널에서 다음과 같이 사용할 수 있습니다.

```
sh srun.sh

sh inference.sh
```

# Acknowledgement

This work was supported by the NRF grant funded by the Korea government (MSIT) (No. RS-2023-00252391), and by IITP grant funded by the Korea government (MSIT) (No. RS-2022-00155911: Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University), IITP-2022-II220078: Explainable Logical Reasoning for Medical Knowledge Generation, No. RS-2024-00509257: Global AI Frontier Lab), and by the MSIT (Ministry of Science and ICT), Korea, under the National Program for Excellence in SW (2023-0-00042) supervised by the IITP in 2025, and conducted by CARAI grant funded by DAPA and ADD (UD230017TD).