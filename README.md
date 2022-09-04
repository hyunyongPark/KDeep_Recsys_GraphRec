# kdeepfashion_RecSys

### Original git
- https://github.com/ahmedrashed-ml/GraphRec


### Build Docker Image
'''
# 도커이미지 빌드
sudo docker build -t graphrec:v1 .

# 생성된 도커이미지 확인
sudo docker images
'''

'''
# 생성된 도커이미지 실행
sudo docker run graphrec:v1
'''
- Result



### Requirements
'''
# python version : 3.8.13
pip install -r requirements.txt 
'''



### 환경 세팅

The install cmd is:
```
conda create -n your_prjname python=3.6
conda activate your_prjname
cd /생성된 가상환경경로
pip install -r rembg/requirements.txt
```

- your_prjname : 생성할 가상환경 이름 (파이썬버전은 3.6 고정)


### 학습 weight file 다운로드 
아래의 링크를 통해 학습 weight 파일을 다운받습니다. 해당 파일은 rembg에서 학습한 pretrained file입니다.
해당 weight 파일은 "가상환경/rembg/src/rembg/cmd/" 에 위치하도록 합니다.  
- https://drive.google.com/drive/folders/1tm6HLIx_r9jNquIUPyGtHk1TQR2XOWIw?usp=sharing

<table>
    <thead>
        <tr>
            <td>Example</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/removal_code/blob/main/img/img1.PNG"/></td>
        </tr>
    </tbody>
</table>


### 사용사항

Remove the background from a remote image
```

python3 rembg/src/rembg/cmd/cli.py --input_path "원본 이미지가 저장된 로컬 경로" --output_path "배경제거 처리 된 이미지가 저장될 경로"

```



### Result
<table>
    <thead>
        <tr>
            <td>Original</td>
            <td>Without background</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/removal_code/blob/master/test1.jpg"/></td>
            <td><img src="https://github.com/hyunyongPark/removal_code/blob/main/output/test1._removed.jpg"/></td>
        </tr>
    </tbody>
</table>

#### Local save
<table>
    <thead>
        <tr>
            <td>Original local folder</td>
            <td>Without background local folder</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/removal_code/blob/main/img/example1.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/removal_code/blob/main/img/example2.PNG"/></td>
        </tr>
    </tbody>
</table>



### References

- https://arxiv.org/pdf/2005.09007.pdf
- https://github.com/NathanUA/U-2-Net
- https://github.com/pymatting/pymatting

