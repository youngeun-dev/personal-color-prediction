# PersonalColor-Prediction
image classification을 이용한 퍼스널 컬러 예측 모델
<br>
<br>
<br>

## 🔑 dataset 구축 과정
<img width="1287" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/26a62867-1764-4a09-9c0b-f3e2a9d51d62">
<img width="1300" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/84892d6e-63b7-4f64-ac31-bd05acb584ad">

1. `Image Collection`: 'Google Images Download' 라이브러리를 이용해 약 140명의 유명인 사진은 100장씩 크롤링한다.
2. `face crop`: 'haarcascade_frontalface_alt2.xml face detect' 검출기를 이용해 수집한 이미지에서 얼굴을 crop 한다.
3. `cheek detect`: 크롭된 이미지에서 'shape_predictor_68_face_landmarks.dat' 검출기를 이용하여 볼을 추출한다.
4. `skin color average`: 볼 추출 부분의 평균색을 이미지로 나타내어 1차 데이터셋을 구성한다.
5. `personal color classification`: 각 퍼스널 컬러에 해당하는 피부색을 hsv 값으로 정량화한 기준을 토대로 1차 데이터셋을 라벨링한다.
<br>

## 🔐 dataset
<img width="1172" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/7862d92f-f5f0-4b13-b4c9-ff36c1257314">

<br>
<br>

## 👑 model
<img width="1167" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/d6a25c06-6f30-4a71-8a09-6c4499a30646">

<br>
<br>

## 🪄 예측 결과
<img width="1171" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/d611798d-5977-4263-b009-960c6b11aaa4">

<img width="1151" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/8b62b9d9-b371-4283-a872-9f057b2b11c8">

왼쪽은 cnn 모델이 예측한 퍼스널 컬러이고, 오른쪽은 볼 영역의 피부색을 추출해 만든 데이터를 hsv값을 통해 분류한 퍼스널 컬러 입니다. 

<br>
<br>

<img width="842" alt="image" src="https://github.com/zer0silver/PersonalColor-Prediction/assets/78026977/c7bee1c5-996c-4a0e-9896-1d6448fc52cc">
test data로 예측한 결과입니다.




