# 실행환경 설정
```sh
conda create -n tf-clean python=3.10 -y
conda activate tf-clean



pip install -r requirements.txt
conda install -c conda-forge xgboost lightgbm shap prophet
```


# 실행방법
- 아래 순으로 전체 실행
1. 데이터수집.ipynb
2. 데이터전처리.ipynb
3. 통계분석요인검정.ipynb
4. 머신러닝_예측모델_구축.ipynb

- 터미널에 아래 명령어 실행
```sh
streamlit run ./src/app.py
```
