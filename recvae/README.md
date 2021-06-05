# RecVAE
모델 갱신 및 생성(GPU 미사용시 느릴 수 있음)
데이터 전처리 python preprocessing.py --dataset <dataset_dir> --output_dir <전처리 된 dataset_dir> --mode "train" // 모델 생성시 필요한 데이터셋 생성, 최소 5개 이상의 스탬프를 받은 사용자만이 데이터로 사용됨
모델 테스트 python run.py --dataset <dataset_dir> // './model' 폴더에 모델 생성됨

모델 적용
데이터 전처리 python preprocessing.py --dataset <dataset_dir> --model_dataset <모델 생성때 전처리 된 dataset_dir> --output_dir <dataset_dir> --mode "test" // model_dataset 필요, 모델 만들때 사용된 관광지만 추천가능
모델 시연 python run.py --dataset <dataset_dir> --mode="test" --topk=20 // topk에 넣은 인자만큼 n개의 상위 추천 여행지의 id 행렬 반환

협업필터링 기반이기 때문에 사용자는 최소 관광지를 4개 이상 방문해야함.

ex)
모델 생성
python preprocessing.py --dataset datasets/raw_data --output_dir "datasets/train" --mode "train"
python run.py --dataset "datasets/train"

모델 적용
python preprocessing.py --dataset datasets/raw_data --model_dataset "datasets/train" --output_dir "datasets/test" --mode "test"
python run.py --dataset "datasets/test" --mode="test" --topk=20