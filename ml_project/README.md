Heart disease detection ML project
==============================
`Author`: Kirill Lyahnovich

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Usage:

Run Train and save model
~~~
python ml\run_pipeline.py configs/train/fast_config.yaml
~~~

Run Predict
~~~
python ml\run_pipeline.py configs/predict/predict_config.yml
~~~

Run Test:
~~~
python -m pytest tests/
~~~
