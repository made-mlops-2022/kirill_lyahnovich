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
python ml\run_pipeline.py configs\train\fast_config.yaml
~~~

Run Predict
~~~
python ml\predict.py configs\predict\predict_config.yml
~~~

Run Test:
~~~
python -m pytest tests/
~~~

Run inference: 
~~~
uvicorn api.run_service:app
~~~

Run inference in docker container: 
~~~
# to pull from dockerhub https://hub.docker.com/repository/docker/klyahnovich/inference_container
docker run --name inference -p 8000:8000 klyahnovich/inference_container:v1
~~~
or
~~~
./build_docker.bash
./run_docker.bash
~~~

