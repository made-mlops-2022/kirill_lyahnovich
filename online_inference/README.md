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

### Docker image optimisation:
I tried to build web service inside a container with minimal libs possible,
so I tried to start with `alpine` series base images.
This attempt failed fo some reason, so I decided to retry with slightly more stuffed 
base image - `slim` series(and took `bullseye` debian distribution since it's pretty fresh)
I had success. And started looking for more ways to reduce the size of image. 
That lead me to use `pip install` with `--no-cache-dir`. 
And of course, I've installed only libs from `requirements.txt`



