from fastapi import HTTPException

from api.API import API
from api.models import HealthInfo
from ml.schemes.inference_config import load_inference_config

INFERENCE_CONFIGS_PATH = './configs/inference/inference_configs.yml'

inference_params = load_inference_config(INFERENCE_CONFIGS_PATH)
app = API(inference_params)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def root():
    if app.inferencer is None:
        raise HTTPException(status_code=400, detail="Model is not ready")
    return {"message": "ok"}


@app.post("/api/predict")
async def predict(health: HealthInfo):
    print(health)
    response = app.inferencer.predict(health)
    print(response)
    return response

# @click.command(name="start_service")
# @click.argument("inference_config_path")
# def start_service_command(inference_config_path: str):
#     start_service(inference_config_path)
#
#
# def start_service(inference_config_path: str):
#
#     inferencer = Inferencer(inference_params)
#
#
# if __name__ == "__main__":
#     start_service()
