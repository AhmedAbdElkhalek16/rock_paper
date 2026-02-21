from roboflow import Roboflow
import os
import dotenv
dotenv.load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("arla").project("rock-paper-scissors-sxsw-rswkh")
version = project.version(1)
dataset = version.download("yolo26")