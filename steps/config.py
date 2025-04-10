from typing import Optional
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "xgboost"  
    save_dir: Optional[str] = "saved_model"  