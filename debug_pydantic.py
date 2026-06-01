import numpy as np
from pydantic import BaseModel

class TestModel(BaseModel):
    val: int = 0

print("int(5.0):", TestModel(val=5.0))
print("int(np.float64(5.0)):", TestModel(val=np.float64(5.0)))
print("int(True):", TestModel(val=True))

try:
    print("int('5'):", TestModel(val="5"))
except Exception as e:
    print("int('5') FAILED:", type(e).__name__, e)

try:
    print("int(float('nan')):", TestModel(val=float("nan")))
except Exception as e:
    print("int(float('nan')) FAILED:", type(e).__name__, e)

try:
    print("int(np.float64('nan')):", TestModel(val=np.float64("nan")))
except Exception as e:
    print("int(np.float64('nan')) FAILED:", type(e).__name__, e)
