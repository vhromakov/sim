import pymeshlab as ml
from pymeshlab import pmeshlab  # gives you pmeshlab.PercentageValue

ms = ml.MeshSet()
ms.load_new_mesh("MODELS/CSC16_U00P_.stl")

ms.generate_alpha_wrap(
    alpha=pmeshlab.PercentageValue(2.0),   # 2% of bbox diagonal
    offset=pmeshlab.PercentageValue(0.2)   # 0.2% of bbox diagonal
)

ms.save_current_mesh("wrapped.stl")
