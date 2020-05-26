from ml import LinearModel
from fake_cut import Fake_Cut
from objects import EndMill, Conditions

PARAMS = [858494934.9591976, -696.3150933941946, 858494934.9591976, -696.3150933941946]
# ERROR = [0.01, 0.01]
# NOISE = [0.1, 0.1]

ERROR = [0, 0]
NOISE = [0, 0]

CONDITIONS = Conditions(1e-3, 3.175e-3, 5e-3, 5000, EndMill(3, 3.175e-3, 3.175e-3, 10e-3, 10e-3))

model = LinearModel()

cut = Fake_Cut(PARAMS, ERROR, NOISE)

data = cut.cut(CONDITIONS)

model.ingest_datum(data)

print("{0:.20f}".format(model.params[1]))