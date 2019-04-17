from data_util import DataLoader
from model import OCRnet
datadir="./Dataset2"
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

print("----- data loading -----")
loader=DataLoader(datadir,classes)
print("----- Model creating -----")
ocrnet=OCRnet(loader.GetNumClasses(),classes)
ocrnet.load_model("./ocrnetmodel2.model")
print("----- Model -----")
print(ocrnet.model)
x,y=loader.GetTrainingData()
val_x,val_y=loader.GetTestingData()
print("----- testing started -----")
print(ocrnet.test_model(x,y))
