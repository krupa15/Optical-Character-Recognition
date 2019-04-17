from data_util import DataLoader
from model import OCRnet
datadir="./Dataset"
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

print("----- data loading -----")
loader=DataLoader(datadir,classes)
print("----- Model creating -----")
ocrnet=OCRnet(loader.GetNumClasses(),classes)
print("----- Model -----")
print(ocrnet.model)
x,y=loader.GetTrainingData()
val_x,val_y=loader.GetTestingData()
print("----- tarining started -----")
ocrnet.train_model(epoch=5,batch_size=100,X_train=x,Y_train=y,X_test=val_x,Y_test=val_y)
print("----- model saved -----")
ocrnet.save_model("./ocrnetmodel2.model")
