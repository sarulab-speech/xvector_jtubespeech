from xvector_jtube import XVector

model = XVector()
x = model("../sample.wav")
print(x.shape)

