import gradio as gr
import torch
from Model import LeNet

labels = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight', 'Nine']

# Locate device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

# Loading model
model = LeNet().to(device)
model.load_state_dict(torch.load("model_mnist.pth", map_location=torch.device('cpu')))


def predict(input):
  input = torch.from_numpy(input.reshape(1, 1, 28, 28)).to(dtype=torch.float32, device=device)

  with torch.no_grad():
    outputs = model(input)  
    prediction = torch.nn.functional.softmax(outputs[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(10)}    
  return confidences

gr.Interface(title='Digit classifier', fn=predict, 
             inputs="sketchpad",
             outputs=gr.Label(num_top_classes=3)).launch(share=False, debug=True)