from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import cv2

workdir = "./"

modelPath = f"{workdir}model_edgetpu.tflite"
labelPath = f"{workdir}labels.txt"

# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)

def main():
    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    
    labels = read_label_file(labelPath)

    image = cv2.imread(f"{workdir}img.jpg")

    results = classifyImage(interpreter, image)

    print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        

if __name__ == '__main__':
    main()