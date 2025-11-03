import cv2

net = cv2.dnn.readNetFromCaffe("Data/MobileNet/mobilenet_v2.caffemodel", 'mobilenet_v2_deploy.prototxt')#завантажуємо модель


classes = []
with open('Data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)
#зчитуємо список назв класів


image = cv2.imread('Data/MobileNet/cat.jpg')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#адаптуємо зоюраження під модель


net.setInput(blob)#підготовлені файли

preds = net.forward()#вектор імовірності для класів



















