import cv2
import numpy as np
from collections import deque
from scipy.stats import chisquare

#   CAPTURANDO O VÍDEO
video = cv2.VideoCapture('./video/dados.mp4')
video.set(15, -4)

#   SETANDO CONSTANTES PARA OS PARÂMETROS DO DETECTOR DO VÍDEO
detectorParams = cv2.SimpleBlobDetector_Params()

min_threshold = 10
detectorParams.minThreshold = min_threshold

max_threshold = 200
detectorParams.maxThreshold = max_threshold

min_area = 60
detectorParams.minArea = min_area

min_circularity = 0.2
detectorParams.minCircularity = min_circularity

min_inertia_ratio = 0.55
detectorParams.minInertiaRatio = min_inertia_ratio

detectorParams.filterByColor = False
detectorParams.filterByArea = True
detectorParams.filterByCircularity = True
detectorParams.filterByInertia = True


#   CRIANDO O DETECTOR
detector = cv2.SimpleBlobDetector_create(detectorParams)


#   CRIANDO AS VARIÁVEIS AUXILIARES PARA TRABALHAR OS DADOS
counter = 0
readings = deque([0, 0, 0, 0, 0, 0, 0, 0, 0,], maxlen=10)
display = deque([0, 0], maxlen=10)

lados = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}

#   FAZENDO A LEITURA
i = 1
while True:
    #   LENDO OS FRAMES DO VÍDEO
    ret, im = video.read()

    #   CASO NÃO TENHA MAIS FRAMES
    if not ret:
        break
    if ret == True:
        #   PONTOS DO VÍDEO QUE INTERESSAM PARA O TRABALHO
        keypoints = detector.detect(im)

        if len(keypoints):
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.namedWindow('dado', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("dado", 360, 640)
            cv2.imshow("dado", im_with_keypoints)

        if counter % 2 == 0:
            #   LENDO OS DADOS DO FRAME DO VÍDEO
            reading = len(keypoints)
            readings.append(reading)

            if readings[-1] == readings[-2] == readings[-3] == readings[-4]== readings[-5]== readings[-6]== readings[-7]== readings[-8]== readings[-9]:
                display.append(readings[-1])

            if display[-1] != display[-2] and display[-1] != 0:
                msg = f"Resultado [{i}] => {display[-1]}"
                lados[display[-1]] = lados[display[-1]] + 1
                i = i + 1
                print(msg)

        counter += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


values = i
values = list(lados.values())
expected_values = np.full(6, (i-1) / 6).tolist()

chisq, p = chisquare(values, f_exp = expected_values)

if p <= 0.05:
    dado = 'eh viciado'
else:
    dado = 'nao eh viciado'
print('\nDados obtidos para cada lado: ')
print(lados)
print(f'\nA porcentagem de vício do dado é de { (round(p, 6)) }, portanto o dado é {dado}\n\n')


cv2.destroyAllWindows()