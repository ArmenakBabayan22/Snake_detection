import pickle
import cv2
import numpy as np

# Метки классов
labels = ['corn_snake', 'milk_snake', 'California_kingsnake', 'grass_snake']

# Загружаем модель
with open('model.odz.pkl', 'rb') as file:
    model = pickle.load(file)

# Включаем камеру
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # ---------- ПОДГОТОВКА ИЗОБРАЖЕНИЯ ----------
    # Переводим BGR → RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Меняем размер под модель (ширина=220, высота=159)
    image = cv2.resize(image, (220, 159))

    # Нормализация (0..1)
    image = image / 255.0

    # Добавляем batch → (1, 159, 220, 3)
    image = np.expand_dims(image, axis=0)

    # ---------- ПРЕДСКАЗАНИЕ ----------
    y_pred = model.predict(image)
    emotion = labels[np.argmax(y_pred)]
    prob = np.max(y_pred)

    # ---------- ВЫВОД ----------
    text = f"{emotion} ({prob:.2f})"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Показываем камеру
    cv2.imshow("Camera", frame)

    # Выход по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()