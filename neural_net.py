import json
import numpy as np

# =====================================================================
# 1. ФУНКЦИЯ АКТИВАЦИИ: СИГМОИДА
# =====================================================================
def sigmoid(x):
    """
    Сигмоида — математическая функция активации: f(x) = 1 / (1 + e^(-x))
    Она сжимает любое входящее число (от -бесконечности до +бесконечности)
    в строгий диапазон от 0 до 1.
    Это превращает выход нейрона в подобие «степени уверенности» или вероятности.
    """
    return 1 / (1 + np.exp(-x))


# =====================================================================
# 2. БАЗОВЫЙ НЕЙРОН (ДЛЯ КЛАССИФИКАЦИИ И СКРЫТЫХ СЛОЕВ)
# =====================================================================
class Neuron:
    # __init__ — это специальный метод-конструктор в Python.
    # Он вызывается АВТОМАТИЧЕСКИ один раз в момент создания нового объекта: n = Neuron(4).
    # 'self' — это ссылка на сам создаваемый конкретный экземпляр нейрона.
    # 'input_size' — сколько входов приходит в этот нейрон (например, 4 признака).
    def __init__(self, input_size):
        # self.weights — веса связей (по одному числу на каждый вход).
        # Инициализируем случайными числами от -1 до 1.
        self.weights = np.random.uniform(-1, 1, input_size)
        
        # self.bias — смещение (свободный член / порог активации нейрона).
        self.bias = np.random.uniform(-1, 1)

    # feedforward — это обычный метод класса (действие, которое нейрон умеет выполнять).
    # Он не вызывается сам по себе, мы вызываем его вручную: n.feedforward([1, 2, 3, 4]).
    def feedforward(self, inputs):
        # np.dot(inputs, self.weights) — скалярное произведение векторов:
        # (x1*w1 + x2*w2 + x3*w3 + ...).
        # Прибавляем смещение bias: z = sum(w_i * x_i) + b
        total = np.dot(inputs, self.weights) + self.bias
        
        # Пропускаем взвешенную сумму через сигмоиду, чтобы получить значение от 0 до 1
        return sigmoid(total)


# =====================================================================
# 3. ЛИНЕЙНЫЙ НЕЙРОН (ДЛЯ ВЫХОДНОГО СЛОЯ РЕГРЕССИИ)
# =====================================================================
class LinearNeuron:
    """
    В отличие от обычного Neuron, здесь НЕТ сигмоиды!
    В задаче регрессии мы предсказываем не класс, а реальное непрерывное число
    (например, массу пингвина в граммах или градус алкоголя).
    Сигмоида бы обрезала результат максимум единицей (1.0).
    Поэтому функция активации линейная: f(z) = z, а производная равна 1.0.
    """
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        # Просто считаем формулу: total = (w1*x1 + w2*x2 + ...) + bias
        # и возвращаем число как float БЕЗ сигмоиды.
        return float(np.dot(inputs, self.weights) + self.bias)


# =====================================================================
# 4. ПОЛНОСВЯЗНАЯ СЕТЬ КЛАССИФИКАЦИИ (4 -> 8 -> 3)
# =====================================================================
class ClassificationNN:
    # Параметры по умолчанию:
    # input_size=4  (4 признака: длина клюва, толщина клюва, ласта, масса)
    # hidden_size=8 (8 нейронов в скрытом промежуточном слое)
    # output_size=3 (3 выходных нейрона — по одному на каждый вид пингвина)
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        # Конструкция [Neuron(...) for _ in range(...)] называется List Comprehension (генератор списка).
        # Это компактная запись цикла for в одну строчку:
        # Символ '_' (нижнее подчеркивание) — это переменная цикла, которая нам не нужна
        # (мы просто хотим повторить создание Neuron(input_size) ровно hidden_size раз).
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        
        # Выходной слой: 3 нейрона, каждый из которых принимает сигналы от всех 8 нейронов скрытого слоя
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        # 1. Прогоняем входной вектор через все 8 нейронов скрытого слоя.
        # Получаем массив 'h' из 8 чисел:
        h = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        
        # 2. Передаем массив 'h' на вход 3 нейронам выходного слоя:
        # Получаем 3 числа (степени уверенности в каждом из 3 видов):
        return np.array([n.feedforward(h) for n in self.output_layer])

    def predict(self, inputs):
        # np.argmax(...) находит индекс самого большого числа в массиве.
        # Например, если выходы [0.05, 0.89, 0.12], то максимальное 0.89 стоит на позиции 1.
        # Значит, предсказан класс 1 (Chinstrap).
        return int(np.argmax(self.feedforward(inputs)))


# =====================================================================
# 5. ПОЛНОСВЯЗНАЯ СЕТЬ РЕГРЕССИИ (3 -> 8 -> 1)
# =====================================================================
class RegressionNN:
    # Входных признаков 3 (длина клюва, толщина клюва, ласта).
    # А целевую массу тела мы предсказываем.
    def __init__(self, input_size=3, hidden_size=8):
        # Скрытый слой: 8 обычных нейронов с сигмоидой
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        
        # На выходе ровно ОДИН ЛИНЕЙНЫЙ нейрон (LinearNeuron)
        self.output_neuron = LinearNeuron(hidden_size)

    def feedforward(self, inputs):
        # 1. Прогоняем входы через 8 нейронов скрытого слоя:
        h = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        
        # 2. Передаем 8 скрытых сигналов в один выходной линейный нейрон:
        return self.output_neuron.feedforward(h)


# =====================================================================
# 6. ФУНКЦИИ ЗАГРУЗКИ ОБУЧЕННЫХ ВЕСОВ ИЗ JSON В ОБЪЕКТЫ СЕТЕЙ
# =====================================================================
def load_classification_model(path):
    """
    Загружает веса классификатора из JSON-файла и воссоздает обученную сеть.
    Здесь нет никакого обучения — только чтение сохраненных чисел!
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Создаем пустую сеть с нужной архитектурой
    net = ClassificationNN(4, 8, 3)
    
    # 2. Функция zip(...) берет два списка и объединяет их попарно:
    # (1-й нейрон сети, 1-й словарь весов из JSON), (2-й нейрон, 2-й словарь весов) и т.д.
    # Так мы копируем обученные веса в каждый нейрон скрытого слоя:
    for neuron, params in zip(net.hidden_layer, data["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    # Точно так же заполняем веса для 3 выходных нейронов:
    for neuron, params in zip(net.output_layer, data["output"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    # Достаем сохраненные минимумы и максимумы (нужны для нормализации новых данных пользователя)
    mn = np.array(data["mn"])
    mx = np.array(data["mx"])
    class_names = data.get("class_names") or data.get("names")
    
    return net, mn, mx, class_names


def load_regression_model(path):
    """
    Загружает веса регрессора из JSON-файла и параметры денормализации.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Создаем пустую сеть регрессии (3 входа -> 8 скрытых -> 1 линейный выход)
    net = RegressionNN(3, 8)
    
    # Заполняем веса скрытого слоя:
    for neuron, params in zip(net.hidden_layer, data["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    # Заполняем веса одного выходного линейного нейрона:
    net.output_neuron.weights = np.array(data["output"]["weights"])
    net.output_neuron.bias = data["output"]["bias"]

    # mn_x, mx_x — для нормализации входов пользователя [0, 1]
    # mn_y, mx_y — для ДЕНОРМАЛИЗАЦИИ выхода сети обратно в реальные граммы
    mn_x = np.array(data["mn_x"])
    mx_x = np.array(data["mx_x"])
    mn_y = float(data["mn_y"])
    mx_y = float(data["mx_y"])
    
    return net, mn_x, mx_x, mn_y, mx_y
