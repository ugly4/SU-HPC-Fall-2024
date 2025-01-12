# Лабораторная работа 3
## Ray tracing on GPU

Программа представляет собой реализацию трассировки лучей (ray tracing) с использованием CUDA для ускорения вычислений на GPU. Программа генерирует изображение, состоящее из сфер и плоскости, с учетом освещения и теней.

## Основные функции
<li> save_bmp -  Сохраняет изображение в формате BMP.
<li> renderKernel - CUDA ядро, которое выполняет трассировку лучей для каждого пикселя изображения.
<li> traceRay - Функция, которая выполняет трассировку луча для заданного направления.

## Используемые структуры данных
<li> Vec3: Структура для представления трехмерного вектора.
<li> Sphere: Структура для представления сферы с центром, радиусом и цветом.
<li> Light: Структура для представления источника света с положением и интенсивностью.
<li> Plane: Структура для представления плоскости с точкой, нормалью и цветом.

