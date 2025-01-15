import queue
import sys
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
import threading
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
from math import floor,sqrt,atan,atan2,degrees,exp
from plotly.graph_objs import *
from plotly.graph_objs.layout import Annotation
from copy import deepcopy
from ACO import ACO




class Node:
    def __init__(self, x, y, start_angle,end_angle, f_score, g_score, parent=None, used_block=None, allowed_intersections=None, avalible_blocks=None):
        self.x = x
        self.y = y
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.f = f_score
        self.g = g_score
        self.parent = parent
        self.block = used_block
        self.avalible_blocks = avalible_blocks or {'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0, 'T4': 0, 'T8': 0, 'B1': 0}
        self.allowed_intersections = allowed_intersections or []

    def __gt__(self, other):
        return self.f > other.f

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y}, f={self.f}, g={self.g})"

    def reconstruct_path(self):
        path = []
        start_ang = 0
        node = self
        while node:
            if node.block:
                path.append(node.block)
                start_ang= node.start_angle
            node = node.parent
            # print(node)
            # print(start_ang)
        return path[::-1],start_ang



# Класс для хранения информации об элементах
class Element:
    def __init__(self, type_, count, cost):
        self.type = type_
        self.count = int(count)
        self.cost = int(cost)

    def __repr__(self):
        return f"(Type={self.type}, count={self.count}, cost={self.cost})"

# Класс для хранения информации о точках маршрута
class RoutePoint:
    def __init__(self, x, y, value):
        self.x = int(x)
        self.y = int(y)
        self.value = int(value)
    def __repr__(self):
        return f"({self.x}, {self.y}), value={self.value}"

class MyExc(Exception):
    pass


def move(block, x, y, angle, direction=1, intermediate=0):
    if block.startswith('L'):
        new_x, new_y, new_angle = move_L(x, y, angle, int(block[1]), intermediate)
    elif block == 'T4':
        new_x, new_y, new_angle = move_T4(x, y, angle, direction, intermediate)
    elif block == 'T8':
        new_x, new_y, new_angle = move_T8(x, y, angle, direction, intermediate)
    elif block == 'B1':
        new_x, new_y, new_angle = move_B1(x, y, angle, intermediate)
    else:
        print('передан некорректный блок в move')

    return new_x, new_y, new_angle

# Функция для чтения данных из файла
def read_input(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    elements = []
    route = []
    order = []

    current_block = None

    for line in lines:
        line = line.strip()
        if line.startswith('--') or not line:
            continue
        if line == 'DATA':
            current_block = 'DATA'
            continue
        elif line == 'ROUTE':
            current_block = 'ROUTE'
            continue
        elif line == 'ORDER':
            current_block = 'ORDER'
            continue
        elif line == '/':
            current_block = None
            continue

        if current_block == 'DATA':
            parts = line.split('--')[0].split()
            elements.append(Element(parts[0], parts[1], parts[2]))
        elif current_block == 'ROUTE':
            parts = line.split('--')[0].split()
            if not route:
                if int(parts[0]) or int(parts[1]):
                    raise ValueError(f"Первая точка в ROUTE не (0,0)")
            route.append(RoutePoint(parts[0], parts[1], parts[2]))
        elif current_block == 'ORDER':
            parts = line.split('--')[0].split()
            order.append((parts[0], int(parts[1])))

    route.append(RoutePoint(0, 0, 0))  # Убедиться, что маршрут возвращается в (0,0)

    return elements, route, order

# Функции для вычисления координат при использовании различных блоков
def move_L(x, y, angle, length, intermediate=0):
    if intermediate:
        length = np.linspace(0,length,5*length)
    x_new = np.array(x + length * np.cos(angle))
    y_new = np.array(y + length * np.sin(angle))
    return x_new, y_new, angle

def move_T4(x, y, angle, direction, intermediate=0):
    radius = 3  # Радиус поворота
    angle_turn = np.pi / 4  # Угол поворота
    direction = -direction

    # Вычисляем новый угол
    new_angle = angle + direction * angle_turn

    # Вычисляем координаты центра окружности, вокруг которой будет поворот
    center_x = x + radius * np.cos(angle + np.pi / 2 * direction)
    center_y = y + radius * np.sin(angle + np.pi / 2 * direction)

    theta = new_angle - np.pi / 2 * direction

    if intermediate:
        theta = np.linspace(new_angle - np.pi / 2 * direction + angle_turn * (-direction),
                            new_angle - np.pi / 2 * direction,
                            15)

    # Вычисляем новые координаты
    x_new = np.array(center_x + radius * np.cos(theta))
    y_new = np.array(center_y + radius * np.sin(theta))
    return x_new, y_new, new_angle


def move_T8(x, y, angle, direction, intermediate=0):
    radius = 3  # Радиус поворота
    angle_turn = np.pi / 8  # Угол поворота
    direction = -direction

    # Вычисляем новый угол
    new_angle = angle + direction * angle_turn

    # Вычисляем координаты центра окружности, вокруг которой будет поворот
    center_x = x + radius * np.cos(angle + np.pi / 2 * direction)
    center_y = y + radius * np.sin(angle + np.pi / 2 * direction)

    theta = new_angle - np.pi / 2 * direction

    if intermediate:
        theta = np.linspace(new_angle - np.pi / 2 * direction + angle_turn * (-direction),
                            new_angle - np.pi / 2 * direction,
                            10)

    # Вычисляем новые координаты
    x_new = np.array(center_x + radius * np.cos(theta))
    y_new = np.array(center_y + radius * np.sin(theta))
    return x_new, y_new, new_angle


def move_B1(x, y, angle, intermediate=0):
    if intermediate:
        length = np.linspace(0, 4, 3)
    else:
        length = 4
    x_new = np.array(x + length * np.cos(angle))
    y_new = np.array(y + length * np.sin(angle))
    return x_new, y_new, angle

def distance(new_x, new_y, target_x, target_y):
    return sqrt((new_x - target_x) ** 2 + (new_y - target_y) ** 2)





def angle_between_lines(p1, p2, p3, p4):

    # Calculate slopes (m1, m2)
    try:
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    except ZeroDivisionError:
        m1 = float('inf')  # Vertical line

    try:
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    except ZeroDivisionError:
        m2 = float('inf')  # Vertical line

    # Handle special cases for vertical lines
    if m1 == float('inf') and m2 == float('inf'):
        return 0  # Parallel vertical lines
    elif m1 == float('inf') or m2 == float('inf'):
        return 90  # One vertical line

    # Calculate the tangent of the angle between the lines
    try:
        tan_theta = abs((m1 - m2) / (1 + m1 * m2))
    except ZeroDivisionError:
        return 90  # Perpendicular lines

    # Calculate the angle in radians and convert to degrees
    angle_radians = atan(tan_theta)
    angle_degrees = degrees(angle_radians)

    return angle_degrees

def is_dot_visited_node(node, tx, ty, threshold=0.5):

    parent = node.parent
    if not parent is None:
        x_all, y_all, new_angle = move(node.block[0], parent.x, parent.y, node.start_angle, node.block[1],1)
        if x_all.size==1 and y_all.size==1:
            if distance(x_all, y_all, tx, ty) <= threshold:
                return True
        else:
            for x,y in zip(x_all, y_all):
                if distance(x, y, tx, ty) <= threshold:
                    return True
    return False

def is_dot_visited(x_arr, y_arr, tx, ty, threshold=0.5):
    for x,y in zip(x_arr, y_arr):
        if distance(x, y, tx, ty) <= threshold:
            return True
    return False


def is_self_intersecting_incremental(old_path, new_segment, grid, allowed, cell_size=1):
    """Проверяет пересечения, используя предыдущий маршрут и сетку, а так же обрабатывая новый отрезок."""
    if old_path:
        segments = []
        for i in range(len(old_path) - 1):
            p1, p2 = old_path[i], old_path[i + 1]
            segments.append((p1, p2))

        new_grid = {k: v.copy() for k, v in grid.items()}
        new_segments_with_start = [old_path[-1]] + new_segment

        # Проверка самопересечений в new_segment
        for i in range(len(new_segments_with_start) - 1):
            for j in range(i + 2, len(new_segments_with_start) - 1):
                if intersects(new_segments_with_start[i], new_segments_with_start[i + 1], new_segments_with_start[j],
                              new_segments_with_start[j + 1]):
                    if intersects_and_allowed(new_segments_with_start[i], new_segments_with_start[i + 1],
                                              new_segments_with_start[j], new_segments_with_start[j + 1], allowed):
                        return (True, (new_segments_with_start[i], new_segments_with_start[i + 1],
                                       new_segments_with_start[j], new_segments_with_start[j + 1]))
        for seg_index, _ in enumerate(new_segments_with_start[:-1]):
            new_segment_start = new_segments_with_start[seg_index]
            new_segment_end = new_segments_with_start[seg_index + 1]
            new_segment_index = len(segments)
            segments.append((new_segment_start, new_segment_end))

            # Обновление сетки для нового отрезка
            for cell in bbox_key(new_segment_start, new_segment_end, cell_size):
                if cell not in new_grid:
                    new_grid[cell] = []
                new_grid[cell].append(new_segment_index)

            # Проверка пересечений нового отрезка со старыми
            for cell in bbox_key(new_segment_start, new_segment_end, cell_size):
                if cell in new_grid:
                    for i_idx in range(len(new_grid[cell])):
                        i = new_grid[cell][i_idx]
                        if i == new_segment_index:
                            continue
                        s1, s2 = segments[i], segments[new_segment_index]
                        if intersects(s1[0], s1[1], s2[0], s2[1]):
                            if intersects_and_allowed(s1[0], s1[1], s2[0], s2[1], allowed):
                                return (True, (s1[0], s1[1], s2[0], s2[1]))

    return (False, None)

def find_intersection_point(p1, q1, p2, q2):
    """
    Находит точку пересечения двух отрезков.
    p1, q1 — концы первого отрезка.
    p2, q2 — концы второго отрезка.
    Возвращает координаты точки пересечения (px, py) или (None, None), если пересечения нет.
    """
    x1, y1 = p1
    x2, y2 = q1
    x3, y3 = p2
    x4, y4 = q2

    # Вычисление знаменателя
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:  # Линии параллельны
        return None, None

    # Вычисление координат точки пересечения
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return px, py

def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def intersects(p1, q1, p2, q2):
    """Проверка пересечения двух отрезков"""
    return (ccw(p1, p2, q2) * ccw(q1, p2, q2) <= 0) and (ccw(p1, q1, p2) * ccw(p1, q1, q2) < -EPSILON)

def intersects_and_allowed(p1, q1, p2, q2, allowed):
    intersection = find_intersection_point(p1, q1, p2, q2)
    if intersection == (None, None):
        return False
    return not any(abs(intersection[0] - a[0]) < EPSILON and abs(intersection[1] - a[1]) < EPSILON for a in allowed)


def bbox_key(p1, p2, cell_size):
    """Создаем ключи для ячеек сетки, покрывающих отрезок"""
    x_min, x_max = sorted([p1[0], p2[0]])
    y_min, y_max = sorted([p1[1], p2[1]])
    keys = []
    x_start, x_end = floor(x_min / cell_size), floor(x_max / cell_size)
    y_start, y_end = floor(y_min / cell_size), floor(y_max / cell_size)

    if x_start == x_end:
        for y in range(y_start, y_end + 1):
            keys.append((x_start, y))
    elif y_start == y_end:
        for x in range(x_start, x_end + 1):
            keys.append((x, y_start))
    else:
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                keys.append((x, y))
    return keys

def make_grid(path):
    global cell_size
    grid = {}
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        for cell in bbox_key(p1, p2, cell_size=cell_size):
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)
    return grid



def angle_difference(current_coords, current_angle, next_coords):

    x1, y1 = current_coords
    x2, y2 = next_coords

    # Вычисление угла направления до следующей точки
    angle_to_next = atan2(y2 - y1, x2 - x1)

    # Вычисление разницы углов
    angle_diff = (angle_to_next - current_angle + np.pi) % (np.pi*2) - np.pi

    return angle_diff


def can_achieve_angle(angle, blocks_22_5, blocks_45):
    """
    Оптимизированная проверка возможности составления угла.

    :param angle: Угол, который нужно получить (в градусах).
    :param blocks_22_5: Количество доступных блоков поворота на 22.5 градусов.
    :param blocks_45: Количество доступных блоков поворота на 45 градусов.
    :return: True, если угол можно составить, иначе False.
    """

    # Максимально возможный угол, который можно составить
    max_angle = blocks_22_5 * 22.5 + blocks_45 * 45
    if abs(angle) > max_angle:
        return False

    return True

def build_route(elements, route, queue, second_queue):
    def heuristic(x, y, target_x, target_y):
        return round(distance(x, y, target_x, target_y)*10)

    def calculate_y(x, C=1000, alpha=0.5, L=1):
        # Экспоненциальная формула с минимальным значением L
        y = C * exp(-alpha * x)
        return max(y, L)

    elements_without_b1 = elements.copy()
    del elements_without_b1[6]
    route2 = route[1:]
    # global full_dots
    global debug
    global enable_limit_nodes
    global threshold
    global cell_size
    global enable_next_point_penalty
    start_angle = 0
    grid = {}
    path = []
    path_dots= []
    avalible_elements = {e.type: e.count for e in elements}
    elements_price = {e.type: e.cost for e in elements}
    print(f"Доступные блоки: {avalible_elements}")
    print(f"Цены на блоки: {elements_price}")
    x, y, angle = 0, 0, 0
    allowed_intersections = [(0, 0)]
    start_time = 0
    for i, target in enumerate(route2, start=1):
        print(
            f"\nЦелевая точка {i}/{len(route2)}: (x={target.x}, y={target.y})  dist={distance(target.x, target.y, x, y):.2f}")

        open_set = []
        if i ==1:
            start_node = Node(x, y, start_angle, angle, 0, 0, None, None, allowed_intersections, avalible_elements)
        else:
            start_node = curr_node

        heapq.heappush(open_set, start_node)
        closed_set = set()
        found = False

        while open_set:
            curr_node = heapq.heappop(open_set)


            time.sleep(1e-12)
            queue.put([i, curr_node.x, curr_node.y, len(open_set), time.time()-start_time])
            start_time = time.time()

            if is_dot_visited_node(curr_node, target.x, target.y, threshold=threshold):
                # node_path,start_angle_tmp = curr_node.reconstruct_path()

                node_path,start_angle = curr_node.reconstruct_path()

                avalible_elements = deepcopy(curr_node.avalible_blocks)
                # old_dots = visited_dots(path, start_angle)
                dots = visited_dots(node_path, start_angle)
                # new_grid = make_grid(dots)
                is_intersect, _ =  is_self_intersecting_incremental(path_dots,dots[len(path_dots):],grid, curr_node.allowed_intersections,cell_size)  # проверям на пересечения в пути
                if is_intersect:
                    print(f'[WARNING] в найденном решении обнаружено перeсечение (такого быть не должно, надо проверять алгоритм)')
                    continue

                print(
                    f"  Завершен путь к точке {i}: (x={curr_node.x:.2f}, y={curr_node.y:.2f}, ang={round(degrees(curr_node.end_angle)%360, 2)})")

                # second_queue.put([path])
                path = deepcopy(node_path)
                path_dots = deepcopy(dots)
                grid = make_grid(dots)
                x, y, angle = curr_node.x, curr_node.y, curr_node.end_angle
                allowed_intersections = deepcopy(curr_node.allowed_intersections)

                point_index2 = 0
                for dot in dots:
                    dot.append(point_index2)
                    point_index2 += 1

                time.sleep(1e-6)
                second_queue.put([dots])
                found = True
                break

            rounded_x = np.round(curr_node.x, 1)
            rounded_y = np.round(curr_node.y, 1)
            rounded_angle = np.round(degrees(curr_node.end_angle))  # Угол в градусах для стабильности

            # Преобразуем в целочисленный формат
            state = (int(rounded_x * 10), int(rounded_y * 10), int(rounded_angle))

            # Пропускаем узел, если он уже обработан и не в разрешенных пересечениях
            if state in closed_set:
                continue




            closed_set.add(state)  # Добавляем только после всех проверок

            current_avalible_elements = curr_node.avalible_blocks

            # Перебираем доступные блоки
            for block in elements_without_b1:

                if current_avalible_elements[block.type] <= 0:
                    continue

                if i == 1:
                    if curr_node.parent is None:  # если первый блок - можем выбрать любой угол
                        for test_angle in np.linspace(0, 2 * np.pi, 17)[:-1]:  # 17 - шаг 22.5 град, 9 - 45град
                            for direction in [-1, 1] if block.type in ['T4', 'T8'] else [1]:

                                new_x, new_y, new_angle = move(block.type, curr_node.x, curr_node.y, test_angle,direction)

                                # чем больше отклонение угла до целевой точки, тем больше (от 0 до 1)
                                angle_penalty = (1 - ((np.pi - abs(angle_difference((new_x, new_y), new_angle,
                                                                                    (target.x, target.y)))) / np.pi))

                                new_g_score = curr_node.g + block.cost + angle_penalty * 3
                                new_h_score = heuristic(new_x, new_y, target.x, target.y)
                                new_f_score = new_g_score + new_h_score
                                # new_start_angle = test_angle

                                if len(open_set) > 200000 and enable_limit_nodes:
                                    print(f"превышен лимит поиска, точка: {i} ({target.x, target.y})")
                                    last_path, start_angle_tmp = curr_node.reconstruct_path()
                                    dots = visited_dots(last_path, start_angle_tmp)
                                    print(last_path)
                                    print([[x.tolist() if isinstance(x, np.ndarray) else x for x in pair] for pair in dots])
                                    print()

                                    # print(full_dots)
                                    # plot_path_on_error(full_dots)
                                    sys.exit()

                                new_node = Node(
                                    x=new_x, y=new_y,
                                    start_angle=test_angle,
                                    end_angle=new_angle,
                                    f_score=new_f_score,
                                    g_score=new_g_score,
                                    parent=curr_node,
                                    used_block=(block.type, direction),
                                    allowed_intersections=allowed_intersections,
                                    avalible_blocks=deepcopy(current_avalible_elements)
                                )

                                new_node.avalible_blocks[block.type] -= 1
                                heapq.heappush(open_set, new_node)
                    else:
                        for direction in [-1, 1] if block.type in ['T4', 'T8'] else [1]:

                            new_x, new_y, new_angle = move(block.type, curr_node.x, curr_node.y, curr_node.end_angle,direction)

                            # чем больше отклонение угла до целевой точки, тем больше (от 0 до 1)
                            angle_penalty = (1 - ((np.pi - abs(angle_difference((new_x, new_y), new_angle,
                                                                                (target.x, target.y)))) / np.pi))

                            new_g_score = curr_node.g + block.cost + angle_penalty * 3
                            new_h_score = heuristic(new_x, new_y, target.x, target.y)
                            new_f_score = new_g_score + new_h_score
                            # new_start_angle = test_angle

                            if len(open_set) > 200000 and enable_limit_nodes:
                                print(f"превышен лимит поиска, точка: {i} ({target.x, target.y})")
                                last_path, start_angle_tmp = curr_node.reconstruct_path()
                                dots = visited_dots(last_path, start_angle_tmp)
                                print(last_path)
                                print([[x.tolist() if isinstance(x, np.ndarray) else x for x in pair] for pair in dots])
                                print()

                                # print(full_dots)
                                # plot_path_on_error(full_dots)
                                sys.exit()

                            new_node = Node(
                                x=new_x, y=new_y,
                                start_angle=curr_node.end_angle,
                                end_angle=new_angle,
                                f_score=new_f_score,
                                g_score=new_g_score,
                                parent=curr_node,
                                used_block=(block.type, direction),
                                allowed_intersections=allowed_intersections,
                                avalible_blocks=deepcopy(current_avalible_elements)
                            )

                            new_node.avalible_blocks[block.type] -= 1
                            heapq.heappush(open_set, new_node)
                else:
                    for direction in [-1, 1] if block.type in ['T4', 'T8'] else [1]:
                        new_x, new_y, new_angle = move(block.type, curr_node.x, curr_node.y, curr_node.end_angle,direction)

                        # чем больше отклонение угла до целевой точки, тем больше (от 0 до 1)
                        angle_penalty = (1 - ((np.pi - abs(angle_difference((new_x, new_y), new_angle,
                                                                            (target.x, target.y)))) / np.pi))

                        try:
                            # чем больше отклонение угла до CЛЕДУЮЩЕЙ точки маршрута, тем больше (от 0 до 1)
                            angle_to_the_next_point_penalty = (1 - ((np.pi - abs(angle_difference((new_x, new_y), new_angle,
                                                                            (route2[i].x, route2[i].y)))) / np.pi))
                            distance_target_and_next = round(distance(target.x, target.y, route2[i].x, route2[i].y))
                        except IndexError:
                            angle_to_the_next_point_penalty = 0
                            distance_target_and_next = 0


                        if target.x and target.y: # если target != (0,0)
                            new_g_score = curr_node.g + block.cost + angle_penalty * 20
                            new_h_score = heuristic(new_x, new_y, target.x, target.y) + angle_to_the_next_point_penalty * calculate_y(distance_target_and_next)*enable_next_point_penalty

                        else:
                            new_g_score = curr_node.g + block.cost + angle_penalty * 15
                            new_h_score = heuristic(new_x, new_y, target.x, target.y)


                        new_f_score = new_g_score + new_h_score

                        node_path, start_angle_tmp = curr_node.reconstruct_path()

                        new_path = node_path +[(block.type, direction)]

                        dots = visited_dots(new_path, start_angle_tmp)
                        new_allowed_intersections = deepcopy(curr_node.allowed_intersections)

                        is_intersect, coord = is_self_intersecting_incremental(path_dots, dots[len(path_dots):], grid, curr_node.allowed_intersections, cell_size)
                        if is_intersect:  # если есть
                            if avalible_elements['B1'] <= 0:
                                continue
                            # intersect_block_type = new_path[dots.index(coord[0])][0]
                            # intersect_angle = calculate_angle_between_lines(*coord)
                            new_x, new_y, new_angle = move('B1', curr_node.x, curr_node.y, curr_node.end_angle,
                                                           direction)
                            # new_path = curr_node.path + [
                            #     ('B1', direction, int(block.type[1]) if block.type.startswith('L') else None)]

                            # last_path, start_angle_tmp = curr_node.reconstruct_path()
                            new_path = node_path + [('B1', direction)]
                            dots = visited_dots(new_path, start_angle_tmp)
                            is_intersect, coord = is_self_intersecting_incremental(path_dots, dots[len(path_dots):],
                                                                                   grid,
                                                                                   curr_node.allowed_intersections,
                                                                                   cell_size)

                            if not is_intersect:
                                continue
                            intersection_point = find_intersection_point(*coord)  # координаты точки пересечения
                            # distance_to_intersection_point = distance(curr_node.x, curr_node.y, intersection_point[0], intersection_point[1]) #расстояние
                            # if 1 < distance_to_intersection_point < 3 and intersect_angle > 45:
                            if intersection_point not in new_allowed_intersections:
                                new_allowed_intersections += [intersection_point]

                            try:
                                angle_to_the_next_point_penalty = (
                                            1 - ((np.pi - abs(angle_difference((new_x, new_y), new_angle,
                                                                               (route2[i].x, route2[i].y)))) / np.pi))
                                distance_target_and_next = round(distance(target.x, target.y, route2[i].x, route2[i].y))
                            except IndexError:
                                angle_to_the_next_point_penalty = 0
                                distance_target_and_next = 0

                            if target.x and target.y:  # если target != (0,0)
                                new_g_score = curr_node.g + block.cost + angle_penalty * 20
                                new_h_score = heuristic(new_x, new_y, target.x,
                                                        target.y) + angle_to_the_next_point_penalty * calculate_y(
                                    distance_target_and_next)*enable_next_point_penalty

                            else:
                                new_g_score = curr_node.g + block.cost + angle_penalty * 15
                                new_h_score = heuristic(new_x, new_y, target.x, target.y)

                            new_f_score = new_g_score + new_h_score

                            new_node = Node(
                                x=new_x, y=new_y,
                                start_angle=curr_node.end_angle,
                                end_angle=new_angle,
                                f_score=new_f_score,
                                g_score=new_g_score,
                                parent=curr_node,
                                used_block=('B1', direction),
                                allowed_intersections=new_allowed_intersections,
                                avalible_blocks=deepcopy(current_avalible_elements)
                            )

                            new_node.avalible_blocks[block.type] -= 1
                            heapq.heappush(open_set, new_node)

                        else:
                            new_node = Node(
                                x=new_x, y=new_y,
                                start_angle=curr_node.end_angle,
                                end_angle=new_angle,
                                f_score=new_f_score,
                                g_score=new_g_score,
                                parent=curr_node,
                                used_block=(block.type, direction),
                                allowed_intersections=new_allowed_intersections,
                                avalible_blocks=deepcopy(current_avalible_elements)
                            )

                            new_node.avalible_blocks[block.type] -= 1
                            heapq.heappush(open_set, new_node)

                        if len(open_set) > 300 and enable_limit_nodes:
                            print(f"превышен лимит поиска, точка: {i} ({target.x, target.y})")
                            print(f"start angle: {start_angle}")
                            last_path, start_angle_tmp = curr_node.reconstruct_path()
                            dots = visited_dots(last_path, start_angle_tmp)
                            print(last_path)
                            print([[x.tolist() if isinstance(x, np.ndarray) else x for x in pair] for pair in dots])
                            print()

                            # print(full_dots)
                            # plot_path_on_error(full_dots)
                            sys.exit()

        if found:
            xx = [i[0] for i in path]
            yy = {el: xx.count(el) for el in xx}
            print(f"  Использованные блоки: {yy}")
            print(f"  Остаток: {avalible_elements}")
            print(f"  Построенный маршрут: {path}")
            if debug:
                print(f"остаток(node) {curr_node.avalible_blocks}")
                print([[x.tolist() if isinstance(x, np.ndarray) else x for x in pair] for pair in dots])
                print('allowed intersections: ', allowed_intersections)

            target_p_degree = degrees(
                angle_difference((curr_node.x, curr_node.y), curr_node.end_angle, (target.x, target.y)))
            if not can_achieve_angle(target_p_degree, curr_node.avalible_blocks['T4'], curr_node.avalible_blocks['T8']):
                print(f"Нельзя достигнуть точки. (недостаточно поворотных блоков)")
                break
                # pass
        if not found:
            print(f"\nНевозможно построить маршрут до точки ({target.x},{target.y}) из доступных блоков {avalible_elements})")
            last_path, start_angle_tmp = curr_node.reconstruct_path()
            print(f"Построенный маршрут: {path}")
            if debug:
                dots = visited_dots(last_path, start_angle_tmp)
                print(last_path)
                print([[x.tolist() if isinstance(x, np.ndarray) else x for x in pair] for pair in dots])
                sys.exit()
            break

    return path,start_angle


def visited_dots(order, start_angle):
    x, y, angle = 0, 0, start_angle
    path = []
    path.append([x, y])

    for block, direction in order:
        x, y, angle = move(block, x, y, angle, direction)
        # if debug:
        #     print(f"block {block} | dir {direction:2} | ang {angle*180/np.pi:6.1f} | curr {x:.4f} {y:.4f}") #debug
        path.append([x, y])

    return path


def is_close(px, py, tx, ty, threshold=0.5):
    """Проверка, находится ли точка на расстоянии threshold от целевой точки."""
    return distance(px, py, tx, ty) <= threshold

# Функция для построения маршрута по секции ORDER
def validate_order(elements, order, route, start_ang=0):
    x, y, angle = [0], [0], start_ang
    global threshold
    path = []
    path.append([x, y])
    valid_dir = (1, -1)
    element_count = {e.type: e.count for e in elements}

    visited_points = set()
    print()
    for block, direction in order:
        if block not in element_count:
            raise ValueError(f"Элемент {block} отсутствует.")
        if element_count[block] <= 0:
            raise ValueError(f"Превышен лимит использования элемента {block}.")
        if direction not in valid_dir:
            raise ValueError(f"Направление не может быть {direction} (допустимые значения: {valid_dir}).")

        # Выполнение движения
        try:
            x, y, angle = move(block, x[-1], y[-1], angle, direction,1)
        except ValueError as err:
            print(f"[Ошибка в validate_order]: {err}")

        if debug:
            print(f"block {block} | dir {direction:2} | ang {angle*180/np.pi:6.1f} | curr {x:.4f} {y:.4f}") #debug

        for x_, y_ in zip(x, y):
            path.append([x_, y_])

        element_count[block] -= 1

        # Проверка прохождения через точки ROUTE
        for i, point in enumerate(route):
            if i not in visited_points and is_dot_visited(x, y, point.x, point.y, threshold=threshold):
                visited_points.add(i)

    del path[0]

    # Проверка возврата в (0, 0)
    if not is_dot_visited(x, y, 0, 0, threshold=threshold):
        # raise ValueError("Маршрут не возвращается в начальную точку (0, 0).")
        print("Маршрут не возвращается в начальную точку (0, 0)")
        plot_path_on_error(path)
        sys.exit()

    # Проверка, все ли точки из ROUTE были посещены
    if len(visited_points) != len(route):
        missed_points = [i for i in range(len(route)) if i not in visited_points]
        print(f"Маршрут не прошел через все точки из ROUTE. Пропущены: {[(route[i].x,route[i].y) for i in missed_points]}")
        plot_path_on_error(path)
        sys.exit()
        # raise ValueError(f"Маршрут не прошел через все точки из ROUTE. Пропущены: {[(route[i].x,route[i].y) for i in missed_points]}")

    return np.array(path)


def validate_my_route(elements, order, route, start_ang=0):
    x, y, angle = [0], [0], start_ang
    path = []
    global threshold
    path.append([x, y])
    valid_dir = (1, -1)
    element_count = {e.type: e.count for e in elements}

    visited_points = set()
    print()
    for block, direction in order:
        if block not in element_count:
            print(f"Элемент {block} отсутствует.")
        if element_count[block] <= 0:
            print(f"Превышен лимит использования элемента {block}.")
        if direction not in valid_dir:
            print(f"Направление не может быть {direction} (допустимые значения: {valid_dir}).")

        # Выполнение движения
        x, y, angle = move(block, x[-1], y[-1], angle, direction,1)

        # if debug:
        #     print(f"block {block} | dir {direction:2} | ang {angle*180/np.pi:6.1f} | curr {x[-1]:.4f} {y[-1]:.4f}") #debug

        for x_, y_ in zip(x, y):
            path.append([x_, y_])
        element_count[block] -= 1

        # Проверка прохождения через точки ROUTE
        for i, point in enumerate(route):
            if i not in visited_points and is_dot_visited(x, y, point.x, point.y, threshold=threshold):
                visited_points.add(i)

    del path[0]

    # Проверка возврата в (0, 0)
    if not is_dot_visited(x, y, 0, 0, threshold=threshold):
        print("Маршрут не возвращается в начальную точку (0, 0)")
        plotly_path(np.array(path),route)
        sys.exit()

    # Проверка, все ли точки из ROUTE были посещены
    if len(visited_points) != len(route):
        missed_points = [i for i in range(len(route)) if i not in visited_points]
        print(f"Маршрут не прошел через все точки из ROUTE. Пропущены: {[(route[i].x,route[i].y) for i in missed_points]}")
        plotly_path(np.array(path),route)
        sys.exit()

    print("Маршрут успешно проверен.")
    return np.array(path)

def plot_path_on_error(path):
    x_coords, y_coords = zip(*path)
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker='.',  linestyle='dotted', color='b', label='Маршрут')
    for i, p in enumerate(route[:-1]):
        plt.text(p.x, p.y, f'{i}', fontsize=9, ha='right')
        plt.scatter(p.x, p.y, color='red', label='Точки ROUTE')
    plt.title('маршрут не окончен')
    plt.grid(True)
    plt.show()

# Функция для визуализации маршрута
def plot_route(path, route):
    x_coords, y_coords = zip(*path)
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker='.', linestyle='dotted', color='b', label='Маршрут')

    # Добавляем точки из ROUTE
    route_x = [p.x for p in route]
    route_y = [p.y for p in route]
    plt.scatter(route_x, route_y, color='red', label='Точки ROUTE')

    plt.title('Маршрут с блоками и точками ROUTE')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    for i, p in enumerate(route):
        plt.text(p.x, p.y, f'{i}', fontsize=9, ha='right')
    plt.legend()
    plt.grid(True)
    plt.show()

def price(elements, route, order):
    element_price = {e.type: e.cost for e in elements}
    blocks_price = 0
    route_price = 0

    # Первая часть формулы: -∑(Vt*Nt)
    for block, _ in order:
        blocks_price += element_price[block]

    # Вторая часть формулы: ∑(Vi/(1+di))
    for i, point in enumerate(route[:-1], start=1):
        route_price += point.value / (1 + distance(route[i].x, route[i].y, point.x, point.y))

    return -blocks_price+route_price


def optimize_route(en_aco=0,pop=100,itr=30):
    global route

    if en_aco:
        coords = [[p.x, p.y] for p in route]
        print(f"РАБОТАЮТ МУРАВЬИ (pop={pop}, iters={itr})")

        best_points = ACO(coords[:-1], population=pop, iters=itr)
        result = [route[i] for i in best_points] + [route[-1]]
    else:
        visited = [False] * len(route)
        path = [0]
        visited[0] = True
        for _ in range(len(route[:-1]) - 1):
            last = path[-1]
            nearest = -1
            min_dist = float('inf')

            for i in range(len(route[:-1])):
                if not visited[i]:
                    d = distance(route[last].x, route[last].y, route[i].x, route[i].y)
                    if d < min_dist:
                        min_dist = d
                        nearest = i

            path.append(nearest)
            visited[nearest] = True
        result = [route[i] for i in path]
        result.append(route[-1])
    print('route optimization done')
    route = result
    return route


def count_blocks_usage(elements, path):
    element_used = {e.type: 0 for e in elements}
    element_count = {e.type: e.count for e in elements}

    for block, value in path:
        if block in element_used:
            element_used[block] += 1
        else:
            element_used[block] = 1

    # Выводим результаты
    print("\nИспользованные блоки:")
    for block, used in element_used.items():
        print(f"Блок {block}: {used}\t/ {element_count[block]}")


def change_block_prices(elems):
    my_elems = deepcopy(elems)
    my_prices = {'L1': 1, 'L2': 1, 'L3': 2, 'L4': 3, 'T4': 4, 'T8': 4, 'B1': 6}
    for element in my_elems:
        element.cost = my_prices[element.type]

    return my_elems


def change_block_counts(elems):
    my_elems = deepcopy(elems)
    my_counts = {'L1': 9999, 'L2': 9999, 'L3': 9999, 'L4': 9999, 'T4': 9999, 'T8': 9999, 'B1': 9999}
    for element in my_elems:
        element.count = my_counts[element.type]

    return my_elems


def plotly_path(coords, route):
    annotations = []
    # route_x = [p.x for p in route]
    # route_y = [p.y for p in route]
    route_x = []
    route_y = []

    for index, point in enumerate(route):
        route_x.append(point.x)
        route_y.append(point.y)
        annotations.append(
            Annotation(
                x=point.x,
                y=point.y,
                text=str(index),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(color="red")
            )
        )
    fig = Figure(
        data=[Scatter(x=coords[:, 0], y=coords[:, 1], name='my route', mode='markers'),
              Scatter(x=route_x, y=route_y, name='route point', mode='markers')
              ],
        layout=Layout(xaxis=dict(title='X axis'), yaxis=dict(title='Y axis'), annotations=annotations)
    )

    fig.show()


def start_all_no_dash():
    global route
    global elements
    global debug
    global enable_limit_nodes
    global threshold
    global cell_size
    global enable_next_point_penalty
    global my_prices
    global optimize_with_aco
    global aco_population
    global aco_iterations


    if valid_input and order:
        print("Найден ORDER. Проверка..")
        try:
            coords = validate_order(elements, order, route)
            print("Маршрут из ORDER успешно проверен.")
            print(f"price = {price(elements, route, order)}")
            plotly_path(coords, route)
        except ValueError as err:
            print(f"[Ошибка при проверке ORDER] {err}")
    else:
        print("ORDER не найден")

    if len(route) > 2:
        print(f"\nОптимизация порядка посещения точек")

        if my_prices and unlim_blocks:
            my_elems = change_block_prices(elements)
            my_elems = change_block_counts(my_elems)
        elif my_prices and not unlim_blocks:
            my_elems = change_block_prices(elements)
        elif not my_prices and unlim_blocks:
            my_elems = change_block_counts(elements)
        else:
            my_elems = elements

        print(f"Запуск построения маршрута")
        start = time.time()
        my_path, start_angle = build_route(my_elems, route, data_queue, second_data_queue)
        end = time.time()

        print(f"\nМаршрут построен за {round(end - start)} сек")
        print(f"\nПроверка построенного маршрута..")
        print(f"start angle: {start_angle}")

        coords = validate_my_route(my_elems, my_path, route, start_angle)

        print(f"Price = {round(price(elements, route, my_path),1)}")
        count_blocks_usage(my_elems, my_path)
        plotly_path(coords, route)


def start_all():
    global route
    global elements
    global debug
    global enable_limit_nodes
    global threshold
    global cell_size
    global enable_next_point_penalty
    global my_prices
    global unlim_blocks

    if valid_input and order:
        print("Найден ORDER. Проверка..")
        try:
            coords = validate_order(elements, order, route)
            print("Маршрут из ORDER успешно проверен.")
            print(f"price = {price(elements, route, order)}")
            plotly_path(coords, route)
        except ValueError as err:
            print(f"[Ошибка при проверке ORDER] {err}")
    else:
        print("ORDER не найден")

    if len(route) > 2:
        print(f"\nОптимизация порядка посещения точек")

        if my_prices and unlim_blocks:
            my_elems = change_block_prices(elements)
            my_elems = change_block_counts(my_elems)
        elif my_prices and not unlim_blocks:
            my_elems = change_block_prices(elements)
        elif not my_prices and unlim_blocks:
            my_elems = change_block_counts(elements)
        else:
            my_elems = elements

        print(f"Запуск построения маршрута")
        start = time.time()
        my_path, start_angle = build_route(my_elems, route, data_queue, second_data_queue)
        end = time.time()

        print(f"\nМаршрут построен за {round(end - start)} сек")
        print(f"\nПроверка построенного маршрута..")
        print(f"start angle: {start_angle}")

        coords = validate_my_route(my_elems, my_path, route, start_angle)

        print(f"Price = {round(price(elements, route, my_path),1)}")
        count_blocks_usage(my_elems, my_path)
        plotly_path(coords, route)



def my_dash():
    global app
    global open_set_len
    global cycle_time
    global data_queue
    global second_data_queue

    app.layout = html.Div([  # Оборачиваем график в html.Div
        html.Button("Построить маршрут", id="start-route", n_clicks=0),
        html.Button("Оптимизировать маршрут", id="start-optimize", n_clicks=0),
        html.H1(
            style={'font-size': '24px'},
            children=[
                html.Span("Кол-во вершин: "),
                html.Span(id='live-value', children=open_set_len)
            ]
        ),
        html.H1(
            style={'font-size': '24px'},
            children=[
                html.Span("Время цикла: "),
                html.Span(id='time', children=cycle_time)
            ]
        ),
        dcc.Graph(
            id='live-graph',
            style={'width': '90vw', 'height': '90vh'},  # Задаем стили здесь
        ),
        dcc.Interval(
            id='interval-component',
            interval=100,
            n_intervals=0
        )

    ], style={'margin': 0})  # стили body

    # Callback для запуска маршрута
    @app.callback(
        Output("start-route", "disabled"),
        [Input("start-route", "n_clicks")]
    )
    def start_route(n_clicks):
        if n_clicks > 0:
            # Запуск фонового потока
            threading.Thread(target=start_all, daemon=True).start()
            return True  # Отключить кнопку
        return False

    # Callback для оптимизации маршрута
    @app.callback(
        Output("start-optimize", "disabled"),
        [Input("start-optimize", "n_clicks")]
    )
    def start_optimize(n_clicks):

        global aco_iterations
        global aco_population
        global optimize_with_aco
        global route

        if n_clicks > 0:
            # Запуск фонового потока
            threading.Thread(target=optimize_route, args=(optimize_with_aco, aco_population, aco_iterations),
                             daemon=True).start()
            return True  # Отключить кнопку
        return False

    MAX_POINTS = 1000  # Максимальное количество отображаемых точек

    # Callback для обновления графика
    @app.callback(
        Output('time', 'children'),
        Output('live-value', 'children'),
        Output('live-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        global full_dots
        global route_dots
        global path_points_data
        global route
        global open_set_len
        global cycle_time

        while not data_queue.empty():
            point_data = data_queue.get()
            if isinstance(point_data, list) and len(point_data) == 5:
                point_index, x, y, set_len, time = point_data
                with data_lock:
                    open_set_len = str(set_len)
                    cycle_time = str(round(time * 1e3, 1)) + ' мс'
                # for x, y in zip(x_coords, y_coords):
                full_dots.append([point_index, x, y])
                if len(full_dots) > MAX_POINTS:
                    full_dots.pop(0)
            else:
                print(f"Предупреждение: Пропущены некорректные данные: {point_data}")

        while not second_data_queue.empty():
            additional_point = second_data_queue.get()  # Получаем [dots]
            if isinstance(additional_point, list) and len(additional_point) == 1 and isinstance(additional_point[0],
                                                                                                list):  # Добавляем проверку, что данные имеют формат [[x1,y1], [x2,y2]]
                with data_lock:
                    path_points_data.extend(additional_point[0])  # Добавляем в additional_dots все пары x и y
            else:
                print(f"Предупреждение: Пропущены некорректные данные: {additional_point}")

        with data_lock:
            # full_dots.sort(key=lambda x: x[0])
            if len(full_dots) > 0:
                dots = np.array([[dot[1], dot[2]] for dot in full_dots])
            else:
                dots = np.array([])

            path_points_data.sort(key=lambda x: x[2])
            if len(path_points_data) > 0:
                path_points = np.array([[dot[0], dot[1]] for dot in path_points_data])
            else:
                path_points = np.array([])

        # Формируем данные для route
        route_x = []
        route_y = []
        annotations = []
        for index, point in enumerate(route):
            route_x.append(point.x)
            route_y.append(point.y)
            annotations.append(
                Annotation(
                    x=point.x,
                    y=point.y,
                    text=str(index),
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(color="red")
                )
            )

        # Формируем график
        figure = Figure(
            data=[
                Scatter(x=route_x, y=route_y,
                        mode='lines+markers', opacity=0.3, name='Route Points'),

                Scatter(x=dots[:, 0] if dots.size > 0 else [],
                        y=dots[:, 1] if dots.size > 0 else [],
                        mode='markers', name='node point'),

                Scatter(x=path_points[:, 0] if path_points.size > 0 else [],
                        y=path_points[:, 1] if path_points.size > 0 else [],
                        mode='lines+markers', name='My path')
            ],
            layout=Layout(xaxis=dict(title='X axis'), yaxis=dict(title='Y axis'), annotations=annotations)
        )

        return cycle_time, open_set_len, figure

# ================================================================================
# =====================================MAIN=======================================
# ================================================================================
if __name__ == '__main__':
    """не трогать"""
    data_lock = threading.Lock()
    data_queue = queue.Queue()  # Очередь для передачи данных
    second_data_queue = queue.Queue()  # Вторая очередь
    order = []
    full_dots = []
    route_points = []
    path_points_data = []
    cycle_time = 0
    open_set_len = 0
    valid_input = False

    """можно менять"""
    filename = './test_files/1.txt'
    debug = 0                           # отладочная инфа в консоль
    cell_size = 3                       # размер сетки при проверке пересечений (меньше - больше точность)
    EPSILON = 1e-15                     # при проверке пересечений
    enable_limit_nodes = False          # вкл/выкл лимит кол-ва вершин при поиске пути (при достижении завершает поиск)
    threshold = 0.25                    # расстояние от пути, при котором точки считаются посещенными
    enable_next_point_penalty = False   # [BETA] при приближении к целевой точке, лучшими вариантами считаются те, у кого меньше отклонение угла до следующей целевой точки
    my_prices = False                   # мои цены для блоков при построении (быстрее ищет путь)
    unlim_blocks = True                 # лимит на все блоки 9999
    optimize_with_aco = 1               # вкл оптимизацию порядка маршрутных точек алгоритмом ACO (если выкл - сортировка по расстоянию)
    aco_population = 250                # кол-во муравьев
    aco_iterations = 70                 # итераций (выбирается лучшая)

    """чтение данных из файла"""
    try:
        elements, route, order = read_input(filename)
        valid_input = True
    except ValueError as err:
        print(f"[Ошибка чтения файла] {err}")

    app = Dash(__name__)
    my_dash()

    app.run_server(debug=True) # закомментить если не нужен web dash
    # start_all_no_dash()

