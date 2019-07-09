import pandas as pd
import numpy as np


def create_grid(side_length: float, delta_x: float, delta_y: float):
    """
    划分网格
    :param side_length:网格边长
    :param delta_x: x轴最大距离差
    :param delta_y: y轴最大距离差
    :return:
    """
    row = delta_y / side_length
    col = delta_x / side_length
    grid_str = str(row) + '_' + str(col)
    return grid_str


def gen_grid_center(x_arr: np.ndarray, y_arr: np.ndarray, t_arr: np.ndarray):
    """
    获取网格中心点
    :return:
    """
    c_x = np.mean(x_arr, axis=1)
    c_y = np.mean(y_arr, axis=1)
    c_t = np.mean(t_arr, axis=1)
    return c_x, c_y, c_t


if __name__ == '__main__':
    df = pd.read_csv('data/10.csv')
    delta_x = df['x'].max - df['x'].min
    delta_y = df['y'].max - df['y'].min
    df['grid'] = df[['x', 'y']].apply(lambda xy: create_grid(200, delta_x, delta_y))
    df.to_csv('data/10_grid.csv', index=None)

    for group in df.groupby('grid'):
        # 网格划分， 降采样
        cx, cy, ct = gen_grid_center(group[1]['x'].values, group[1]['y'].values, group[1]['time'].values)
