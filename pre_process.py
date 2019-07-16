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
    grid_str = str(int(row)) + '_' + str(int(col))
    return grid_str


def gen_grid_center(x_arr: list, y_arr: list, t_arr: list):
    """
    获取网格中心点
    :return:
    """
    c_x = np.median(x_arr)
    c_y = np.median(y_arr)
    c_t = np.median(t_arr)
    return c_x, c_y, c_t


if __name__ == '__main__':
    # df = pd.read_csv('data/10_grid.csv')
    df = pd.read_csv('data/10.csv')
    delta_x = df['x'].max() - df['x'].min()
    delta_y = df['y'].max() - df['y'].min()
    df['grid'] = df[['x', 'y']].apply(lambda xy: create_grid(100, xy[0], xy[1]), axis=1)
    df.to_csv('data/10_grid.csv', index=None)
    new_id = []
    new_x = []
    new_y = []
    new_t = []
    for g in df.groupby('id'):
        gs = g[1].sort_values('timestamp')
        x_arr = []
        y_arr = []
        t_arr = []
        last_grid = ''
        count = 0
        for i in range(0, len(gs['x'])):
            cur_grid = gs['grid'].values[i]
            if cur_grid == last_grid:
                count += 1
                if count > 1:
                    x_arr.append(gs['x'].values[i])
                    y_arr.append(gs['y'].values[i])
                    t_arr.append(gs['timestamp'].values[i])
            else:
                count = 0
                last_grid = cur_grid
                x, y, t = gen_grid_center(x_arr, y_arr, t_arr)
                new_id.append(g[0])
                new_x.append(x)
                new_y.append(y)
                new_t.append(t)
                x_arr = []
                y_arr = []
                t_arr = []
        x, y, t = gen_grid_center(x_arr, y_arr, t_arr)
        new_id.append(g[0])
        new_x.append(x)
        new_y.append(y)
        new_t.append(t)
    new_df = pd.DataFrame()
    new_df['id'] = new_id
    new_df['x'] = new_x
    new_df['y'] = new_y
    new_df['timestamp'] = new_t
    new_df.to_csv('data/new_10.csv', index=None)
