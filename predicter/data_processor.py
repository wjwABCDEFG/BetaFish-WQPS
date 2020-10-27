import numpy as np
from sklearn import preprocessing
from .models import Waterquality


def get_day(Years, Month, first_day):
    ly = False
    if Years % 100 == 0:  # 若年份能被100整除
        if Years % 400 == 0:  # 且能被400整除
            ly = True  # 则是闰年
        else:
            ly = False
    elif Years % 4 == 0:  # 若能被4整除
        ly = True  # 则是闰年
    else:
        ly = False

    if ly == True:  # 若是闰年，则二月为29天
        ms = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        ms = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    Month_days_num = ms[Month]

    second_day = (first_day + 1) % Month_days_num
    if first_day + 1 == Month_days_num:
        second_day = Month_days_num
    third_day = (second_day + 1) % Month_days_num
    if second_day + 1 == Month_days_num:
        third_day = Month_days_num
    return first_day, second_day, third_day, Month_days_num


def load_all_data():
    # 加载每一条数据成array格式
    raw_data = Waterquality.objects.order_by('station', 'date')

    data = []
    data_show=[]
    for waterquality in raw_data:
        # month = waterquality.date.month
        day = waterquality.date.day
        data.append([day, waterquality.do, waterquality.nh3n, waterquality.ph, waterquality.waterTemperature])
        data_show.append([waterquality.date,waterquality.station,waterquality.do, waterquality.nh3n, waterquality.ph, waterquality.waterTemperature])
    # np.set_printoptions(threshold=np.inf)

    # print("load_all_data_data_show",np.array(data_show))
    return np.array(data)


def select_Y(data, feature_num):
    # 选则列数据
    feature_num = feature_num - 1
    Y = np.ones(len(data)).reshape(len(data), 1)
    for i in range(0, len(data)):
        Y[i] = data[i][feature_num]
    # data[:,feature_num] 为什么不这样写
    return Y


def select_X(data, feature_lst):
    # 多少列数据合并  按feature_lst顺序取出数据
    X = select_Y(data, feature_lst[0])
    for i in range(1, len(feature_lst)):
        X = np.hstack((X, select_Y(data, feature_lst[i])))
    return X


def loop_feature(data, feature_num, loop):
    # lstm模型循环一列数据，loop是几天的数据做为一组
    feature = select_Y(data, feature_num)
    # print(feature[-1][0])
    length = len(feature)

    # looped_feature = np.ones([loop, length - loop])
    #     for i in range(0, length - loop):
    # 以上注释是原语句
    # 这里不length - loop+1  最后一个是预测结果  前3个作为数据 第四个作为结果标签，所以最后一个是标签
    looped_feature = np.ones([loop, length - loop])
    for i in range(0, length - loop):
        for j in range(0, loop):
            looped_feature[j][i] = feature[i + j]
    return looped_feature.T


def rearrange(x, loop):
    # lstm模型循环一列结果数据，loop是几天的数据做为一组
    # X本来是2 前三个是月份 后三个数据 经过这个函数=>月份|数据 月份|数据 月份|数据
    length = len(x)
    res = x[:, 0].reshape(length, 1)
    for i in range(0, loop):
        combined = x[:, i].reshape(length, 1)
        for j in range(1, 5):
            combined = np.hstack((combined, x[:, i + loop * j].reshape(length, 1)))
        # The original
        # for j in range(0, 1):
        #     combined = np.hstack((combined, x[:, i + loop].reshape(length, 1)))
        res = np.hstack((res, combined))
    return res[:, 1:]


def transform_x(x, loop=3):
    # lstm模型 数据处理
    x = rearrange(x, loop)  # 557,6 #(1,4,2,5,3,6) =》#(1,4,7,10,13,2,5,8,11,14,3,6,9,12,15)
    # x = x.reshape((x.shape[0], loop, 2))
    x = x.reshape((x.shape[0], loop, 5))
    return x


def generate_sets(data, obj, mode=1, loop=3):
    obj_lst = {'day': 1, 'DO': 2, 'NH3N': 3, 'PH': 4, "WATERTEMPERATURE": 5}
    obj_num = obj_lst[obj]
    # day = loop_feature(data, obj_lst['day'], loop)
    # X = loop_feature(data, obj_num, loop)

    X_all_Factor = []
    for all_obj_index in range(1, 6):
        X_all_Factor.append(loop_feature(data, all_obj_index, loop))
    X = np.hstack((X_all_Factor[0], X_all_Factor[1], X_all_Factor[2], X_all_Factor[3], X_all_Factor[4]))

    Y = select_Y(data, obj_num)
    length = len(data)
    Y = Y[loop:length, :]
    # 合成数据集
    # sets = np.hstack((day,X, Y))
    sets = np.hstack((X, Y))
    # 划分数据集
    if mode == 2:
        training_set = build_training_set(sets, 0.6)
        valid_set = build_valid_set(sets, 0.6, 0.2)
        test_set = build_test_set(sets, 0.2)
        return {'training_set': training_set, 'valid_set': valid_set, 'test_set': test_set}
    training_set = build_training_set(sets, 0.8)
    test_set = build_test_set(sets, 0.2)
    return {'training_set': training_set, 'test_set': test_set}


def build_training_set(sets, percent):
    length = int(len(sets) * percent)
    return sets[0:length, :]


def build_test_set(sets, percent):
    length = int(len(sets) * (1 - percent))
    return sets[length:len(sets), :]


def build_valid_set(sets, training_set_percent, valid_set_percent):
    start = int(len(sets) * training_set_percent)
    end = int(len(sets) * (training_set_percent + valid_set_percent))
    return sets[start:end, :]


def standardize(data):
    # 标准化数据

    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

    return data


def standardize_for_prediction(mean_and_std, data, Month_days_num):
    obj_mean = mean_and_std["mean"]
    obj_std = mean_and_std["std"]

    days_list = [day for day in range(Month_days_num)]
    days_mean = np.mean(days_list)
    days_std = np.std(days_list)

    # month_mean = 6.5
    # month_std = 3.6
    for i in range(0, 3):
        data[0][i] = (data[0][i] - days_mean) / days_std

    for i in range(3, 15):
        data[0][i] = (data[0][i] - obj_mean) / obj_std
    return data


def get_mean_and_std(data, obj):
    obj_lst = {'day': 0, 'DO': 1, 'NH3N': 2, 'PH': 3, "WATERTEMPERATURE": 4}
    obj_num = obj_lst[obj]
    scaler = preprocessing.StandardScaler().fit(data)
    # print("get_mean_and_std_month",scaler.mean_[0],np.sqrt(scaler.var_)[0])  get_mean_and_std_month 6.5 3.452052529534663
    return {'mean': scaler.mean_[obj_num], 'std': np.sqrt(scaler.var_)[obj_num]}


def build_x(data, obj):
    obj_lst = {'month': 0, 'DO': 1, 'NH3N': 2, 'PH': 3}
    date = data[:, obj_lst['month']].T
    feature = data[:, obj_lst[obj]].T
    X = np.hstack((date, feature))
    return X.reshape(1, -1)


def get_x(data):
    col_num = np.size(data, 1)
    x = data[:, 0:col_num - 1]
    return x


def get_y(data):
    col_num = np.size(data, 1)
    y = data[:, col_num - 1]
    return y


def get_last_loop_data(data, loop):
    length = np.size(data, 0)
    return data[length - loop:length, :]


def get_last_months_data(obj, month_num):
    obj = str.lower(obj)
    raw_data = Waterquality.objects.order_by('-station', '-date')[:month_num]
    date = raw_data.values('date')
    obj_data = raw_data.values(obj)
    month = []
    data = []
    for i in range(0, len(date)):
        month.append(date[i]['date'].month)
        data.append(obj_data[i][obj])
    return {'month': month, 'data': data}


def save_data(data):
    water = Waterquality(date=data['date'], do=data['DO'],
                         ph=data['PH'], nh3n=data['NH3N'], station=10)
    water.save()


def get_uploaded_data():
    raw_data = Waterquality.objects.filter(station=10).order_by('date')
    return raw_data


def delete(waterquality_id):
    Waterquality.objects.filter(id=waterquality_id).delete()


def delete_uploaded_data(start, nums):
    raw_data = Waterquality.objects.filter(station=10).order_by('date')
    index = 0
    cnt = 0
    flag = False
    for waterquality in raw_data:
        if index == start:
            flag = True
        if flag:
            waterquality.delete()
            cnt = cnt + 1
        index = index + 1
        if cnt == nums:
            break
