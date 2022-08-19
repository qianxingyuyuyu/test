import pandas as pd
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import scikitplot as skplt

base_elo = 1600  # 初始化elo值
team_elos = {}
team_stats={}
X=[]
y=[]
sum=0
count=0

def initialize_data(Mstat, Ostat, Tstat):
    # 这个函数要完成的任务在于将原始读入的诸多队伍的数据经过修剪，使其变为一个以team为索引的排列的特征数据
    # 丢弃与球队实力无关的统计量
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)
    # 将多个数据通过相同的index：team合并为一个数据
    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
    # 将team作为index的数据返回
    return team_stats1.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]

def calc_elo(win_team, lose_team):
    # winteam, loseteam的输入应为字符串
    # 给出当前两个队伍的elo分数
    winner_rank  = get_elo(win_team)
    loser_rank = get_elo(lose_team)
    # 计算比赛后的等级分，参考elo计算公式
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    # 更新 rank 数值
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_loser_rank = round(loser_rank + (k * (0 - odds)))
    return new_winner_rank, new_loser_rank

def  build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index, row in all_data.iterrows():
        Wteam = row['WTeam']
        Lteam = row['LTeam']
        #获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)
        # 给主场比赛的队伍加上100的elo值
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100
        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]
        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)
        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)
        if skip == 0:
            print('X',X)
            skip = 1
        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank
    return np.nan_to_num(X), y

def predict_winner(team_1, team_2, model):
    features = []
    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)
    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)
    features = np.nan_to_num(features)
    return model.predict_proba([features])

if __name__ == '__main__':
    # 设置导入数据表格文件的地址并读入数据
    Mstat = pd.read_csv('2018-2019Miscellaneous Stats.csv')
    Ostat = pd.read_csv('2018-2019Opponent Per Game Stats.csv')
    Tstat = pd.read_csv('2018-2019Team Per Game Stats.csv')
    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv('2018-2019_result.csv')
    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(X, y)

    skplt.estimators.plot_learning_curve(model, X, y)
    plt.show()

    # 利用10折交叉验证计算训练正确率
    print("1Doing cross-validation..")
    print(cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1).mean())
    # 利用训练好的model在19-20年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv('2019-2020_schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
            # if row['V-PTS']>row['H-PTS']:
            #     count=count+1
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])
            # if row['V-PTS']<row['H-PTS']:
            #     count=count+1
        # sum=sum+1
    with open('19-20_Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)
        print('done.')
    # print(count/sum)